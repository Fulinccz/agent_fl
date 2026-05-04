package com.fulin;

import org.apache.flink.streaming.api.functions.sink.legacy.RichSinkFunction;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;
import redis.clients.jedis.Pipeline;
import redis.clients.jedis.Response;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@SuppressWarnings("deprecation")
public class BatchRedisSink extends RichSinkFunction<Resume> {

    private static final Pattern PROJECT_NAME_PATTERN = Pattern.compile("项目背景[：:]\\s*(.*?)[。；;]");
    private static final String PROJECT_SET_KEY = "project_names";
    private static final int BATCH_SIZE = 100;
    private static final int MAX_BUFFER_SIZE = 1000;
    private static final int MAX_RETRY = 3;

    private final String redisHost;
    private final int redisPort;
    private final String redisPassword;
    private final int redisDb;
    private final int ttl;
    private final String keyPrefix;
    private final String setKey;

    private transient JedisPool jedisPool;
    private transient List<Resume> buffer;
    private transient Set<String> batchProjectNames;
    private transient int totalProcessed;
    private transient int totalWritten;
    private transient int totalDeduped;

    public BatchRedisSink(String redisHost, int redisPort, String redisPassword,
                          int redisDb, int ttl, String keyPrefix, String setKey) {
        this.redisHost = redisHost;
        this.redisPort = redisPort;
        this.redisPassword = redisPassword;
        this.redisDb = redisDb;
        this.ttl = ttl;
        this.keyPrefix = keyPrefix;
        this.setKey = setKey;
    }

    @Override
    public void open(org.apache.flink.api.common.functions.OpenContext parameters) {
        JedisPoolConfig poolConfig = new JedisPoolConfig();
        poolConfig.setMaxTotal(20);
        poolConfig.setMaxIdle(10);
        poolConfig.setMinIdle(5);
        poolConfig.setMaxWaitMillis(5000);
        poolConfig.setTestOnBorrow(true);

        if (redisPassword != null && !redisPassword.isEmpty()) {
            jedisPool = new JedisPool(poolConfig, redisHost, redisPort, 5000, redisPassword, redisDb);
        } else {
            jedisPool = new JedisPool(poolConfig, redisHost, redisPort, 5000, null, redisDb);
        }

        buffer = new ArrayList<>();
        batchProjectNames = new HashSet<>();
        totalProcessed = 0;
        totalWritten = 0;
        totalDeduped = 0;
    }

    @Override
    public void invoke(Resume resume, org.apache.flink.streaming.api.functions.sink.legacy.SinkFunction.Context context) {
        if (resume == null || resume.getResumeId() == null) {
            return;
        }

        String projectName = extractProjectName(resume.getProjectDescription());
        if (projectName == null || projectName.isEmpty()) {
            return;
        }

        // Batch internal dedup
        if (batchProjectNames.contains(projectName)) {
            totalDeduped++;
            return;
        }

        buffer.add(resume);
        batchProjectNames.add(projectName);

        // Safety check: flush if buffer too large
        if (buffer.size() >= BATCH_SIZE || buffer.size() >= MAX_BUFFER_SIZE) {
            flushWithRetry();
        }
    }

    private void flushWithRetry() {
        int retry = 0;
        while (retry < MAX_RETRY) {
            try {
                flush();
                return;
            } catch (Exception e) {
                retry++;
                System.err.println("[Batch] Flush failed (attempt " + retry + "/" + MAX_RETRY + "): " + e.getMessage());
                if (retry >= MAX_RETRY) {
                    System.err.println("[Batch] Dropping batch of " + buffer.size() + " records after " + MAX_RETRY + " retries");
                    buffer.clear();
                    batchProjectNames.clear();
                    return;
                }
                try {
                    Thread.sleep(1000 * retry);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
    }

    private void flush() {
        if (buffer.isEmpty()) {
            return;
        }

        try (Jedis jedis = jedisPool.getResource()) {
            // Build project name list
            List<String> projectNames = new ArrayList<>();
            for (Resume r : buffer) {
                String name = extractProjectName(r.getProjectDescription());
                if (name != null) {
                    projectNames.add(name);
                }
            }

            // Batch check Redis using pipeline
            Set<String> existingProjects = new HashSet<>();
            if (!projectNames.isEmpty()) {
                Pipeline checkPipe = jedis.pipelined();
                List<Response<Boolean>> responses = new ArrayList<>();
                for (String name : projectNames) {
                    responses.add(checkPipe.sismember(PROJECT_SET_KEY, name));
                }
                checkPipe.sync();
                for (int i = 0; i < responses.size(); i++) {
                    if (responses.get(i).get()) {
                        existingProjects.add(projectNames.get(i));
                    }
                }
            }

            // Write valid resumes
            Pipeline writePipe = jedis.pipelined();
            int written = 0;
            int redisDeduped = 0;

            for (Resume resume : buffer) {
                String projectName = extractProjectName(resume.getProjectDescription());
                if (projectName == null || existingProjects.contains(projectName)) {
                    if (projectName != null && existingProjects.contains(projectName)) {
                        redisDeduped++;
                    }
                    continue;
                }

                String key = keyPrefix + resume.getResumeId();
                Map<String, String> fields = new HashMap<>();
                fields.put("degree", resume.getDegree() == null ? "" : resume.getDegree());
                fields.put("university_type", resume.getUniversityType() == null ? "" : resume.getUniversityType());
                fields.put("work_description", resume.getWorkDescription() == null ? "" : resume.getWorkDescription());
                fields.put("project_description", resume.getProjectDescription() == null ? "" : resume.getProjectDescription());
                fields.put("hash", computeHash(resume));

                writePipe.hset(key, fields);
                writePipe.expire(key, ttl);
                writePipe.sadd(setKey, resume.getResumeId());
                writePipe.sadd(PROJECT_SET_KEY, projectName);
                written++;
            }

            writePipe.sync();

            totalProcessed += buffer.size();
            totalWritten += written;
            totalDeduped += redisDeduped;

            System.out.println("[Batch] Batch: processed=" + buffer.size() +
                    ", written=" + written +
                    ", batchDeduped=" + (buffer.size() - written - redisDeduped) +
                    ", redisDeduped=" + redisDeduped +
                    ", totalProcessed=" + totalProcessed +
                    ", totalWritten=" + totalWritten +
                    ", totalDeduped=" + totalDeduped);
        }

        buffer.clear();
        batchProjectNames.clear();
    }

    @Override
    public void close() {
        flushWithRetry();
        System.out.println("[Batch] Final stats: processed=" + totalProcessed +
                ", written=" + totalWritten +
                ", deduped=" + totalDeduped);
        if (jedisPool != null) {
            jedisPool.close();
        }
    }

    private String extractProjectName(String projectDescription) {
        if (projectDescription == null || projectDescription.isEmpty()) {
            return null;
        }
        Matcher matcher = PROJECT_NAME_PATTERN.matcher(projectDescription);
        if (matcher.find()) {
            return matcher.group(1).trim();
        }
        return null;
    }

    private String computeHash(Resume resume) {
        String content = (resume.getWorkDescription() == null ? "" : resume.getWorkDescription()) +
                        (resume.getProjectDescription() == null ? "" : resume.getProjectDescription());
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] digest = md.digest(content.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (byte b : digest) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            return String.valueOf(content.hashCode());
        }
    }
}
