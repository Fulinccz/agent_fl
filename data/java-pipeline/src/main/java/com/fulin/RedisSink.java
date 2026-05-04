package com.fulin;

import org.apache.flink.streaming.api.functions.sink.legacy.RichSinkFunction;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;
import redis.clients.jedis.Pipeline;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@SuppressWarnings("deprecation")
public class RedisSink extends RichSinkFunction<Resume> {

    private static final Pattern PROJECT_NAME_PATTERN = Pattern.compile("项目背景[：:]\\s*(.*?)[。；;]");
    private static final String PROJECT_SET_KEY = "project_names";

    private final String redisHost;
    private final int redisPort;
    private final String redisPassword;
    private final int redisDb;
    private final int ttl;
    private final String keyPrefix;
    private final String setKey;

    private transient JedisPool jedisPool;

    public RedisSink(String redisHost, int redisPort, String redisPassword,
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
        poolConfig.setMaxTotal(10);
        poolConfig.setMaxIdle(5);

        if (redisPassword != null && !redisPassword.isEmpty()) {
            jedisPool = new JedisPool(poolConfig, redisHost, redisPort, 5000, redisPassword, redisDb);
        } else {
            jedisPool = new JedisPool(poolConfig, redisHost, redisPort, 5000, null, redisDb);
        }
    }

    @Override
    public void invoke(Resume resume, org.apache.flink.streaming.api.functions.sink.legacy.SinkFunction.Context context) {
        if (resume == null || resume.getResumeId() == null) {
            return;
        }

        String projectDesc = resume.getProjectDescription();
        String projectName = extractProjectName(projectDesc);

        // Skip if no project name found
        if (projectName == null || projectName.isEmpty()) {
            return;
        }

        try (Jedis jedis = jedisPool.getResource()) {
            // Check duplicate by project name
            if (jedis.sismember(PROJECT_SET_KEY, projectName)) {
                // Project already exists, skip
                return;
            }

            String key = keyPrefix + resume.getResumeId();
            String hash = computeHash(resume);

            // Save resume
            Pipeline pipe = jedis.pipelined();
            pipe.hset(key, "degree", resume.getDegree() == null ? "" : resume.getDegree());
            pipe.hset(key, "university_type", resume.getUniversityType() == null ? "" : resume.getUniversityType());
            pipe.hset(key, "work_description", resume.getWorkDescription() == null ? "" : resume.getWorkDescription());
            pipe.hset(key, "project_description", projectDesc == null ? "" : projectDesc);
            pipe.hset(key, "hash", hash);
            pipe.expire(key, ttl);
            pipe.sadd(setKey, resume.getResumeId());
            pipe.sadd(PROJECT_SET_KEY, projectName);
            pipe.sync();
        }
    }

    @Override
    public void close() {
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
