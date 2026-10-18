package com.fulin;

import org.apache.flink.streaming.api.functions.sink.legacy.RichSinkFunction;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;
import redis.clients.jedis.Pipeline;
import redis.clients.jedis.Response;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 批量 Redis Sink（有界批处理作业）。
 *
 * 设计要点：
 * 1. 去重粒度是 resume（resumeId），不是项目名——多份简历可以描述同名项目，
 *    旧实现按项目名去重会丢弃绝大多数简历。
 * 2. 项目名仅作为二级索引写入 project_names 集合；抽不到项目名不影响简历写入。
 * 3. 写入幂等（HSET 覆盖、EXPIRE 重设、SADD 去重），作业可安全重跑。
 * 4. 重试耗尽后抛出异常让作业失败（有界作业可整体重跑），绝不静默丢批。
 */
@SuppressWarnings("deprecation")
public class BatchRedisSink extends RichSinkFunction<Resume> {

    private static final Pattern PROJECT_NAME_PATTERN = Pattern.compile("项目背景[：:]\\s*(.*?)[。；;]");
    private static final String PROJECT_SET_KEY = "project_names";
    private static final int BATCH_SIZE = 100;
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
    private transient Set<String> batchResumeIds;
    private transient int totalProcessed;
    private transient int totalWritten;
    private transient int totalDeduped;
    private transient int totalNoProjectName;

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
        batchResumeIds = new HashSet<>();
        totalProcessed = 0;
        totalWritten = 0;
        totalDeduped = 0;
        totalNoProjectName = 0;
    }

    @Override
    public void invoke(Resume resume, org.apache.flink.streaming.api.functions.sink.legacy.SinkFunction.Context context) {
        if (resume == null || resume.getResumeId() == null || resume.getResumeId().isEmpty()) {
            return;
        }

        // 批内按 resumeId 去重（MySQL 主键级粒度，同名项目的不同简历都保留）
        if (!batchResumeIds.add(resume.getResumeId())) {
            totalDeduped++;
            return;
        }

        buffer.add(resume);
        totalProcessed++;

        if (buffer.size() >= BATCH_SIZE) {
            flushWithRetry();
        }
    }

    private void flushWithRetry() {
        Exception lastError = null;
        for (int retry = 1; retry <= MAX_RETRY; retry++) {
            try {
                flush();
                return;
            } catch (Exception e) {
                lastError = e;
                System.err.println("[Batch] Flush failed (attempt " + retry + "/" + MAX_RETRY + "): " + e.getMessage());
                if (retry < MAX_RETRY) {
                    try {
                        Thread.sleep(1000L * retry);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException("Interrupted while waiting to retry Redis flush", ie);
                    }
                }
            }
        }
        // 有界批处理 + 幂等写入：失败时让作业整体失败重跑，比静默丢数据安全
        throw new RuntimeException(
                "Redis flush failed after " + MAX_RETRY + " retries, " + buffer.size() + " records unwritten",
                lastError);
    }

    private void flush() {
        if (buffer.isEmpty()) {
            return;
        }

        List<Resume> batch = new ArrayList<>(buffer);

        try (Jedis jedis = jedisPool.getResource()) {
            // 批量查询哪些 resumeId 已存在（仅用于统计与省带宽；HSET 本身幂等）
            Set<String> existingIds = new HashSet<>();
            Pipeline checkPipe = jedis.pipelined();
            List<Response<Boolean>> responses = new ArrayList<>();
            for (Resume r : batch) {
                responses.add(checkPipe.sismember(setKey, r.getResumeId()));
            }
            checkPipe.sync();
            for (int i = 0; i < responses.size(); i++) {
                if (Boolean.TRUE.equals(responses.get(i).get())) {
                    existingIds.add(batch.get(i).getResumeId());
                }
            }

            Pipeline writePipe = jedis.pipelined();
            int written = 0;
            int redisDeduped = 0;
            int noProjectName = 0;

            for (Resume resume : batch) {
                if (existingIds.contains(resume.getResumeId())) {
                    redisDeduped++;
                    continue;
                }

                String key = keyPrefix + resume.getResumeId();
                Map<String, String> fields = new HashMap<>();
                fields.put("degree", resume.getDegree() == null ? "" : resume.getDegree());
                fields.put("university_type", resume.getUniversityType() == null ? "" : resume.getUniversityType());
                fields.put("work_description", resume.getWorkDescription() == null ? "" : resume.getWorkDescription());
                fields.put("project_description", resume.getProjectDescription() == null ? "" : resume.getProjectDescription());
                fields.put("hash", resume.getHash());

                writePipe.hset(key, fields);
                writePipe.expire(key, ttl);
                writePipe.sadd(setKey, resume.getResumeId());

                // 项目名只做二级索引；抽不到不丢简历，仅计数
                String projectName = extractProjectName(resume.getProjectDescription());
                if (projectName != null && !projectName.isEmpty()) {
                    writePipe.sadd(PROJECT_SET_KEY, projectName);
                } else {
                    noProjectName++;
                }
                written++;
            }

            writePipe.sync();

            totalWritten += written;
            totalDeduped += redisDeduped;
            totalNoProjectName += noProjectName;

            System.out.println("[Batch] Batch: size=" + batch.size() +
                    ", written=" + written +
                    ", redisDeduped=" + redisDeduped +
                    ", noProjectName=" + noProjectName +
                    ", totalProcessed=" + totalProcessed +
                    ", totalWritten=" + totalWritten +
                    ", totalDeduped=" + totalDeduped);
        }

        // 仅在整批写入成功后清空缓冲（失败时保留，供重试或失败传播）
        buffer.clear();
        batchResumeIds.clear();
    }

    @Override
    public void close() {
        try {
            flushWithRetry();
        } finally {
            System.out.println("[Batch] Final stats: processed=" + totalProcessed +
                    ", written=" + totalWritten +
                    ", deduped=" + totalDeduped +
                    ", noProjectName=" + totalNoProjectName);
            if (jedisPool != null) {
                jedisPool.close();
            }
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
}
