package com.fulin;

import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.connector.jdbc.JdbcInputFormat;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.types.Row;

public class FlinkResumeJob {

    public static void main(String[] args) throws Exception {
        // Config
        String mysqlHost = System.getenv().getOrDefault("DB_HOST", "localhost");
        int mysqlPort = Integer.parseInt(System.getenv().getOrDefault("DB_PORT", "3306"));
        String mysqlUser = System.getenv().getOrDefault("DB_USER", "root");
        String mysqlPassword = System.getenv().getOrDefault("DB_PASSWORD", "");
        String mysqlDatabase = System.getenv().getOrDefault("RESUME_DB_NAME", "resume_db");

        String redisHost = System.getenv().getOrDefault("REDIS_HOST", "localhost");
        int redisPort = Integer.parseInt(System.getenv().getOrDefault("REDIS_PORT", "6379"));
        int redisDb = Integer.parseInt(System.getenv().getOrDefault("REDIS_DB", "0"));
        String redisPassword = System.getenv().getOrDefault("REDIS_PASSWORD", "");
        int redisTtl = Integer.parseInt(System.getenv().getOrDefault("REDIS_TTL", "604800"));
        String keyPrefix = System.getenv().getOrDefault("RESUME_KEY_PREFIX", "resume:");
        String setKey = System.getenv().getOrDefault("RESUME_SET_KEY", "resumes:ids");

        String jdbcUrl = String.format("jdbc:mysql://%s:%d/%s?useSSL=false&serverTimezone=Asia/Shanghai&allowPublicKeyRetrieval=true",
                mysqlHost, mysqlPort, mysqlDatabase);

        // Flink Environment
        // 注意：当前是批处理模式（Bounded Stream），从 MySQL 读取有限数据后结束
        // 批处理不需要 Checkpoint，因为：
        // 1. 数据源是有限的，失败可以重新运行整个作业
        // 2. 输出到 Redis 是幂等的（相同的 resume_id 会覆盖）
        // 如果未来改为实时流（如监听 MySQL Binlog），再启用 Checkpoint
        org.apache.flink.configuration.Configuration conf = new org.apache.flink.configuration.Configuration();
        conf.setString("execution.target", "local");

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment(conf);
        env.setParallelism(Integer.parseInt(System.getenv().getOrDefault("FLINK_PARALLELISM", "2")));

        // MySQL Source
        JdbcInputFormat jdbcInput = JdbcInputFormat.buildJdbcInputFormat()
                .setDrivername("com.mysql.cj.jdbc.Driver")
                .setDBUrl(jdbcUrl)
                .setUsername(mysqlUser)
                .setPassword(mysqlPassword)
                .setQuery("SELECT resume_id, degree, university_type, work_description, project_description " +
                         "FROM resumes")
                .setRowTypeInfo(new RowTypeInfo(
                        org.apache.flink.api.common.typeinfo.Types.STRING,
                        org.apache.flink.api.common.typeinfo.Types.STRING,
                        org.apache.flink.api.common.typeinfo.Types.STRING,
                        org.apache.flink.api.common.typeinfo.Types.STRING,
                        org.apache.flink.api.common.typeinfo.Types.STRING
                ))
                .finish();

        DataStream<Row> source = env.createInput(jdbcInput);

        // Map to Resume
        DataStream<Resume> resumes = source.map(new MapFunction<Row, Resume>() {
            @Override
            public Resume map(Row row) {
                Resume r = new Resume();
                r.setResumeId((String) row.getField(0));
                r.setDegree((String) row.getField(1));
                r.setUniversityType((String) row.getField(2));
                r.setWorkDescription(ResumeDataCleaner.cleanText((String) row.getField(3)));
                r.setProjectDescription(ResumeDataCleaner.cleanText((String) row.getField(4)));
                return r;
            }
        }).name("MapToResume");

        // Filter invalid
        DataStream<Resume> validResumes = resumes.filter(new FilterFunction<Resume>() {
            @Override
            public boolean filter(Resume resume) {
                return ResumeDataCleaner.validate(resume);
            }
        }).name("FilterValid");

        // Sink to Redis (batch mode with exactly-once semantics)
        // 使用 BatchRedisSink 的批量写入 + 幂等设计保证数据一致性
        // 幂等性基于：相同的 resume_id 写入 Redis 会覆盖旧值
        validResumes.addSink(new BatchRedisSink(redisHost, redisPort, redisPassword,
                redisDb, redisTtl, keyPrefix, setKey))
                .name("BatchRedisSink");

        // Execute
        // 批处理模式：作业完成后自动退出
        // 如果需要定时执行，建议用 Linux cron 或 Airflow 调度
        env.execute("Resume MySQL to Redis Batch Pipeline");
    }
}
