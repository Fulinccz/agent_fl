-- 职位(JD)数据库初始化脚本
-- 创建数据库
CREATE DATABASE IF NOT EXISTS job_crawler
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE job_crawler;

-- 职位信息表（精简版）
CREATE TABLE IF NOT EXISTS job_descriptions (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增ID',
    job_id VARCHAR(64) NOT NULL UNIQUE COMMENT '职位ID',
    title VARCHAR(255) NOT NULL COMMENT '职位标题',
    company VARCHAR(255) COMMENT '公司名称',
    salary VARCHAR(100) COMMENT '薪资文本',
    location VARCHAR(255) COMMENT '工作地点',
    tags JSON COMMENT '标签列表',
    jd TEXT COMMENT '职位描述（JD）',
    source VARCHAR(50) COMMENT '数据来源',
    source_url VARCHAR(500) COMMENT '原始链接',
    raw_data JSON COMMENT '原始数据',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    INDEX idx_title (title),
    INDEX idx_company (company),
    INDEX idx_source (source),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='职位信息表';

-- 采集任务日志表
CREATE TABLE IF NOT EXISTS collection_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    task_id VARCHAR(64) NOT NULL UNIQUE COMMENT '任务ID',
    keywords VARCHAR(255) COMMENT '搜索关键词',
    location VARCHAR(255) COMMENT '地点筛选',
    total_fetched INT DEFAULT 0 COMMENT '获取数量',
    total_saved INT DEFAULT 0 COMMENT '保存数量',
    total_skipped INT DEFAULT 0 COMMENT '跳过数量',
    total_failed INT DEFAULT 0 COMMENT '失败数量',
    start_time DATETIME COMMENT '开始时间',
    end_time DATETIME COMMENT '结束时间',
    duration_seconds INT COMMENT '耗时（秒）',
    status VARCHAR(20) DEFAULT 'running' COMMENT '状态',
    error_message TEXT COMMENT '错误信息',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_status (status),
    INDEX idx_start_time (start_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='采集任务日志表';
