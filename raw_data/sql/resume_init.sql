-- 简历数据库初始化脚本
CREATE DATABASE IF NOT EXISTS resume_db
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE resume_db;

-- 简历信息表（精简版）
CREATE TABLE IF NOT EXISTS resumes (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增ID',
    resume_id VARCHAR(64) NOT NULL UNIQUE COMMENT '简历编号',
    gender VARCHAR(10) COMMENT '性别',
    age INT COMMENT '年龄',
    target_position VARCHAR(100) COMMENT '意向岗位',
    degree VARCHAR(20) COMMENT '学历层次',
    university_type VARCHAR(50) COMMENT '院校类别',
    work_description TEXT COMMENT '工作描述',
    project_description TEXT COMMENT '项目描述',
    source VARCHAR(50) DEFAULT 'tianchi' COMMENT '数据来源',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    INDEX idx_target_position (target_position),
    INDEX idx_degree (degree),
    INDEX idx_source (source)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='简历信息表';
