package logger

import (
	"os"
)

// Config 日志配置
type Config struct {
	Level  string
	Output string
	Format string
}

// LoadConfig 从环境变量加载配置
func LoadConfig() Config {
	return Config{
		Level:  getEnv("LOG_LEVEL", "info"),
		Output: getEnv("LOG_OUTPUT", "stdout"),
		Format: getEnv("LOG_FORMAT", "text"),
	}
}

// getEnv 获取环境变量，若不存在则返回默认值
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
