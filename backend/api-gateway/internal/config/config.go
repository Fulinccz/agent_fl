package config

import (
	"agent-api/pkg/logger"
	"fmt"

	"github.com/spf13/viper"
)

type Config struct {
	Server ServerConfig
	Python PythonConfig
}

type ServerConfig struct {
	Port string
}

type PythonConfig struct {
	BaseURL   string
	AgentPath string
}

func LoadConfig() (*Config, error) {
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath("./config")
	viper.AddConfigPath("../config")
	viper.AddConfigPath("../../config")

	// 设置默认值（本地开发默认 localhost，Docker 环境可通过环境变量覆盖）
	viper.SetDefault("server.port", "8080")
	viper.SetDefault("python.baseURL", "http://localhost:8000")
	viper.SetDefault("python.agentPath", "/api/agent")

	// 从环境变量读取（PYTHON_BASEURL / PYTHON_AGENTPATH 等）
	viper.AutomaticEnv()

	// 尝试读取配置文件
	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}
		// 配置文件不存在时，使用默认值
		logger.GetLogger().Info("Config file not found, using default values")
	}

	var config Config
	if err := viper.Unmarshal(&config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	return &config, nil
}
