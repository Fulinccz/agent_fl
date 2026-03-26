// cmd/server/main.go
package main

import (
	"fmt"

	"agent-api/internal/config"
	"agent-api/internal/router"
	"agent-api/pkg/logger"
)

func main() {
	cfg := logger.LoadConfig()
	logger.Init(cfg.Level, logger.GetOutput(cfg.Output), cfg.Format)
	appConfig, err := config.LoadConfig()
	if err != nil {
		logger.GetLogger().Fatalf("Failed to load config: %v", err)
		return
	}

	// 设置路由
	r := router.SetupRouter(appConfig)

	// 启动服务器
	serverPort := fmt.Sprintf(":%s", appConfig.Server.Port)
	logger.GetLogger().Infof("Server running on %s", serverPort)
	if err := r.Run(serverPort); err != nil {
		logger.GetLogger().Fatalf("Failed to start server: %v", err)
	}
}
