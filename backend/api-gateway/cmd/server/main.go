// cmd/server/main.go
package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"agent-api/internal/config"
	"agent-api/internal/router"
	"agent-api/internal/service"
	"agent-api/pkg/logger"

	"github.com/gin-gonic/gin"
)

func main() {
	cfg := logger.LoadConfig()
	logger.Init(cfg.Level, logger.GetOutput(cfg.Output), cfg.Format)

	appConfig, err := config.LoadConfig()
	if err != nil {
		logger.GetLogger().Fatalf("Failed to load config: %v", err)
	}

	// 未显式指定 GIN_MODE 时默认 release，避免调试日志污染生产输出
	if os.Getenv("GIN_MODE") == "" {
		gin.SetMode(gin.ReleaseMode)
	}

	// 设置路由
	r := router.SetupRouter(appConfig)

	// 后台周期性探测上游服务健康状态，结果可通过 /services 查看
	go service.GetRegistry(appConfig).StartHealthCheck(15 * time.Second)

	// 启动服务器（显式 http.Server 以支持优雅关闭）
	serverPort := fmt.Sprintf(":%s", appConfig.Server.Port)
	srv := &http.Server{
		Addr:              serverPort,
		Handler:           r,
		ReadHeaderTimeout: 10 * time.Second, // 防 Slowloris；不设整体读写超时以保护 SSE 长连接
	}

	go func() {
		logger.GetLogger().Infof("Server running on %s, upstream=%s", serverPort, appConfig.Python.BaseURL)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.GetLogger().Fatalf("Failed to start server: %v", err)
		}
	}()

	// 等待终止信号，收到后停止接收新连接并排在途请求（SSE 流自然完结或超时截断）
	quit, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	<-quit.Done()

	logger.GetLogger().Info("Shutting down, draining in-flight requests...")
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		logger.GetLogger().Warnf("Graceful shutdown timed out: %v", err)
	}
	logger.GetLogger().Info("Server stopped")
}
