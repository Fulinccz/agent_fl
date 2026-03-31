package router

import (
	"net/http"
	"time"

	"agent-api/internal/config"
	"agent-api/internal/mid"
	"agent-api/internal/proxy"

	"github.com/gin-gonic/gin"
)

// SetupRouter 设置路由并返回Gin引擎
func SetupRouter(cfg *config.Config) *gin.Engine {
	// 设置代理配置
	proxy.SetConfig(cfg)

	r := gin.New()
	// 注册中间件
	r.Use(mid.Logger())
	r.Use(mid.Recovery())
	r.Use(mid.CORS())

	// 健康检查路由
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":    "ok",
			"timestamp": time.Now().Unix(),
		})
	})

	// 根路由
	r.GET("/", func(c *gin.Context) {
		c.String(http.StatusOK, "AI Agent API Service")
	})

	// API路由组
	api := r.Group("/api")
	{
		api.POST("/agent", proxy.PythonProxy())
		api.POST("/agent/stream", proxy.PythonStreamProxy())
	}

	return r
}
