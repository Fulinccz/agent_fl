package router

import (
	"net/http"
	"time"

	"agent-api/internal/config"
	"agent-api/internal/middleware"
	"agent-api/internal/proxy"
	"agent-api/internal/service"

	"github.com/gin-gonic/gin"
)

// SetupRouter 设置路由并返回Gin引擎
func SetupRouter(cfg *config.Config) *gin.Engine {
	// 设置代理配置
	proxy.SetConfig(cfg)

	// 初始化服务注册表
	serviceRegistry := service.GetRegistry(cfg)

	r := gin.New()

	// 注册中间件（按顺序）
	r.Use(middleware.RequestLogger())
	r.Use(middleware.Recovery())
	r.Use(middleware.CORS())

	// 可选：添加限流中间件（开发环境可注释掉）
	// r.Use(middleware.RateLimitMiddleware(middleware.RateLimiterConfig{
	//     Rate:  100,
	//     Burst: 50,
	// }))

	// 健康检查路由
	r.GET("/health", proxy.HealthCheck())

	// 服务发现路由
	r.GET("/services", func(c *gin.Context) {
		services := serviceRegistry.ListServices()
		c.JSON(http.StatusOK, gin.H{
			"services": services,
			"count":    len(services),
		})
	})

	// 根路由
	r.GET("/", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"name":      "Fulin AI API Gateway",
			"version":   "1.0.0",
			"status":    "running",
			"timestamp": time.Now().Unix(),
		})
	})

	// API路由组
	api := r.Group("/api")
	{
		// Agent 相关路由
		agent := api.Group("/agent")
		{
			agent.POST("", proxy.PythonProxy())
			agent.POST("/stream", proxy.PythonStreamProxy())
			agent.POST("/upload", proxy.PythonProxy()) // 文件上传
		}

		// 未来扩展：更多 API 路由
		// resume := api.Group("/resume")
		// {
		//     resume.POST("/parse", ...)
		//     resume.POST("/optimize", ...)
		// }
	}

	return r
}
