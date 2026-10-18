package router

import (
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"agent-api/internal/config"
	"agent-api/internal/middleware"
	"agent-api/internal/proxy"
	"agent-api/internal/service"

	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

// envFloat 读取浮点环境变量，缺省返回默认值
func envFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.ParseFloat(v, 64); err == nil {
			return n
		}
	}
	return def
}

// envInt 读取整型环境变量，缺省返回默认值
func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

// trustedProxies 可信任的代理网段（默认 Docker 内网 + 本机回环）
func trustedProxies() []string {
	if v := os.Getenv("GATEWAY_TRUSTED_PROXIES"); v != "" {
		return strings.Split(v, ",")
	}
	return []string{"172.16.0.0/12", "127.0.0.1/32", "::1/128"}
}

// SetupRouter 设置路由并返回 Gin 引擎
func SetupRouter(cfg *config.Config) *gin.Engine {
	// 设置代理配置
	proxy.SetConfig(cfg)

	// 初始化服务注册表
	serviceRegistry := service.GetRegistry(cfg)

	r := gin.New()

	// 仅信任内网反代写入的 X-Forwarded-For，防止外网伪造客户端 IP 绕过限流
	_ = r.SetTrustedProxies(trustedProxies())

	// 注册中间件（按顺序）
	r.Use(middleware.RequestLogger())
	r.Use(middleware.Recovery())
	r.Use(middleware.SecurityHeaders())
	r.Use(middleware.CORS())
	r.Use(middleware.RateLimitMiddleware(middleware.RateLimiterConfig{
		Rate:  rate.Limit(envFloat("GATEWAY_RATE_LIMIT", 50)),  // 每秒令牌数
		Burst: envInt("GATEWAY_RATE_BURST", 100),               // 突发容量
	}))

	// 健康检查路由
	r.GET("/health", proxy.HealthCheck())

	// 服务发现路由（注册表 + 后台健康探测结果）
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

	// 所有 /api/* 请求统一反向代理到 Python api-service。
	// 网关与后端路由一一对应（含 auth/chat/resume/upload/skill 全部端点），
	// 由 httputil.ReverseProxy 透明转发，流式端点（SSE/JSON-lines）即时 flush。
	r.Any("/api/*path", proxy.PythonReverseProxy())

	// 未匹配路由统一 404
	r.NoRoute(func(c *gin.Context) {
		c.JSON(http.StatusNotFound, gin.H{
			"error": "Not Found",
			"path":  c.Request.URL.Path,
		})
	})

	return r
}
