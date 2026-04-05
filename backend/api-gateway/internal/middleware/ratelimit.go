package middleware

import (
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

// RateLimiterConfig 限流配置
type RateLimiterConfig struct {
	Rate    rate.Limit // 每秒请求数
	Burst   int        // 突发请求数
	KeyFunc func(*gin.Context) string
}

// RateLimitMiddleware 返回限流中间件
func RateLimitMiddleware(config RateLimiterConfig) gin.HandlerFunc {
	limiter := rate.NewLimiter(config.Rate, config.Burst)

	return func(c *gin.Context) {
		if config.KeyFunc != nil {
			_ = config.KeyFunc(c)
		}

		if !limiter.Allow() {
			c.JSON(429, gin.H{
				"error":                 "Too many requests",
				"retry_after":           1,
				"rate_limit_per_second": config.Rate,
			})
			c.Abort()
			return
		}

		c.Next()
	}
}

// RequestLogger 请求日志中间件
func RequestLogger() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path

		c.Next()

		latency := time.Since(start)
		statusCode := c.Writer.Status()

		// 记录请求信息
		c.Set("request_info", map[string]interface{}{
			"path":      path,
			"method":    c.Request.Method,
			"status":    statusCode,
			"latency":   latency.String(),
			"client_ip": c.ClientIP(),
		})
	}
}
