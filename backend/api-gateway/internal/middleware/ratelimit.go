package middleware

import (
	"net/http"
	"sync"
	"time"

	"agent-api/pkg/logger"

	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

// RateLimiterConfig 限流配置：每 IP 令牌桶
type RateLimiterConfig struct {
	Rate  rate.Limit // 每秒产生的令牌数
	Burst int        // 桶容量（突发请求数）

	// KeyFunc 自定义限流键（默认按客户端 IP）
	KeyFunc func(*gin.Context) string
}

type visitor struct {
	limiter  *rate.Limiter
	lastSeen int64 // unix nano
}

// RateLimitMiddleware 基于 IP 的令牌桶限流中间件。
// 后台每分钟清理超过 5 分钟不活跃的 visitor，避免 map 无界增长。
func RateLimitMiddleware(config RateLimiterConfig) gin.HandlerFunc {
	var (
		mu       sync.Mutex
		visitors = make(map[string]*visitor)
	)

	// 后台清理协程：每个中间件实例一个，进程生命周期内运行
	go func() {
		ticker := time.NewTicker(time.Minute)
		defer ticker.Stop()
		for range ticker.C {
			cutoff := time.Now().Add(-5 * time.Minute).UnixNano()
			mu.Lock()
			for key, v := range visitors {
				if v.lastSeen < cutoff {
					delete(visitors, key)
				}
			}
			mu.Unlock()
		}
	}()

	return func(c *gin.Context) {
		key := c.ClientIP()
		if config.KeyFunc != nil {
			key = config.KeyFunc(c)
		}

		mu.Lock()
		v, exists := visitors[key]
		if !exists {
			v = &visitor{limiter: rate.NewLimiter(config.Rate, config.Burst)}
			visitors[key] = v
		}
		v.lastSeen = time.Now().UnixNano()
		limiter := v.limiter
		mu.Unlock()

		if !limiter.Allow() {
			c.Header("Retry-After", "1")
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
				"error":   "Too Many Requests",
				"message": "请求过于频繁，请稍后再试",
			})
			return
		}

		c.Next()
	}
}

// RequestLogger 访问日志中间件：请求结束后输出一条结构化访问日志
func RequestLogger() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path
		if raw := c.Request.URL.RawQuery; raw != "" {
			path = path + "?" + raw
		}

		c.Next()

		logger.GetLogger().Infof(
			"access %s %s %d %s ip=%s",
			c.Request.Method,
			path,
			c.Writer.Status(),
			time.Since(start),
			c.ClientIP(),
		)
	}
}
