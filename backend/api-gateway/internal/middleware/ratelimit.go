package middleware

import (
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

type RateLimiterConfig struct {
	Rate    rate.Limit
	Burst   int
	KeyFunc func(*gin.Context) string
}

func RateLimitMiddleware(config RateLimiterConfig) gin.HandlerFunc {
	type visitor struct {
		limiter  *rate.Limiter
		lastSeen int64
	}

	var (
		mu       sync.Mutex
		visitors = make(map[string]*visitor)
	)

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
		v.lastSeen = 1
		limiter := v.limiter
		mu.Unlock()

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

func RequestLogger() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path

		c.Next()

		latency := time.Since(start)
		statusCode := c.Writer.Status()

		c.Set("request_info", map[string]interface{}{
			"path":      path,
			"method":    c.Request.Method,
			"status":    statusCode,
			"latency":   latency.String(),
			"client_ip": c.ClientIP(),
		})
	}
}
