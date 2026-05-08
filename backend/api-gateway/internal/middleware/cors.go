package middleware

import (
	"os"
	"strings"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// CORS 返回生产级 CORS 中间件
func CORS() gin.HandlerFunc {
	// 从环境变量读取允许的域名，默认只允许本地
	allowOrigins := os.Getenv("ALLOWED_ORIGINS")
	if allowOrigins == "" {
		allowOrigins = "http://localhost,http://localhost:3000,http://localhost:5173"
	}

	origins := strings.Split(allowOrigins, ",")

	return cors.New(cors.Config{
		AllowOrigins:     origins,
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization", "X-Trace-Id", "X-Request-ID"},
		ExposeHeaders:    []string{"Content-Length", "X-Trace-Id", "X-Request-ID"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	})
}

// SecurityHeaders 安全响应头中间件
func SecurityHeaders() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 防止点击劫持
		c.Header("X-Frame-Options", "DENY")
		// 防止 MIME 嗅探
		c.Header("X-Content-Type-Options", "nosniff")
		// XSS 保护
		c.Header("X-XSS-Protection", "1; mode=block")
		// 强制 HTTPS（生产环境）
		c.Header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		// CSP 内容安全策略
		c.Header("Content-Security-Policy", "default-src 'self'")
		// 限制 referrer 泄露
		c.Header("Referrer-Policy", "strict-origin-when-cross-origin")

		c.Next()
	}
}
