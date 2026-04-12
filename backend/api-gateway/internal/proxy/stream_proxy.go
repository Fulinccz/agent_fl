package proxy

import (
	"agent-api/internal/config"
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

// 全局配置变量
var appConfig *config.Config

// SetConfig 设置配置
func SetConfig(cfg *config.Config) {
	appConfig = cfg
}

// StreamProxyConfig 流式代理配置
type StreamProxyConfig struct {
	TargetBaseURL string
	TargetPath    string
	BufferSize    int // 缓冲区大小（默认 4096）
}

// PythonStreamProxy 返回流式代理到 Python 服务的中间件
func PythonStreamProxy() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 根据请求路径动态选择目标路径
		targetPath := c.Request.URL.Path

		config := StreamProxyConfig{
			TargetBaseURL: appConfig.Python.BaseURL,
			TargetPath:    targetPath,
			BufferSize:    4096,
		}

		ExecuteStreamProxy(c, config)
	}
}

// ExecuteStreamProxy 执行流式代理
func ExecuteStreamProxy(c *gin.Context, config StreamProxyConfig) {
	targetURL := fmt.Sprintf("%s%s", config.TargetBaseURL, config.TargetPath)

	// 创建代理请求
	proxyReq, err := http.NewRequest(c.Request.Method, targetURL, c.Request.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to create proxy request",
			"details": err.Error(),
		})
		return
	}

	// 复制请求头
	for key, values := range c.Request.Header {
		for _, value := range values {
			proxyReq.Header.Add(key, value)
		}
	}

	// 创建可取消的上下文
	ctx, cancel := context.WithCancel(c.Request.Context())
	defer cancel()
	proxyReq = proxyReq.WithContext(ctx)

	// 监听客户端中止请求
	go func() {
		<-c.Request.Context().Done()
		cancel()
	}()

	// 发送请求
	client := &http.Client{}
	resp, err := client.Do(proxyReq)
	if err != nil {
		// 检查是否是上下文取消错误（客户端中止请求）
		if err == context.Canceled || strings.Contains(err.Error(), "context canceled") {
			return
		}

		c.JSON(http.StatusInternalServerError, gin.H{
			"error":      "Failed to connect to Python service",
			"details":    err.Error(),
			"python_url": targetURL,
		})
		return
	}
	defer resp.Body.Close()

	// 复制响应头
	for key, values := range resp.Header {
		for _, value := range values {
			c.Header(key, value)
		}
	}
	c.Status(resp.StatusCode)

	// 流式复制响应体
	bufSize := config.BufferSize
	if bufSize <= 0 {
		bufSize = 4096
	}
	buf := make([]byte, bufSize)

	for {
		// 检查客户端是否已经中止请求
		select {
		case <-ctx.Done():
			return
		default:
		}

		n, err := resp.Body.Read(buf)
		if n > 0 {
			c.Writer.Write(buf[:n])
			c.Writer.Flush()
		}
		if err != nil {
			break
		}
	}
}

// HealthCheck 健康检查端点
func HealthCheck() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status":    "healthy",
			"service":   "api-gateway",
			"version":   "1.0.0",
			"timestamp": getTimestamp(),
		})
	}
}

// getTimestamp 获取当前时间戳字符串
func getTimestamp() string {
	return time.Now().Format(time.RFC3339)
}
