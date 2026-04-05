package proxy

import (
	"agent-api/internal/config"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
)

// PythonStreamProxy 返回流式代理到Python服务的中间件
func PythonStreamProxy() gin.HandlerFunc {
	return func(c *gin.Context) {
		pythonURL := fmt.Sprintf("%s%s", appConfig.Python.BaseURL, appConfig.Python.AgentStreamPath)
		proxyReq, err := http.NewRequest(c.Request.Method, pythonURL, c.Request.Body)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":   "Failed to create proxy request",
				"details": err.Error(),
			})
			return
		}
		for key, values := range c.Request.Header {
			for _, value := range values {
				proxyReq.Header.Add(key, value)
			}
		}

		// 创建一个上下文，当客户端中止请求时，也中止对Python服务的请求
		ctx, cancel := context.WithCancel(c.Request.Context())
		defer cancel()
		proxyReq = proxyReq.WithContext(ctx)

		// 监听客户端中止请求
		go func() {
			<-c.Request.Context().Done()
			// 客户端中止请求，取消对Python服务的请求
			cancel()
		}()

		client := &http.Client{}
		resp, err := client.Do(proxyReq)
		if err != nil {
			// 检查是否是上下文取消错误（客户端中止请求）
			if err == context.Canceled || strings.Contains(err.Error(), "context canceled") {
				// 客户端中止请求，直接返回，不返回错误
				return
			}
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":      "Failed to connect to Python service",
				"details":    err.Error(),
				"python_url": pythonURL,
			})
			return
		}
		defer resp.Body.Close()
		for key, values := range resp.Header {
			for _, value := range values {
				c.Header(key, value)
			}
		}
		c.Status(resp.StatusCode)
		// 关键：流式复制响应体
		buf := make([]byte, 4096)
		for {
			// 检查客户端是否已经中止请求
			select {
			case <-ctx.Done():
				// 客户端中止请求，停止复制响应体
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
}

// 全局配置变量
var appConfig *config.Config

// SetConfig 设置配置
func SetConfig(cfg *config.Config) {
	appConfig = cfg
}

// PythonProxy 返回代理到Python服务的中间件
func PythonProxy() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 构建Python服务URL
		pythonURL := fmt.Sprintf("%s%s", appConfig.Python.BaseURL, appConfig.Python.AgentPath)

		// 创建到Python服务的请求
		proxyReq, err := http.NewRequest(c.Request.Method, pythonURL, c.Request.Body)
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

		// 发送请求到Python服务
		client := &http.Client{}
		resp, err := client.Do(proxyReq)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":      "Failed to connect to Python service",
				"details":    err.Error(),
				"python_url": pythonURL,
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

		// 设置响应状态码
		c.Status(resp.StatusCode)

		// 复制响应体
		io.Copy(c.Writer, resp.Body)
	}
}
