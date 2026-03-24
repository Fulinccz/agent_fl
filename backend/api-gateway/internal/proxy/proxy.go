// internal/proxy/python_proxy.go
package proxy

import (
	"fmt"
	"io"
	"net/http"

	"agent-api/internal/config"

	"github.com/gin-gonic/gin"
)

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
