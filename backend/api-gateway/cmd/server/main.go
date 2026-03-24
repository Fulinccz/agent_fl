// cmd/server/main.go
package main

import (
	"fmt"
	"io"
	"net/http"
	"time"

	config "agent-api/internal"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

var appConfig *config.Config

func main() {
	// 加载配置，使用 config.LoadConfig()（与包名匹配）
	var err error
	appConfig, err = config.LoadConfig()
	if err != nil {
		fmt.Printf("Failed to load config: %v\n", err)
		return
	}

	// 创建Gin引擎
	r := gin.Default()

	// 添加CORS中间件
	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// 添加错误处理中间件
	r.Use(gin.CustomRecovery(func(c *gin.Context, recovered interface{}) {
		if err, ok := recovered.(string); ok {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": fmt.Sprintf("Internal Server Error: %s", err),
			})
		}
		c.AbortWithStatus(http.StatusInternalServerError)
	}))

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
		// 代理到Python服务的路由
		api.POST("/agent", proxyToPythonService)
	}

	// 启动服务器
	serverPort := fmt.Sprintf(":%s", appConfig.Server.Port)
	fmt.Printf("Server running on %s\n", serverPort)
	if err := r.Run(serverPort); err != nil {
		fmt.Printf("Failed to start server: %v\n", err)
	}
}

// proxyToPythonService 代理请求到Python服务
func proxyToPythonService(c *gin.Context) {
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
