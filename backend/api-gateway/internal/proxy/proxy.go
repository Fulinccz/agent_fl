package proxy

import (
	"fmt"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"
)

// ProxyConfig 代理配置
type ProxyConfig struct {
	TargetBaseURL string
	TargetPath    string
	Method        string
	StreamMode    bool
}

// ProxyResponse 代理响应
type ProxyResponse struct {
	StatusCode int
	Headers    http.Header
	Body       io.ReadCloser
	Error      error
}

// NewProxyRequest 创建代理请求
func NewProxyRequest(c *gin.Context, config ProxyConfig) (*http.Request, error) {
	targetURL := fmt.Sprintf("%s%s", config.TargetBaseURL, config.TargetPath)

	req, err := http.NewRequest(
		config.Method,
		targetURL,
		c.Request.Body,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create proxy request: %w", err)
	}

	for key, values := range c.Request.Header {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	return req, nil
}

// ExecuteProxy 执行代理请求（非流式）
func ExecuteProxy(c *gin.Context, config ProxyConfig) *ProxyResponse {
	proxyReq, err := NewProxyRequest(c, config)
	if err != nil {
		return &ProxyResponse{
			StatusCode: http.StatusInternalServerError,
			Error:      err,
		}
	}

	client := &http.Client{}
	resp, err := client.Do(proxyReq)
	if err != nil {
		return &ProxyResponse{
			StatusCode: http.StatusBadGateway,
			Error:      fmt.Errorf("failed to connect to target service: %w", err),
		}
	}

	return &ProxyResponse{
		StatusCode: resp.StatusCode,
		Headers:    resp.Header,
		Body:       resp.Body,
	}
}

// CopyResponse 复制响应到客户端（非流式）
func CopyResponse(c *gin.Context, proxyResp *ProxyResponse) {
	if proxyResp.Error != nil {
		c.JSON(proxyResp.StatusCode, gin.H{
			"error":   proxyResp.Error.Error(),
			"details": "proxy request failed",
		})
		return
	}
	defer proxyResp.Body.Close()

	for key, values := range proxyResp.Headers {
		for _, value := range values {
			c.Header(key, value)
		}
	}

	c.Status(proxyResp.StatusCode)
	io.Copy(c.Writer, proxyResp.Body)
}

// PythonProxy 返回代理到Python服务的中间件（非流式）
func PythonProxy() gin.HandlerFunc {
	return func(c *gin.Context) {
		config := ProxyConfig{
			TargetBaseURL: appConfig.Python.BaseURL,
			TargetPath:    appConfig.Python.AgentPath,
			Method:        c.Request.Method,
			StreamMode:    false,
		}

		proxyResp := ExecuteProxy(c, config)
		CopyResponse(c, proxyResp)
	}
}
