package proxy

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"time"

	"agent-api/internal/config"
	"agent-api/pkg/logger"

	"github.com/gin-gonic/gin"
)

// maxBodySize 网关层请求体上限。后端业务限制为 10MB，网关提前拦截超大请求。
const maxBodySize = 20 << 20

var (
	appConfig *config.Config

	proxyMu      sync.RWMutex
	cachedProxy  *httputil.ReverseProxy
	cachedTarget string
)

// SetConfig 设置上游服务配置
func SetConfig(cfg *config.Config) {
	appConfig = cfg
}

// GetAppConfig 获取当前配置（用于测试）
func GetAppConfig() *config.Config {
	return appConfig
}

// newTransport 构造带连接级别超时的 Transport（不设置整体请求超时，避免杀死长连接 SSE）
func newTransport() *http.Transport {
	return &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   5 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          100,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   5 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}
}

// buildReverseProxy 基于标准库 httputil.ReverseProxy 构造反向代理。
// ReverseProxy 自动处理：透传 query string、剔除 hop-by-hop 头、
// 客户端断连时取消上游请求（入站 ctx 自动传播）、SSE 即时 flush。
func buildReverseProxy(target *url.URL) *httputil.ReverseProxy {
	director := func(req *http.Request) {
		req.URL.Scheme = target.Scheme
		req.URL.Host = target.Host
		req.Host = target.Host
		if req.Header.Get("X-Forwarded-Proto") == "" {
			req.Header.Set("X-Forwarded-Proto", "http")
		}
	}

	return &httputil.ReverseProxy{
		Director:      director,
		Transport:     newTransport(),
		FlushInterval: -1, // 每次 Write 立即 flush，SSE / JSON-lines 不攒包
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			logger.GetLogger().Warnf("upstream proxy error: %s %s: %v", r.Method, r.URL.Path, err)
			// 请求体超过网关上限（chunked 传输无法预检时在此兜底）
			if strings.Contains(err.Error(), "http: request body too large") {
				w.Header().Set("Content-Type", "application/json; charset=utf-8")
				w.WriteHeader(http.StatusRequestEntityTooLarge)
				_ = json.NewEncoder(w).Encode(gin.H{
					"error":   "Payload Too Large",
					"message": "上传文件大小超过限制（20MB）",
				})
				return
			}
			// 客户端主动断开：返回非标准 499，不向前端写错误体
			if r.Context().Err() == context.Canceled {
				w.WriteHeader(499)
				return
			}
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
			w.WriteHeader(http.StatusBadGateway)
			_ = json.NewEncoder(w).Encode(gin.H{
				"error":   "Bad Gateway",
				"message": "上游服务暂不可用",
			})
		},
	}
}

// getReverseProxy 按 BaseURL 缓存单例 ReverseProxy（复用连接池）
func getReverseProxy() (*httputil.ReverseProxy, error) {
	baseURL := appConfig.Python.BaseURL

	proxyMu.RLock()
	if cachedProxy != nil && cachedTarget == baseURL {
		rp := cachedProxy
		proxyMu.RUnlock()
		return rp, nil
	}
	proxyMu.RUnlock()

	target, err := url.Parse(baseURL)
	if err != nil {
		return nil, err
	}

	rp := buildReverseProxy(target)
	proxyMu.Lock()
	cachedProxy = rp
	cachedTarget = baseURL
	proxyMu.Unlock()
	return rp, nil
}

// PythonReverseProxy 将所有 /api/* 请求透明转发到 Python api-service
func PythonReverseProxy() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 已知 Content-Length 时直接拒绝超大请求，避免读入后再失败
		if c.Request.ContentLength > maxBodySize {
			c.JSON(http.StatusRequestEntityTooLarge, gin.H{
				"error":   "Payload Too Large",
				"message": "上传文件大小超过限制（20MB）",
			})
			return
		}
		// chunked 等无 Content-Length 场景：读取过程中触顶即中断
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxBodySize)

		rp, err := getReverseProxy()
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":   "Bad Gateway Configuration",
				"message": "上游服务地址配置无效",
			})
			return
		}
		rp.ServeHTTP(c.Writer, c.Request)
	}
}

// HealthCheck 网关自身健康检查端点
func HealthCheck() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":    "healthy",
			"service":   "api-gateway",
			"version":   "1.0.0",
			"timestamp": time.Now().Format(time.RFC3339),
		})
	}
}
