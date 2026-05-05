package apitest

import (
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"agent-api/internal/config"
	"agent-api/internal/proxy"

	"github.com/gin-gonic/gin"
)

func TestHealthCheck_Returns200(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.GET("/health", proxy.HealthCheck())

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/health", nil)
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}
}

func TestNewProxyRequest_BuildsCorrectURL(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request, _ = http.NewRequest("POST", "/api/test", nil)

	cfg := proxy.ProxyConfig{
		TargetBaseURL: "http://localhost:8000",
		TargetPath:    "/api/chat",
		Method:        "POST",
	}

	proxyReq, err := proxy.NewProxyRequest(c, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expectedURL := "http://localhost:8000/api/chat"
	if proxyReq.URL.String() != expectedURL {
		t.Errorf("URL mismatch: got %s, want %s", proxyReq.URL.String(), expectedURL)
	}
	if proxyReq.Method != "POST" {
		t.Errorf("Method mismatch: got %s, want POST", proxyReq.Method)
	}
}

func TestNewProxyRequest_ForwardsHeaders(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("POST", "/api/test", nil)
	req.Header.Set("X-Custom-Header", "test-value")
	req.Header.Set("Authorization", "Bearer token123")
	c, _ := gin.CreateTestContext(w)
	c.Request = req

	cfg := proxy.ProxyConfig{
		TargetBaseURL: "http://target:8000",
		TargetPath:    "/path",
		Method:        "POST",
	}

	proxyReq, err := proxy.NewProxyRequest(c, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if proxyReq.Header.Get("X-Custom-Header") != "test-value" {
		t.Error("X-Custom-Header not forwarded")
	}
	if proxyReq.Header.Get("Authorization") != "Bearer token123" {
		t.Error("Authorization header not forwarded")
	}
}

func TestExecuteProxy_TargetUnreachable(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request, _ = http.NewRequest("GET", "/api/test", nil)

	cfg := proxy.ProxyConfig{
		TargetBaseURL: "http://nonexistent-host-99999:1",
		TargetPath:    "/test",
		Method:        "GET",
	}

	resp := proxy.ExecuteProxy(c, cfg)
	if resp.StatusCode != 502 {
		t.Errorf("expected 502 for unreachable target, got %d", resp.StatusCode)
	}
	if resp.Error == nil {
		t.Error("expected error for unreachable target")
	}
}

func TestCopyResponse_ErrorReturnsJSON(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)

	proxyResp := &proxy.ProxyResponse{
		StatusCode: 500,
		Error:      http.ErrAbortHandler,
	}

	proxy.CopyResponse(c, proxyResp)

	if w.Code != 500 {
		t.Errorf("expected status 500, got %d", w.Code)
	}
}

func TestSetConfig(t *testing.T) {
	cfg := &config.Config{
		Python: config.PythonConfig{BaseURL: "http://custom:9000"},
	}
	proxy.SetConfig(cfg)

	if proxy.GetAppConfig() == nil {
		t.Fatal("appConfig should not be nil after SetConfig")
	}
	if proxy.GetAppConfig().Python.BaseURL != "http://custom:9000" {
		t.Errorf("BaseURL mismatch: got %s", proxy.GetAppConfig().Python.BaseURL)
	}
}

func TestExecuteStreamProxy_NonexistentPath(t *testing.T) {
	gin.SetMode(gin.TestMode)

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Skipf("skipping: cannot create tcp listener on this system: %v", err)
	}
	baseURL := "http://" + listener.Addr().String()
	go http.Serve(listener, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(404)
		w.Write([]byte("not found"))
	}))
	defer listener.Close()

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request, _ = http.NewRequest("POST", "/api/agent/stream", nil)

	streamCfg := proxy.StreamProxyConfig{
		TargetBaseURL: baseURL,
		TargetPath:    "/test-stream",
		BufferSize:    64,
	}

	proxy.ExecuteStreamProxy(c, streamCfg)

	if w.Code != 404 {
		t.Errorf("expected 404 from upstream, got %d", w.Code)
	}
}
