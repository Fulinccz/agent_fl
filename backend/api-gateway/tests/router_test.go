package apitest

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"agent-api/internal/config"
	"agent-api/internal/router"

	"github.com/gin-gonic/gin"
)

func testConfig(baseURL string) *config.Config {
	return &config.Config{
		Server: config.ServerConfig{Port: "8080"},
		Python: config.PythonConfig{
			BaseURL:         baseURL,
			AgentPath:       "/api/agent",
			AgentStreamPath: "/api/agent/stream",
			ResumePath:      "/api/resume",
		},
	}
}

// echoUpstream 固定返回 200 的模拟上游
func echoUpstream(t *testing.T) *httptest.Server {
	return startTCPTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"proxied":true}`))
	}))
}

func TestSetupRouter_ReturnsEngine(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := router.SetupRouter(testConfig("http://localhost:8000"))
	if r == nil {
		t.Fatal("SetupRouter returned nil")
	}
}

func TestRootRoute(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := router.SetupRouter(testConfig("http://localhost:8000"))

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/", nil)
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}
}

func TestHealthRoute(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := router.SetupRouter(testConfig("http://localhost:8000"))

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/health", nil)
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}
}

func TestServicesRoute(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := router.SetupRouter(testConfig("http://localhost:8000"))

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/services", nil)
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}
}

// TestAPIRoutes_ForwardToUpstream 覆盖后端真实存在的全部关键路由（真实 TCP 回环）
func TestAPIRoutes_ForwardToUpstream(t *testing.T) {
	gin.SetMode(gin.TestMode)
	upstream := echoUpstream(t)

	r := router.SetupRouter(testConfig(upstream.URL))
	gw := startTCPTestServer(t, r)

	testCases := []struct {
		method string
		path   string
	}{
		// 认证（旧网关漏转发，经网关无法登录）
		{"POST", "/api/auth/login"},
		{"POST", "/api/auth/refresh"},
		{"GET", "/api/auth/me"},
		// 对话
		{"POST", "/api/chat"},
		{"POST", "/api/chat/stream"},
		{"GET", "/api/chat/sessions"},
		{"GET", "/api/chat/sessions/session-123/history"},
		{"DELETE", "/api/chat/sessions/session-456"},
		{"DELETE", "/api/chat/sessions/session-789/clear"},
		// 简历（含旧网关 404 的导出端点）
		{"POST", "/api/resume/optimize/stream"},
		{"POST", "/api/resume/export-docx"},
		// 文件上传流式
		{"POST", "/api/agent/upload_stream"},
		// 技能
		{"POST", "/api/skill/execute"},
		{"POST", "/api/skill/execute/auto"},
		{"POST", "/api/skill/route"},
		{"GET", "/api/skill/list"},
	}

	for _, tc := range testCases {
		req, _ := http.NewRequest(tc.method, gw.URL+tc.path, nil)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Errorf("route %s %s: request error: %v", tc.method, tc.path, err)
			continue
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Errorf("route %s %s: expected upstream 200, got %d", tc.method, tc.path, resp.StatusCode)
		}
	}
}

func TestUnknownRoute_Returns404(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := router.SetupRouter(testConfig("http://localhost:8000"))

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/nonexistent", nil)
	r.ServeHTTP(w, req)

	if w.Code != 404 {
		t.Errorf("expected 404 for unknown route, got %d", w.Code)
	}
}
