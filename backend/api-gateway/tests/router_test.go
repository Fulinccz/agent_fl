package apitest

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"agent-api/internal/config"
	"agent-api/internal/router"

	"github.com/gin-gonic/gin"
)

func testConfig() *config.Config {
	return &config.Config{
		Server: config.ServerConfig{Port: "8080"},
		Python: config.PythonConfig{
			BaseURL:         "http://localhost:8000",
			AgentPath:       "/api/agent",
			AgentStreamPath: "/api/agent/stream",
			ResumePath:      "/api/resume",
		},
	}
}

func TestSetupRouter_ReturnsEngine(t *testing.T) {
	gin.SetMode(gin.TestMode)
	cfg := testConfig()
	r := router.SetupRouter(cfg)

	if r == nil {
		t.Fatal("SetupRouter returned nil")
	}
}

func TestRootRoute(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := router.SetupRouter(testConfig())

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/", nil)
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}
}

func TestHealthRoute(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := router.SetupRouter(testConfig())

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/health", nil)
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}
}

func TestServicesRoute(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := router.SetupRouter(testConfig())

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/services", nil)
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}
}

func TestAPIAgentRoute_Registered(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := router.SetupRouter(testConfig())

	testCases := []struct {
		method string
		path   string
	}{
		{"POST", "/api/agent"},
		{"POST", "/api/agent/stream"},
		{"POST", "/api/resume/optimize"},
		{"POST", "/api/resume/optimize/stream"},
		{"POST", "/api/chat"},
		{"POST", "/api/chat/stream"},
		{"GET", "/api/chat/sessions"},
	}

	for _, tc := range testCases {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest(tc.method, tc.path, nil)
		r.ServeHTTP(w, req)

		if w.Code == 404 {
			t.Errorf("route %s %s not registered (got 404)", tc.method, tc.path)
		}
	}
}

func TestChatSessionParamRoutes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := router.SetupRouter(testConfig())

	testCases := []struct {
		method string
		path   string
		desc   string
	}{
		{"GET", "/api/chat/sessions/session-123/history", "session history"},
		{"DELETE", "/api/chat/sessions/session-456", "delete session"},
		{"DELETE", "/api/chat/sessions/session-789/clear", "clear session"},
	}

	for _, tc := range testCases {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest(tc.method, tc.path, nil)
		r.ServeHTTP(w, req)

		if w.Code == 404 {
			t.Errorf("%s (%s %s): route not registered (got 404)", tc.desc, tc.method, tc.path)
		}
	}
}

func TestUnknownRoute_Returns404(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := router.SetupRouter(testConfig())

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/nonexistent", nil)
	r.ServeHTTP(w, req)

	if w.Code != 404 {
		t.Errorf("expected 404 for unknown route, got %d", w.Code)
	}
}
