package apitest

import (
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"agent-api/internal/config"
	"agent-api/internal/proxy"

	"github.com/gin-gonic/gin"
)

// startTCPTestServer 在 IPv4 回环上启动测试 HTTP 服务；环境不允许监听时跳过
func startTCPTestServer(t *testing.T, handler http.Handler) *httptest.Server {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Skipf("cannot listen on loopback in this environment: %v", err)
	}
	srv := httptest.NewUnstartedServer(handler)
	srv.Listener = ln
	srv.Start()
	t.Cleanup(srv.Close)
	return srv
}

// startProxyServer 启动一个真实 TCP 回环的网关测试实例（避免 ResponseRecorder 不支持 SSE flush 协议）
func startProxyServer(t *testing.T, baseURL string) *httptest.Server {
	t.Helper()
	gin.SetMode(gin.TestMode)
	proxy.SetConfig(&config.Config{
		Python: config.PythonConfig{BaseURL: baseURL},
	})
	r := gin.New()
	r.Any("/api/*path", proxy.PythonReverseProxy())
	return startTCPTestServer(t, r)
}

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

func TestReverseProxy_ForwardsPathQueryHeadersBody(t *testing.T) {
	var gotMethod, gotPath, gotQuery, gotAuth, gotBody string
	upstream := startTCPTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		gotAuth = r.Header.Get("Authorization")
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.Header().Set("X-Upstream", "python")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"ok":true}`))
	}))

	gw := startProxyServer(t, upstream.URL)

	req, _ := http.NewRequest("POST", gw.URL+"/api/resume/export-docx?format=docx", strings.NewReader("hello-body"))
	req.Header.Set("Authorization", "Bearer token123")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", resp.StatusCode, body)
	}
	if gotMethod != "POST" {
		t.Errorf("method: got %s, want POST", gotMethod)
	}
	if gotPath != "/api/resume/export-docx" {
		t.Errorf("path: got %s", gotPath)
	}
	if gotQuery != "format=docx" {
		t.Errorf("query string lost: got %q", gotQuery)
	}
	if gotAuth != "Bearer token123" {
		t.Errorf("Authorization not forwarded: %q", gotAuth)
	}
	if gotBody != "hello-body" {
		t.Errorf("body: got %q", gotBody)
	}
	if resp.Header.Get("X-Upstream") != "python" {
		t.Error("upstream response header not forwarded")
	}
}

func TestReverseProxy_PropagatesUpstreamStatus(t *testing.T) {
	upstream := startTCPTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte("not found"))
	}))

	gw := startProxyServer(t, upstream.URL)
	resp, err := http.Post(gw.URL+"/api/agent/stream", "application/json", nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("expected 404 from upstream, got %d", resp.StatusCode)
	}
}

func TestReverseProxy_UpstreamUnreachable_Returns502(t *testing.T) {
	// 未监听的本地端口
	gw := startProxyServer(t, "http://127.0.0.1:1")
	resp, err := http.Get(gw.URL + "/api/skill/list")
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("expected 502 for unreachable target, got %d", resp.StatusCode)
	}
	if !strings.Contains(string(body), "Bad Gateway") {
		t.Errorf("502 body should contain generic error, got: %s", body)
	}
}

func TestReverseProxy_RejectsOversizedBody(t *testing.T) {
	upstream := startTCPTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.WriteHeader(http.StatusOK)
	}))

	gw := startProxyServer(t, upstream.URL)
	big := strings.NewReader(strings.Repeat("x", (20<<20)+1))
	resp, err := http.Post(gw.URL+"/api/agent/upload_stream", "application/octet-stream", big)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Errorf("expected 413 for oversized body, got %d", resp.StatusCode)
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
