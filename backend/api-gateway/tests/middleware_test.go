package apitest

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"agent-api/internal/middleware"

	"github.com/gin-gonic/gin"
)

func setupRouter(mwHandler gin.HandlerFunc) *gin.Engine {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(mwHandler)
	r.GET("/test", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})
	return r
}

func TestRateLimitMiddleware_AllowsRequestsUnderLimit(t *testing.T) {
	mw := middleware.RateLimitMiddleware(middleware.RateLimiterConfig{
		Rate:  10,
		Burst: 5,
	})
	r := setupRouter(mw)

	for i := 0; i < 5; i++ {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/test", nil)
		req.RemoteAddr = "192.168.1.1:1234"
		r.ServeHTTP(w, req)

		if w.Code != 200 {
			t.Fatalf("request %d: expected 200, got %d", i, w.Code)
		}
	}
}

func TestRateLimitMiddleware_RejectsRequestsOverBurst(t *testing.T) {
	mw := middleware.RateLimitMiddleware(middleware.RateLimiterConfig{
		Rate:  1,
		Burst: 2,
	})
	r := setupRouter(mw)

	for i := 0; i < 3; i++ {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/test", nil)
		req.RemoteAddr = "192.168.1.2:5678"
		r.ServeHTTP(w, req)

		if i < 2 && w.Code != 200 {
			t.Fatalf("request %d (under burst): expected 200, got %d", i, w.Code)
		}
		if i >= 2 && w.Code != 429 {
			t.Fatalf("request %d (over burst): expected 429, got %d", i, w.Code)
		}
	}
}

func TestRateLimitMiddleware_IPBasedIsolation(t *testing.T) {
	mw := middleware.RateLimitMiddleware(middleware.RateLimiterConfig{
		Rate:  100,
		Burst: 1,
	})
	r := setupRouter(mw)

	w1 := httptest.NewRecorder()
	req1, _ := http.NewRequest("GET", "/test", nil)
	req1.RemoteAddr = "10.0.0.1:1111"
	r.ServeHTTP(w1, req1)

	w2 := httptest.NewRecorder()
	req2, _ := http.NewRequest("GET", "/test", nil)
	req2.RemoteAddr = "10.0.0.2:2222"
	r.ServeHTTP(w2, req2)

	if w1.Code != 200 {
		t.Errorf("IP 1 expected 200, got %d", w1.Code)
	}
	if w2.Code != 200 {
		t.Errorf("IP 2 expected 200, got %d", w2.Code)
	}
}

func TestRateLimitMiddleware_CustomKeyFunc(t *testing.T) {
	called := false
	mw := middleware.RateLimitMiddleware(middleware.RateLimiterConfig{
		Rate:  10,
		Burst: 2,
		KeyFunc: func(c *gin.Context) string {
			called = true
			return c.GetHeader("X-API-Key")
		},
	})
	r := setupRouter(mw)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/test", nil)
	req.Header.Set("X-API-Key", "my-custom-key")
	r.ServeHTTP(w, req)

	if !called {
		t.Error("KeyFunc should have been called")
	}
	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestRateLimitMiddleware_429ResponseBody(t *testing.T) {
	mw := middleware.RateLimitMiddleware(middleware.RateLimiterConfig{
		Rate:  1,
		Burst: 0,
	})
	r := setupRouter(mw)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/test", nil)
	req.RemoteAddr = "1.2.3.4:9999"
	r.ServeHTTP(w, req)

	if w.Code != 429 {
		t.Fatalf("expected 429, got %d", w.Code)
	}

	body := w.Body.String()
	if len(body) == 0 || body[0] != '{' {
		t.Fatalf("response is not JSON: %s", body)
	}
}

func TestRequestLogger_DoesNotPanic(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(middleware.RequestLogger())
	r.GET("/ping", func(c *gin.Context) {
		c.JSON(200, gin.H{"ok": true})
	})

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/ping", nil)
	req.RemoteAddr = "127.0.0.1:8080"

	assertNotPanics(t, func() {
		r.ServeHTTP(w, req)
	})

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func assertNotPanics(t *testing.T, f func()) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("function panicked: %v", r)
		}
	}()
	f()
}
