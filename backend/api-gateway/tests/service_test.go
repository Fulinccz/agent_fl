package apitest

import (
	"net"
	"net/http"
	"testing"

	"agent-api/internal/config"
	"agent-api/internal/service"
)

func testServiceConfig() *config.Config {
	return &config.Config{
		Python: config.PythonConfig{BaseURL: "http://localhost:8000"},
	}
}

func TestGetRegistry_Singleton(t *testing.T) {
	cfg := testServiceConfig()
	r1 := service.GetRegistry(cfg)
	r2 := service.GetRegistry(cfg)

	if r1 != r2 {
		t.Error("GetRegistry should return singleton instance")
	}
}

func TestRegisterAndGetService(t *testing.T) {
	cfg := testServiceConfig()
	reg := service.GetRegistry(cfg)

	reg.Register("test-service", "http://test:9000")

	svc, err := reg.GetService("test-service")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if svc.BaseURL != "http://test:9000" {
		t.Errorf("BaseURL mismatch: got %s", svc.BaseURL)
	}
	if svc.Name != "test-service" {
		t.Errorf("Name mismatch: got %s", svc.Name)
	}
}

func TestGetService_NotFound(t *testing.T) {
	cfg := testServiceConfig()
	reg := service.GetRegistry(cfg)

	_, err := reg.GetService("nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent service")
	}
}

func TestListServices(t *testing.T) {
	cfg := testServiceConfig()
	reg := service.GetRegistry(cfg)

	reg.Register("svc-a", "http://a:1000")
	reg.Register("svc-b", "http://b:2000")

	services := reg.ListServices()
	if len(services) < 2 {
		t.Errorf("expected at least 2 services, got %d", len(services))
	}
}

func TestCheckHealth_HealthyService(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Skipf("skipping: cannot create tcp listener on this system: %v", err)
	}
	baseURL := "http://" + listener.Addr().String()
	go http.Serve(listener, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte(`{"status":"healthy"}`))
	}))
	defer listener.Close()

	cfg := &config.Config{Python: config.PythonConfig{BaseURL: baseURL}}
	reg := service.GetRegistry(cfg)

	reg.Register("healthy-svc", baseURL)
	ok := reg.CheckHealth("healthy-svc")

	if !ok {
		t.Error("expected healthy service to return true")
	}
}

func TestCheckHealth_UnhealthyService(t *testing.T) {
	cfg := testServiceConfig()
	reg := service.GetRegistry(cfg)

	reg.Register("unhealthy", "http://nonexistent-host-99999:1")
	ok := reg.CheckHealth("unhealthy")

	if ok {
		t.Error("expected unhealthy service to return false")
	}
}

func TestGetServiceURL(t *testing.T) {
	cfg := testServiceConfig()
	reg := service.GetRegistry(cfg)

	reg.Register("url-test", "http://my-url:5000")
	url, err := reg.GetServiceURL("url-test")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if url != "http://my-url:5000" {
		t.Errorf("URL mismatch: got %s", url)
	}
}

func TestDefaultPythonServiceRegistered(t *testing.T) {
	cfg := testServiceConfig()
	reg := service.GetRegistry(cfg)

	svc, err := reg.GetService("python")
	if err != nil {
		t.Fatalf("python service not registered: %v", err)
	}
	if svc.BaseURL != "http://localhost:8000" {
		t.Errorf("python BaseURL: got %s, want http://localhost:8000", svc.BaseURL)
	}
}

func BenchmarkCheckHealth(b *testing.B) {
	listener, _ := net.Listen("tcp", "127.0.0.1:0")
	baseURL := "http://" + listener.Addr().String()
	go http.Serve(listener, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
	}))
	defer listener.Close()

	cfg := &config.Config{Python: config.PythonConfig{BaseURL: baseURL}}
	reg := service.GetRegistry(cfg)
	reg.Register("bench-svc", baseURL)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reg.CheckHealth("bench-svc")
	}
}

func BenchmarkGetService(b *testing.B) {
	cfg := testServiceConfig()
	reg := service.GetRegistry(cfg)
	reg.Register("bench-get", "http://bench:9999")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reg.GetService("bench-get")
	}
}
