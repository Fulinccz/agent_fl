package apitest

import (
	"os"
	"testing"

	"agent-api/internal/config"
)

func TestLoadConfig_DefaultValues(t *testing.T) {
	os.Clearenv()
	cfg, err := config.LoadConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if cfg.Server.Port != "8080" {
		t.Errorf("default port: got %s, want 8080", cfg.Server.Port)
	}
	if cfg.Python.BaseURL != "http://localhost:8000" {
		t.Errorf("default baseURL: got %s, want http://localhost:8000", cfg.Python.BaseURL)
	}
	if cfg.Python.AgentPath != "/api/agent" {
		t.Errorf("default agentPath: got %s, want /api/agent", cfg.Python.AgentPath)
	}
	if cfg.Python.ResumePath != "/api/resume" {
		t.Errorf("default resumePath: got %s, want /api/resume", cfg.Python.ResumePath)
	}
}

func TestConfig_StructFields(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{Port: "3000"},
		Python: config.PythonConfig{
			BaseURL:         "http://service:5000",
			AgentPath:       "/api/v1/agent",
			AgentStreamPath: "/api/v1/agent/stream",
			ResumePath:      "/api/v1/resume",
		},
	}

	if cfg.Server.Port != "3000" {
		t.Error("Server.Port mismatch")
	}
	if cfg.Python.BaseURL != "http://service:5000" {
		t.Error("Python.BaseURL mismatch")
	}
	if cfg.Python.AgentPath != "/api/v1/agent" {
		t.Error("Python.AgentPath mismatch")
	}
	if cfg.Python.ResumePath != "/api/v1/resume" {
		t.Error("Python.ResumePath mismatch")
	}
}

func TestLoadConfig_MissingFileUsesDefaults(t *testing.T) {
	os.Clearenv()
	cfg, err := config.LoadConfig()
	if err != nil {
		t.Fatalf("unexpected error with missing config: %v", err)
	}
	if cfg == nil {
		t.Fatal("config should not be nil")
	}
}

// 嵌套键必须经 EnvKeyReplacer 映射到下划线环境变量名，
// 否则 docker-compose 注入的 PYTHON_BASEURL 会被静默忽略（容器内回退 localhost 打回自身）
func TestLoadConfig_EnvOverride(t *testing.T) {
	os.Clearenv()
	t.Setenv("PYTHON_BASEURL", "http://python-service:9000")
	t.Setenv("SERVER_PORT", "9090")

	cfg, err := config.LoadConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Python.BaseURL != "http://python-service:9000" {
		t.Errorf("env baseURL: got %s, want http://python-service:9000", cfg.Python.BaseURL)
	}
	if cfg.Server.Port != "9090" {
		t.Errorf("env port: got %s, want 9090", cfg.Server.Port)
	}
}
