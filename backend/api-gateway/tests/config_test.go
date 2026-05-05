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
	if cfg.Python.BaseURL != "http://localhost:8001" {
		t.Errorf("default baseURL: got %s, want http://localhost:8001", cfg.Python.BaseURL)
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
