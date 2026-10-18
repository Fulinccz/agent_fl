package service

import (
	"agent-api/internal/config"
	"agent-api/pkg/logger"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// ServiceInfo 服务信息
type ServiceInfo struct {
	Name      string    `json:"name"`
	BaseURL   string    `json:"base_url"`
	Healthy   bool      `json:"healthy"`
	LastCheck time.Time `json:"last_check"`
}

// ServiceRegistry 服务注册表（静态注册 + 后台健康探测）
type ServiceRegistry struct {
	mu       sync.RWMutex
	services map[string]*ServiceInfo
	client   *http.Client
	config   *config.Config
}

var (
	globalRegistry *ServiceRegistry
	once           sync.Once
)

// GetRegistry 获取全局服务注册表（单例）
func GetRegistry(cfg *config.Config) *ServiceRegistry {
	once.Do(func() {
		globalRegistry = &ServiceRegistry{
			services: make(map[string]*ServiceInfo),
			client:   &http.Client{Timeout: 5 * time.Second},
			config:   cfg,
		}

		// 注册默认服务
		globalRegistry.Register("python", cfg.Python.BaseURL)
	})

	return globalRegistry
}

// Register 注册服务
func (r *ServiceRegistry) Register(name, baseURL string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.services[name] = &ServiceInfo{
		Name:    name,
		BaseURL: baseURL,
		Healthy: true, // 初始假设健康，首轮探测后更新
	}
}

// GetService 获取服务信息（返回副本指针，避免调用方在锁外读写内部对象造成 data race）
func (r *ServiceRegistry) GetService(name string) (*ServiceInfo, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	service, exists := r.services[name]
	if !exists {
		return nil, fmt.Errorf("service '%s' not found", name)
	}

	cp := *service
	return &cp, nil
}

// GetServiceURL 获取服务 URL
func (r *ServiceRegistry) GetServiceURL(name string) (string, error) {
	service, err := r.GetService(name)
	if err != nil {
		return "", err
	}
	return service.BaseURL, nil
}

// CheckHealth 检查服务健康状态并更新注册表
func (r *ServiceRegistry) CheckHealth(name string) bool {
	service, err := r.GetService(name)
	if err != nil {
		return false
	}

	healthy := false
	resp, err := r.client.Get(service.BaseURL + "/health")
	if err == nil {
		healthy = resp.StatusCode == http.StatusOK
		resp.Body.Close()
	}

	r.mu.Lock()
	if current, ok := r.services[name]; ok {
		current.Healthy = healthy
		current.LastCheck = time.Now()
	}
	r.mu.Unlock()

	return healthy
}

// StartHealthCheck 后台周期性探测全部已注册服务的健康状态
func (r *ServiceRegistry) StartHealthCheck(interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for range ticker.C {
		r.mu.RLock()
		names := make([]string, 0, len(r.services))
		for name := range r.services {
			names = append(names, name)
		}
		r.mu.RUnlock()

		for _, name := range names {
			func() {
				defer func() {
					if rec := recover(); rec != nil {
						logger.GetLogger().Warnf("health check panic for %s: %v", name, rec)
					}
				}()
				r.CheckHealth(name)
			}()
		}
	}
}

// ListServices 列出所有服务（返回副本切片）
func (r *ServiceRegistry) ListServices() []ServiceInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()

	services := make([]ServiceInfo, 0, len(r.services))
	for _, svc := range r.services {
		services = append(services, *svc)
	}

	return services
}
