package service

import (
	"agent-api/internal/config"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// ServiceInfo 服务信息
type ServiceInfo struct {
	Name      string
	BaseURL   string
	Healthy   bool
	LastCheck time.Time
}

// ServiceRegistry 服务注册表
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
		Healthy: true, // 初始假设健康
	}
}

// GetService 获取服务信息
func (r *ServiceRegistry) GetService(name string) (*ServiceInfo, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	service, exists := r.services[name]
	if !exists {
		return nil, fmt.Errorf("service '%s' not found", name)
	}
	
	return service, nil
}

// GetServiceURL 获取服务 URL
func (r *ServiceRegistry) GetServiceURL(name string) (string, error) {
	service, err := r.GetService(name)
	if err != nil {
		return "", err
	}
	return service.BaseURL, nil
}

// CheckHealth 检查服务健康状态
func (r *ServiceRegistry) CheckHealth(name string) bool {
	service, err := r.GetService(name)
	if err != nil {
		return false
	}
	
	resp, err := r.client.Get(service.BaseURL + "/health")
	if err != nil || resp.StatusCode != http.StatusOK {
		r.mu.Lock()
		service.Healthy = false
		service.LastCheck = time.Now()
		r.mu.Unlock()
		return false
	}
	
	resp.Body.Close()
	
	r.mu.Lock()
	service.Healthy = true
	service.LastCheck = time.Now()
	r.mu.Unlock()
	
	return true
}

// ListServices 列出所有服务
func (r *ServiceRegistry) ListServices() []*ServiceInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	services := make([]*ServiceInfo, 0, len(r.services))
	for _, svc := range r.services {
		services = append(services, svc)
	}
	
	return services
}
