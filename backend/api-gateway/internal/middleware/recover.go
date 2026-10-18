package middleware

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net/http"
	"runtime/debug"
	"time"

	"agent-api/pkg/logger"

	"github.com/gin-gonic/gin"
)

// ErrorResponse 统一错误响应结构
type ErrorResponse struct {
	Timestamp string `json:"timestamp"`
	Status    int    `json:"status"`
	Error     string `json:"error"`
	Message   string `json:"message"`
	TraceID   string `json:"trace_id,omitempty"`
	Path      string `json:"path"`
}

// newTraceID 生成 16 字符十六进制链路 ID
func newTraceID() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return time.Now().Format("20060102150405")
	}
	return hex.EncodeToString(b)
}

// Recovery 返回错误恢复中间件。
// panic 堆栈仅记录在服务端日志，响应体不泄漏内部错误细节。
func Recovery() gin.HandlerFunc {
	return gin.CustomRecovery(func(c *gin.Context, recovered interface{}) {
		traceID := c.GetHeader("X-Trace-Id")
		if traceID == "" {
			traceID = newTraceID()
		}

		logger.GetLogger().Errorf(
			"panic recovered: trace=%s path=%s err=%v\n%s",
			traceID, c.Request.URL.Path, recovered, debug.Stack(),
		)

		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Timestamp: time.Now().Format(time.RFC3339),
			Status:    http.StatusInternalServerError,
			Error:     "Internal Server Error",
			Message:   fmt.Sprintf("服务器内部错误（trace_id: %s）", traceID),
			TraceID:   traceID,
			Path:      c.Request.URL.Path,
		})
		c.Abort()
	})
}
