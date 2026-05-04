package middleware

import (
	"fmt"
	"net/http"
	"time"

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

// Recovery 返回错误恢复中间件
func Recovery() gin.HandlerFunc {
	return gin.CustomRecovery(func(c *gin.Context, recovered interface{}) {
		traceID := c.GetHeader("X-Trace-Id")
		if traceID == "" {
			traceID = c.Writer.Header().Get("X-Trace-Id")
		}

		var message string
		if err, ok := recovered.(string); ok {
			message = err
		} else if err, ok := recovered.(error); ok {
			message = err.Error()
		} else {
			message = fmt.Sprintf("unknown error: %v", recovered)
		}

		// 记录错误日志
		c.Error(fmt.Errorf("panic recovered: %s", message))

		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Timestamp: time.Now().Format(time.RFC3339),
			Status:    http.StatusInternalServerError,
			Error:     "Internal Server Error",
			Message:   message,
			TraceID:   traceID,
			Path:      c.Request.URL.Path,
		})
		c.Abort()
	})
}
