package middleware

import (
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
)

// Recovery 返回错误恢复中间件
func Recovery() gin.HandlerFunc {
	return gin.CustomRecovery(func(c *gin.Context, recovered interface{}) {
		if err, ok := recovered.(string); ok {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": fmt.Sprintf("Internal Server Error: %s", err),
			})
		}
		c.AbortWithStatus(http.StatusInternalServerError)
	})
}
