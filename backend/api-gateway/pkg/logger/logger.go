package logger

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/gin-gonic/gin"
)

// Level 日志级别
type Level int

const (
	DebugLevel Level = iota
	InfoLevel
	WarnLevel
	ErrorLevel
	FatalLevel
)

// Logger 日志接口
type Logger interface {
	Debug(args ...interface{})
	Info(args ...interface{})
	Warn(args ...interface{})
	Error(args ...interface{})
	Fatal(args ...interface{})

	Debugf(format string, args ...interface{})
	Infof(format string, args ...interface{})
	Warnf(format string, args ...interface{})
	Errorf(format string, args ...interface{})
	Fatalf(format string, args ...interface{})

	WithField(key string, value interface{}) Logger
	WithFields(fields map[string]interface{}) Logger
	WithContext(ctx context.Context) Logger
}

// 全局日志实例
var globalLogger Logger

// 基础日志实现
type baseLogger struct {
	level  Level
	output io.Writer
	fields map[string]interface{}
}

// 初始化全局日志
func init() {
	// 默认配置：输出到标准输出，INFO 级别，文本格式
	Init("info", os.Stdout, "text")
}

// Init 初始化日志
func Init(level string, output io.Writer, format string) error {
	lvl := parseLevel(level)
	var logger Logger

	switch format {
	case "json":
		logger = newJSONLogger(lvl, output)
	default:
		logger = newTextLogger(lvl, output)
	}

	globalLogger = logger
	return nil
}

// GetLogger 获取全局日志实例
func GetLogger() Logger {
	return globalLogger
}

// 解析日志级别
func parseLevel(level string) Level {
	switch strings.ToLower(level) {
	case "debug":
		return DebugLevel
	case "info":
		return InfoLevel
	case "warn":
		return WarnLevel
	case "error":
		return ErrorLevel
	case "fatal":
		return FatalLevel
	default:
		return InfoLevel
	}
}

func GetOutput(outputType string) io.Writer {
	switch outputType {
	case "stdout":
		return os.Stdout
	case "stderr":
		return os.Stderr
	default:
		// 默认输出到标准输出
		return os.Stdout
	}
}

// GinLogger 返回 Gin 框架的日志中间件
func GinLogger() gin.HandlerFunc {
	return gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		log := GetLogger()
		log.Infof("[GIN] %s | %3d | %13v | %15s | %s | %s",
			param.TimeStamp.Format("2006/01/02 - 15:04:05"),
			param.StatusCode,
			param.Latency,
			param.ClientIP,
			param.Method,
			param.Path,
		)
		return ""
	})
}

// 文本日志实现
func newTextLogger(level Level, output io.Writer) Logger {
	return &baseLogger{
		level:  level,
		output: output,
		fields: make(map[string]interface{}),
	}
}

// JSON 日志实现（简化版）
func newJSONLogger(level Level, output io.Writer) Logger {
	return &baseLogger{
		level:  level,
		output: output,
		fields: make(map[string]interface{}),
	}
}

// 实现 Logger 接口方法
func (l *baseLogger) Debug(args ...interface{}) {
	if l.level <= DebugLevel {
		fmt.Fprintf(l.output, "[DEBUG] ")
		fmt.Fprintln(l.output, args...)
	}
}

func (l *baseLogger) Info(args ...interface{}) {
	if l.level <= InfoLevel {
		fmt.Fprintf(l.output, "[INFO] ")
		fmt.Fprintln(l.output, args...)
	}
}

func (l *baseLogger) Warn(args ...interface{}) {
	if l.level <= WarnLevel {
		fmt.Fprintf(l.output, "[WARN] ")
		fmt.Fprintln(l.output, args...)
	}
}

func (l *baseLogger) Error(args ...interface{}) {
	if l.level <= ErrorLevel {
		fmt.Fprintf(l.output, "[ERROR] ")
		fmt.Fprintln(l.output, args...)
	}
}

func (l *baseLogger) Fatal(args ...interface{}) {
	if l.level <= FatalLevel {
		fmt.Fprintf(l.output, "[FATAL] ")
		fmt.Fprintln(l.output, args...)
		os.Exit(1)
	}
}

func (l *baseLogger) Debugf(format string, args ...interface{}) {
	if l.level <= DebugLevel {
		fmt.Fprintf(l.output, "[DEBUG] "+format+"\n", args...)
	}
}

func (l *baseLogger) Infof(format string, args ...interface{}) {
	if l.level <= InfoLevel {
		fmt.Fprintf(l.output, "[INFO] "+format+"\n", args...)
	}
}

func (l *baseLogger) Warnf(format string, args ...interface{}) {
	if l.level <= WarnLevel {
		fmt.Fprintf(l.output, "[WARN] "+format+"\n", args...)
	}
}

func (l *baseLogger) Errorf(format string, args ...interface{}) {
	if l.level <= ErrorLevel {
		fmt.Fprintf(l.output, "[ERROR] "+format+"\n", args...)
	}
}

func (l *baseLogger) Fatalf(format string, args ...interface{}) {
	if l.level <= FatalLevel {
		fmt.Fprintf(l.output, "[FATAL] "+format+"\n", args...)
		os.Exit(1)
	}
}

func (l *baseLogger) WithField(key string, value interface{}) Logger {
	newFields := make(map[string]interface{})
	for k, v := range l.fields {
		newFields[k] = v
	}
	newFields[key] = value
	return &baseLogger{
		level:  l.level,
		output: l.output,
		fields: newFields,
	}
}

func (l *baseLogger) WithFields(fields map[string]interface{}) Logger {
	newFields := make(map[string]interface{})
	for k, v := range l.fields {
		newFields[k] = v
	}
	for k, v := range fields {
		newFields[k] = v
	}
	return &baseLogger{
		level:  l.level,
		output: l.output,
		fields: newFields,
	}
}

func (l *baseLogger) WithContext(ctx context.Context) Logger {
	// 从 context 中提取字段（可根据需要扩展）
	return l
}
