class AppError(Exception):
    """应用级别基类错误，用于统一捕获与处理。"""

    def __init__(self, message: str, code: str = "app_error", **context):
        super().__init__(message)
        self.code = code
        self.context = context


class NotFoundError(AppError):
    def __init__(self, message: str = "Resource not found", **context):
        super().__init__(message, code="not_found", **context)


class ValidationError(AppError):
    def __init__(self, message: str = "Validation failed", **context):
        super().__init__(message, code="validation_error", **context)


class ServiceError(AppError):
    def __init__(self, message: str = "Service execution failed", **context):
        super().__init__(message, code="service_error", **context)
