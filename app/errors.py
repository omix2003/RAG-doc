from fastapi import Request
from fastapi.responses import JSONResponse
from openai import APIError, AuthenticationError, RateLimitError


def register_exception_handlers(app) -> None:
    @app.exception_handler(ValueError)
    async def handle_value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_request", "message": str(exc)},
        )

    @app.exception_handler(AuthenticationError)
    async def handle_openai_auth(_: Request, exc: AuthenticationError) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={"error": "openai_auth_error", "message": str(exc)},
        )

    @app.exception_handler(RateLimitError)
    async def handle_openai_rate_limit(_: Request, exc: RateLimitError) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={"error": "openai_rate_limit", "message": str(exc)},
        )

    @app.exception_handler(APIError)
    async def handle_openai_api(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"error": "openai_api_error", "message": str(exc)},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "message": "Unexpected server error."},
        )
