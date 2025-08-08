from fastapi import FastAPI, Depends
from fastapi.security import HTTPBearer
from fastapi.openapi.utils import get_openapi
from routes import router

app = FastAPI()

# Register router
app.include_router(router, prefix="/api/v1/hackrx")

# Optional: Add docs security schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="LLM Document Query System",
        version="0.1.0",
        description="API to query insurance policy documents using LLM.",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method.setdefault("security", []).append({"BearerAuth": []})
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
