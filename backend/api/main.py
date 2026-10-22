from contextlib import asynccontextmanager

from config.settings import settings
from exceptions import ConfigurationError, IngestionError
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from logger import _LOGGER_INSTANCE
from logger import GLOBAL_LOGGER as log

from api import agent_router, health_router, rag_router, status_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "Meridian AI starting", mode=settings.app_mode, llm=settings.llm_model_name
    )
    yield  # the app runs here
    # Shutdown (cloud mode only): flush logs to GCS, because Cloud Run's disk disappears
    _LOGGER_INSTANCE.flush_to_gcs()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    description="Multi agent procurement assistant with RAG capabilities. This API is designed to facilitate interactions between agents and a retrieval-augmented generation (RAG) system, enabling efficient information retrieval and processing.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(status_router)
app.include_router(agent_router)
app.include_router(rag_router)


def custom_openapi():
    """Patch OpenAPI schema to 3.0.3 so Swagger UI renders list[UploadFile] as native file pickers."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version="3.0.3",
        description=app.description,
        routes=app.routes,
    )
    schema["openapi"] = "3.0.3"
    for comp in schema.get("components", {}).get("schemas", {}).values():
        for prop in comp.get("properties", {}).values():
            if prop.get("type") == "array" and "items" in prop:
                prop["items"] = {"type": "string", "format": "binary"}
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.exception_handler(ConfigurationError)
async def configuration_error_handler(request: Request, exc: ConfigurationError):
    """A missing setting is not a crash: tell the caller exactly what to set."""
    log.error("Configuration error", error=str(exc))
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(IngestionError)
async def ingestion_error_handler(request: Request, exc: IngestionError):
    """An ingestion error is not a crash: tell the caller exactly what went wrong."""
    log.error("Ingestion error", error=str(exc))
    return JSONResponse(status_code=422, content={"detail": str(exc)})
