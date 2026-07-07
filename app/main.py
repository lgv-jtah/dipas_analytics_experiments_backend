"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import Base, engine
from app.migrations import sync_schema
from app.routes import router

# Create any missing tables, then add any missing nullable columns to
# existing tables (create_all alone won't do the latter).
Base.metadata.create_all(bind=engine)
sync_schema(engine, Base.metadata)

app = FastAPI(
    title="DIPAS Analytics Evaluation API",
    description=(
        "Backend for human evaluation of model-predicted key messages "
        "and stances in participatory planning contributions."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def root():
    return {"message": "DIPAS Analytics Evaluation API – visit /docs for the OpenAPI UI"}
