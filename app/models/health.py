from pydantic import BaseModel, Field


class HealthStatus(BaseModel):
    status: str
    version: str = Field(description="Semantic API version string")
