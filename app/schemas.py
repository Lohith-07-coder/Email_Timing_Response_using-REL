from pydantic import BaseModel, ConfigDict, Field


class EmailRequest(BaseModel):
    """Input schema for a single email to be evaluated by the DQN agent."""

    subject: str = Field(..., example="Meeting Tomorrow")
    sender: str = Field(..., example="boss@company.com")
    priority: int = Field(..., ge=1, le=3, description="1=low, 2=medium, 3=high")
    sender_importance: int = Field(
        ..., ge=1, le=3, description="1=promo, 2=normal, 3=academic/gov"
    )
    waiting_time: int = Field(
        ..., ge=0, description="Minutes the email has been waiting"
    )
    workload: int = Field(..., ge=1, le=3, description="1=light, 2=moderate, 3=heavy")
    time_of_day: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")


class PredictionResponse(BaseModel):
    """Output schema returned after the agent selects an action."""

    action_id: int
    action_label: str
    confidence: float
    state_vector: list[float]
    model_version: str
    model_config = ConfigDict(protected_namespaces=())


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    model_config = ConfigDict(protected_namespaces=())


class ModelVersionResponse(BaseModel):
    current_version: str
    available_versions: list[str]
    weights_path: str
    pkl_path: str
    model_config = ConfigDict(protected_namespaces=())
