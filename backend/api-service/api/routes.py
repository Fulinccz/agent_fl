# api/routes.py
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from services.agent_service import AgentService
from services.exceptions import AppError
from logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


class AgentRequest(BaseModel):
    query: str
    provider: str = "local"
    model: str | None = None


@router.post("/agent")
async def agent_query(request: AgentRequest):
    try:
        logger.debug("Received agent request: %s", request.dict())
        result = AgentService().generate(request.query, provider=request.provider, model=request.model)
        logger.info("Agent result returned, len=%d", len(result) if isinstance(result, str) else 0)
        return {"response": result}
    except AppError as err:
        logger.warning("Agent business error: %s", err, exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"code": err.code, "message": str(err), "context": err.context})
    except Exception as err:
        logger.error("agent_query uncaught error: %s", err, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
