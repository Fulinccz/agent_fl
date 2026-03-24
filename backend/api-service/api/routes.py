# api/routes.py
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class AgentRequest(BaseModel):
    query: str

@router.post("/agent")
async def agent_query(request: AgentRequest):
    # Agent逻辑
    return {"response": f"Agent received: {request.query}"}