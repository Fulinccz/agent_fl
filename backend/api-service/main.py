# main.py
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from agents.registry import get_agent

app = FastAPI()


class AgentRequest(BaseModel):
    query: str
    # 可选：选择模型提供方，支持 local/online/openai
    provider: Optional[str] = "local"
    # 可选：传给模型的具体模型名，例如 gpt-3.5-turbo
    model: Optional[str] = None


@app.post("/api/agent")
async def agent_query(request: AgentRequest):
    # 通过 provider 路由到不同的模型实现（本地/线上）
    agent = get_agent(provider=request.provider, model=request.model)
    response = agent.generate(request.query)
    return {"response": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)