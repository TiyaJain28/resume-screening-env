from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

from env.environment import ResumeScreeningEnv

app = FastAPI()

# Global env instance
env = ResumeScreeningEnv()


# ---------------- RESET ----------------
@app.post("/reset")
def reset():
    state = env.reset()
    return {"observation": state}


# ---------------- STEP ----------------
class ActionInput(BaseModel):
    skills: list[str]
    match_score: float
    decision: str
    reason: str


@app.post("/step")
def step(action: ActionInput):
    state, reward, done, info = env.step(action.dict())
    return {
        "observation": state,
        "reward": reward,
        "done": done,
        "info": info
    }


# ---------------- STATE ----------------
@app.get("/state")
def state():
    return {"observation": env.state()}