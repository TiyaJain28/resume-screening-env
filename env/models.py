from pydantic import BaseModel
from typing import List


class Observation(BaseModel):
    resume: str
    job_role: str
    requirements: List[str]


class Action(BaseModel):
    skills: List[str]          # extracted skills
    match_score: float         # 0–1
    decision: str              # shortlist/reject
    reason: str                # explanation


class Reward(BaseModel):
    value: float