
 # resume-screening-env
#it is aligned with all pre-submission checklist
 Resume Screening OpenEnv Environment
📌 Overview

This project implements a real-world AI environment for resume screening, where an agent evaluates candidates for a job role.

It follows the OpenEnv specification and supports:

Multi-step reasoning (skills extraction → scoring → decision → explanation)
Reward-based learning
Easy → Medium → Hard tasks
Live Demo (Hugging Face Space)

 
 






 Note: The root URL (/) may show "Not Found" — this is expected.
Please use /docs to interact with the API.

API Endpoints
Reset Environment
POST /reset

Returns initial observation:

{
  "observation": {
    "resume": "...",
    "job_role": "...",
    "requirements": [...]
  }
}
🔹 Take Step
POST /step

Input:

{
  "skills": ["Python", "SQL"],
  "match_score": 0.8,
  "decision": "shortlist",
  "reason": "Strong backend skills"
}

Output:

{
  "observation": {...},
  "reward": 1.2,
  "done": false,
  "info": {}
}
 Get Current State
GET /state
 Task Design
- Easy
Only decision (shortlist / reject)
- Medium
Decision + match score
- Hard
Extract skills
Compute match score
Decide
Provide explanation
 Reward Function

The reward is dense and multi-step:

Skill match → +0.3
Match score accuracy → +0.3
Correct decision → +0.5
Explanation quality → +0.2
.Inference

The baseline agent:

Uses Hugging Face inference API
Generates structured JSON output
Produces reproducible scores

Run locally:

python inference.py
 Docker Support

Build and run:

docker build -t resume-env .
docker run -p 8000:8000 resume-env
📂 Project Structure
.
├── app.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── README.md
└── env/
    ├── environment.py
    ├── models.py
    ├── tasks.py
 Environment Variables

MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
OpenEnv Compliance
