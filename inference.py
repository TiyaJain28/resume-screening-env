import os
import json

from env.environment import ResumeScreeningEnv


# ---------------- ENV VARIABLES ----------------
API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")


# ---------------- MODEL ACTION ----------------
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)


def get_action(observation):
    prompt = f"""
You are an HR assistant.

Resume:
{observation['resume']}

Job Role:
{observation['job_role']}

Requirements:
{observation['requirements']}

TASK:
- Extract relevant skills
- Give match_score (0 to 1)
- Decide shortlist or reject
- Give short reason

Return STRICT JSON ONLY:
{{
  "skills": ["Python"],
  "match_score": 0.8,
  "decision": "shortlist",
  "reason": "Matches requirements"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # proxy will handle internally
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150
        )

        text = response.choices[0].message.content

        start = text.find("{")
        end = text.rfind("}") + 1

        return json.loads(text[start:end])

    except Exception:
        return {
            "skills": ["Python"],
            "match_score": 0.5,
            "decision": "reject",
            "reason": "fallback"
        }


# ---------------- MAIN ----------------
def main():
    tasks = ["easy", "medium", "hard"]

    for task in tasks:
        print(f"[START] task={task}", flush=True)

        env = ResumeScreeningEnv(task=task)

        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < 10:
            action = get_action(state)

            state, reward, done, _ = env.step(action)

            total_reward += reward
            steps += 1

            print(f"[STEP] step={steps} reward={reward}", flush=True)

        print(f"[END] task={task} score={total_reward} steps={steps}", flush=True)


if __name__ == "__main__":
    main()
