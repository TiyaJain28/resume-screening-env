import os
import json
import requests

from env.environment import ResumeScreeningEnv


# ---------------- MODEL ACTION ----------------
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
- Extract skills
- Give match_score (0 to 1)
- Decide shortlist or reject
- Give reason

Return STRICT JSON ONLY:
{{
  "skills": ["Python"],
  "match_score": 0.8,
  "decision": "shortlist",
  "reason": "Matches requirements"
}}
"""

    try:
        response = requests.post(
            os.environ["API_BASE_URL"] + "/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['API_KEY']}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 150
            }
        )

        result = response.json()

        text = result["choices"][0]["message"]["content"]

        # Extract JSON safely
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

        # Normalize score to (0,1)
        score = total_reward / (steps + 1)

        if score <= 0:
            score = 0.01
        elif score >= 1:
            score = 0.99

        print(f"[END] task={task} score={score} steps={steps}", flush=True)


if __name__ == "__main__":
    main()
