import os
import json


from env.environment import ResumeScreeningEnv


# ---------------- ENV VARIABLES ----------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")




# ---------------- MODEL ACTION ----------------
import requests

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
        response = requests.post(
            f"{API_BASE_URL}/{MODEL_NAME}",
            headers={
                "Authorization": f"Bearer {API_KEY}"
            },
            json={
                "inputs": prompt,
                "parameters": {
                    "temperature": 0.2,
                    "max_new_tokens": 150
                }
            }
        )

        result = response.json()

        text = result[0]["generated_text"]

        # extract JSON safely
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

# ---------------- RUN TASK ----------------
def run_task(task_name):
    env = ResumeScreeningEnv(task=task_name)

    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 10:
        action = get_action(state)

        state, reward, done, _ = env.step(action)

        total_reward += reward
        steps += 1

    return total_reward


# ---------------- MAIN ----------------
import os
import json


from env.environment import ResumeScreeningEnv


# ---------------- ENV VARIABLES ----------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")




# ---------------- MODEL ACTION ----------------
import requests

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
        response = requests.post(
            f"{API_BASE_URL}/{MODEL_NAME}",
            headers={
                "Authorization": f"Bearer {API_KEY}"
            },
            json={
                "inputs": prompt,
                "parameters": {
                    "temperature": 0.2,
                    "max_new_tokens": 150
                }
            }
        )

        result = response.json()

        text = result[0]["generated_text"]

        # extract JSON safely
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

# ---------------- RUN TASK ----------------
def run_task(task_name):
    env = ResumeScreeningEnv(task=task_name)

    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 10:
        action = get_action(state)

        state, reward, done, _ = env.step(action)

        total_reward += reward
        steps += 1

    return total_reward


# ---------------- MAIN ----------------
def main():
    print("START")

    tasks = ["easy", "medium", "hard"]
    results = {}

    for task in tasks:
        print(f"STEP: {task}")
        score = run_task(task)
        results[task] = score
        print(f"{task} score: {score:.2f}")

    print("END")
    print(results)


if __name__ == "__main__":
    main()
