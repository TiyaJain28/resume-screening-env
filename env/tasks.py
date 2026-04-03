from typing import List, Dict


# ---------------- EASY ----------------
def easy_task(data: List[Dict]):
    return [
        {
            "resume": item["resume"],
            "job_role": item["job_role"],
            "requirements": item["requirements"],
            "label": item["label"]
        }
        for item in data
    ]


# ---------------- MEDIUM ----------------
def medium_task(data: List[Dict]):
    return data


# ---------------- HARD ----------------
def hard_task(data: List[Dict]):
    return data