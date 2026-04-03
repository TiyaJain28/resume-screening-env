from typing import Tuple, Dict

from .models import Observation
from .tasks import easy_task, medium_task, hard_task


class ResumeScreeningEnv:
    def __init__(self, task: str = "easy"):
        self.task = task
        self.current_index = 0
        self.done = False

        # Dataset
        self.raw_data = [
            {
                "resume": "Python developer with Django and SQL experience",
                "job_role": "Backend Developer",
                "requirements": ["Python", "SQL", "APIs"],
                "label": "shortlist"
            },
            {
                "resume": "Frontend developer skilled in React and CSS",
                "job_role": "Backend Developer",
                "requirements": ["Python", "SQL", "APIs"],
                "label": "reject"
            },
            {
                "resume": "Data scientist with Python and ML experience",
                "job_role": "ML Engineer",
                "requirements": ["Python", "ML", "Statistics"],
                "label": "shortlist"
            }
        ]

        
        if self.task == "easy":
            self.data = easy_task(self.raw_data)
        elif self.task == "medium":
            self.data = medium_task(self.raw_data)
        else:
            self.data = hard_task(self.raw_data)

   
    def reset(self) -> Dict:
        self.current_index = 0
        self.done = False
        return self._get_observation()


    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        if self.done:
            return {}, 0.0, True, {}

        sample = self.data[self.current_index]
        true_label = sample["label"]
        requirements = sample["requirements"]

        reward = 0.0

        
        if self.task == "easy":
            if action.get("decision") == true_label:
                reward += 1.0
            else:
                reward -= 1.0

       
        elif self.task == "medium":
            if action.get("decision") == true_label:
                reward += 0.7
            else:
                reward -= 0.7

            if "match_score" in action:
                expected = 1.0 if true_label == "shortlist" else 0.0
                reward += 0.3 * (1 - abs(action["match_score"] - expected))

     
        else:
            # Skill extraction reward
            if "skills" in action:
                matched = sum(1 for s in action["skills"] if s in requirements)
                reward += 0.3 * (matched / len(requirements))

            # Match score reward
            if "match_score" in action:
                expected = 1.0 if true_label == "shortlist" else 0.0
                reward += 0.3 * (1 - abs(action["match_score"] - expected))

            # Decision reward
            if action.get("decision") == true_label:
                reward += 0.5
            else:
                reward -= 0.5

            # Reason reward
            if "reason" in action and len(action["reason"]) > 10:
                reward += 0.2

        # Move next
        self.current_index += 1

        if self.current_index >= len(self.data):
            self.done = True
            return {}, reward, True, {}

        return self._get_observation(), reward, False, {}

  
    def state(self) -> Dict:
        if self.done:
            return {}
        return self._get_observation()

   
    def _get_observation(self) -> Dict:
        sample = self.data[self.current_index]

        return Observation(
            resume=sample["resume"],
            job_role=sample["job_role"],
            requirements=sample["requirements"]
        ).dict()