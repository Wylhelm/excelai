from typing import List, Dict
import openai
import os

class AIMatcher:
    def __init__(self):
        # Configure OpenAI to use LM-Studio's local API
        openai.api_base = "http://localhost:1234/v1"  # Adjust this URL if LM-Studio uses a different port
        openai.api_key = "not-needed"  # LM-Studio doesn't require an API key for local use

    def match_candidates(self, candidates: List[Dict], request: Dict) -> List[Dict]:
        matched_candidates = []
        
        for candidate in candidates:
            score = self._calculate_match_score(candidate, request)
            matched_candidates.append({
                "candidate": candidate,
                "score": score
            })
        
        return sorted(matched_candidates, key=lambda x: x["score"], reverse=True)

    def _calculate_match_score(self, candidate: Dict, request: Dict) -> float:
        # Prepare the input for the LLM
        user_message = f"""You are an AI assistant that calculates match scores between job requests and candidates.

        Job Request:
        Position: {request['position']}
        Seniority: {request['seniority']}
        Period: {request['period']}
        Skills: {request.get('skills', '')}

        Candidate:
        Position: {candidate['Position']}
        Seniority: {candidate['Seniority']}
        Period: {candidate['Period']}
        Skills: {candidate['Skills']}

        Based on the job request and candidate information provided above, calculate a match score between 0 and 1, where 1 is a perfect match and 0 is no match at all. Only respond with the numeric score, nothing else.
        """

        try:
            response = openai.ChatCompletion.create(
                model="local-model",  # This should match the model name in LM-Studio
                messages=[
                    {"role": "user", "content": user_message}
                ],
                max_tokens=10,
                n=1,
                stop=None,
                temperature=0.5,
            )
            
            # Extract the score from the response
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            
            # Ensure the score is between 0 and 1
            return max(0, min(score, 1))
        except Exception as e:
            print(f"Error calculating match score: {e}")
            return 0.0

# Example usage
if __name__ == "__main__":
    matcher = AIMatcher()
    sample_candidates = [
        {"Position": "Software Engineer", "Seniority": "Senior", "Period": "Immediate", "Name": "John Doe", "Skills": "Python;JavaScript;React"},
        {"Position": "Data Scientist", "Seniority": "Mid", "Period": "1 month", "Name": "Jane Smith", "Skills": "Python;R;Machine Learning"}
    ]
    request = {"position": "Software Engineer", "seniority": "Senior", "period": "Immediate", "skills": "Python;React"}
    matches = matcher.match_candidates(sample_candidates, request)
    print(matches)
