from typing import List, Dict
from openai import OpenAI
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from excel_processor import CSVProcessor
from vector_store import VectorStore

class AIMatcher:
    def __init__(self, csv_file_path: str):
        # Configure OpenAI client to use LM-Studio's local API
        self.client = OpenAI(
            base_url="http://localhost:1234/v1",  # Adjust this URL if LM-Studio uses a different port
            api_key="not-needed"  # LM-Studio doesn't require an API key for local use
        )
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.csv_processor = CSVProcessor(csv_file_path)
        self.candidates = self.csv_processor.process_csv_file()
        self.vector_store = VectorStore(self.model.get_sentence_embedding_dimension())
        self.vector_store.add_embeddings(self.csv_processor.get_all_embeddings())

    def match_candidates(self, request: Dict, top_k: int = 10) -> List[Dict]:
        request_embedding = self._create_request_embedding(request)
        similar_candidates = self.vector_store.search(request_embedding, top_k * 2)  # Get more candidates initially
        
        matched_candidates = []
        for candidate_name, similarity in similar_candidates:
            candidate = next(c for c in self.candidates if c['Name'] == candidate_name)
            score = self._calculate_match_score(candidate, request)
            matched_candidates.append({
                "candidate": candidate,
                "score": score
            })
        
        return sorted(matched_candidates, key=lambda x: x["score"], reverse=True)[:top_k]

    def _create_request_embedding(self, request: Dict) -> np.ndarray:
        text = f"{request['position']} {request['seniority']} {request['period']} {request.get('skills', '')}"
        return self.model.encode(text)

    def _calculate_match_score(self, candidate: Dict, request: Dict) -> float:
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

        Based on the job request and candidate information provided above, calculate a match score between 0 and 1, where 1 is a perfect match and 0 is no match at all. Consider the following factors:
        1. How closely the positions align (30% weight)
        2. The match between seniority levels (20% weight)
        3. The compatibility of the time periods (20% weight)
        4. The overlap in required and possessed skills (30% weight)

        Respond with only the numeric score, nothing else.
        """

        try:
            response = self.client.chat.completions.create(
                model="local-model",  # This should match the model name in LM-Studio
                messages=[
                    {"role": "user", "content": user_message}
                ],
                max_tokens=10,
                n=1,
                stop=None,
                temperature=0.3
            )
            
            # Extract the score from the response
            score_text = response.choices[0].message.content.strip()
            
            # Use regex to extract the first float-like string from the response
            match = re.search(r'\d+(\.\d+)?', score_text)
            if match:
                score = float(match.group())
                # Ensure the score is between 0 and 1
                return max(0, min(score, 1))
            else:
                print(f"Unable to extract numeric score from LLM response: {score_text}")
                return 0.0
        except Exception as e:
            print(f"Error calculating match score: {e}")
            return 0.0

# Example usage
if __name__ == "__main__":
    matcher = AIMatcher("../data/candidates.csv")
    request = {"position": "Software Engineer", "seniority": "Senior", "period": "Immediate", "skills": "Python;React"}
    matches = matcher.match_candidates(request)
    print(matches)
