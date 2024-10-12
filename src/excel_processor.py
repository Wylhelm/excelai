import csv
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

class CSVProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = {}

    def process_csv_file(self) -> List[Dict]:
        with open(self.file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            data = list(csv_reader)
            
        for candidate in data:
            self.create_embedding(candidate)
        
        return data

    def create_embedding(self, candidate: Dict):
        # Combine relevant fields for embedding
        text = f"{candidate['Position']} {candidate['Seniority']} {candidate['Period']} {candidate['Skills']}"
        embedding = self.model.encode(text)
        self.embeddings[candidate['Name']] = embedding

    def get_embedding(self, candidate_name: str) -> np.ndarray:
        return self.embeddings.get(candidate_name, None)

    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        return self.embeddings

# Example usage
if __name__ == "__main__":
    processor = CSVProcessor("../data/candidates.csv")
    data = processor.process_csv_file()
    print(f"Processed {len(data)} candidates")
    print(f"Created {len(processor.get_all_embeddings())} embeddings")
