import faiss
import numpy as np
from typing import List, Tuple, Dict

class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.id_to_candidate = {}

    def add_embeddings(self, embeddings: Dict[str, np.ndarray]):
        for candidate_name, embedding in embeddings.items():
            if embedding.shape[0] != self.dimension:
                raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {embedding.shape[0]}")
            
            id = self.index.ntotal
            self.index.add(embedding.reshape(1, -1))
            self.id_to_candidate[id] = candidate_name

    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query embedding dimension mismatch. Expected {self.dimension}, got {query_embedding.shape[0]}")
        
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for i, distance in zip(indices[0], distances[0]):
            if i != -1:  # -1 indicates no match found
                candidate_name = self.id_to_candidate[i]
                results.append((candidate_name, float(distance)))
        
        return results

    def get_total_candidates(self) -> int:
        return self.index.ntotal

# Example usage
if __name__ == "__main__":
    dimension = 384  # Dimension of the embeddings from 'all-MiniLM-L6-v2' model
    vector_store = VectorStore(dimension)
    
    # Add some sample embeddings
    sample_embeddings = {
        "Candidate1": np.random.rand(dimension),
        "Candidate2": np.random.rand(dimension),
        "Candidate3": np.random.rand(dimension),
    }
    vector_store.add_embeddings(sample_embeddings)
    
    # Perform a sample search
    query = np.random.rand(dimension)
    results = vector_store.search(query, k=2)
    
    print(f"Total candidates: {vector_store.get_total_candidates()}")
    print("Search results:")
    for candidate, distance in results:
        print(f"{candidate}: distance = {distance}")
