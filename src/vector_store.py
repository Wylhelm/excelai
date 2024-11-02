import faiss
import numpy as np
import torch
from typing import List, Tuple, Dict
import platform
import atexit

class VectorStore:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
        return cls._instance

    def __init__(self, dimension: int):
        if self._initialized:
            return
            
        self.dimension = dimension
        self.device = self._detect_hardware()
        self.gpu_resource = None
        self.index = None
        self.id_to_candidate = {}
        
        # Initialize the index
        self._initialize_index()
        
        # Register cleanup handler
        atexit.register(self._cleanup)
        
        self._initialized = True

    def _detect_hardware(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        # Temporarily disable MPS due to stability issues
        # elif torch.backends.mps.is_available():
        #     return "mps"
        return "cpu"

    def _initialize_index(self):
        """Initialize the index with a simple FlatL2 index initially"""
        self.index = faiss.IndexFlatL2(self.dimension)
        
        if self.device == "cuda":
            self.gpu_resource = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, self.index)

    def _optimize_index(self, num_vectors: int):
        """Optimize the index based on the number of vectors"""
        # Only optimize if we have enough vectors
        if num_vectors < 1000:
            return  # Keep using the simple FlatL2 index for small datasets
            
        # Calculate optimal number of clusters (1 cluster per 30 vectors, min 2, max 100)
        nlist = min(max(2, num_vectors // 30), 100)
        
        # Create new optimized index
        quantizer = faiss.IndexFlatL2(self.dimension)
        new_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
        
        # If we have vectors in the current index, transfer them
        if self.index.ntotal > 0:
            # Convert current index to CPU if needed
            current_index = self.index
            if self.device == "cuda":
                current_index = faiss.index_gpu_to_cpu(self.index)
            
            # Get the vectors from current index
            vectors = faiss.vector_to_array(current_index.get_xb()).reshape(-1, self.dimension)
            
            # Train and add to new index
            new_index.train(vectors)
            new_index.add(vectors)
            
            # Update ID mapping (no need to change as the order remains the same)
        
        # Move to GPU if needed
        if self.device == "cuda":
            new_index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, new_index)
        
        # Replace old index
        self.index = new_index

    def _cleanup(self):
        """Cleanup resources properly"""
        if self.device == "cuda" and self.gpu_resource is not None:
            if self.index is not None:
                self.index = faiss.index_gpu_to_cpu(self.index)
            self.gpu_resource = None
        
        self.index = None
        self.id_to_candidate.clear()

    def add_embeddings(self, embeddings: Dict[str, np.ndarray]):
        if not embeddings:
            return
            
        vectors = []
        names = []
        for name, embedding in embeddings.items():
            vectors.append(embedding)
            names.append(name)
        
        vectors = np.stack(vectors)
        
        # Optimize index based on total number of vectors
        total_vectors = len(vectors) + (self.index.ntotal if self.index else 0)
        self._optimize_index(total_vectors)
        
        # Add vectors in batches
        batch_size = 1000
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_names = names[i:i + batch_size]
            batch_ids = range(self.index.ntotal, self.index.ntotal + len(batch_vectors))
            
            self.index.add(batch_vectors)
            
            # Update id mapping
            for idx, name in zip(batch_ids, batch_names):
                self.id_to_candidate[idx] = name

    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query embedding dimension mismatch. Expected {self.dimension}, got {query_embedding.shape[0]}")
        
        # Adjust search parameters for IVF index
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = min(20, self.index.ntotal)
        
        distances, indices = self.index.search(query_embedding.reshape(1, -1), min(k, self.index.ntotal))
        
        results = []
        for i, distance in zip(indices[0], distances[0]):
            if i != -1 and i in self.id_to_candidate:
                candidate_name = self.id_to_candidate[i]
                similarity = 1 / (1 + distance)
                results.append((candidate_name, float(similarity)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_total_candidates(self) -> int:
        return self.index.ntotal if self.index else 0

# Example usage
if __name__ == "__main__":
    dimension = 384
    vector_store = VectorStore(dimension)
    
    sample_embeddings = {
        "Candidate1": np.random.rand(dimension),
        "Candidate2": np.random.rand(dimension),
        "Candidate3": np.random.rand(dimension),
    }
    vector_store.add_embeddings(sample_embeddings)
    
    query = np.random.rand(dimension)
    results = vector_store.search(query, k=2)
    
    print(f"Total candidates: {vector_store.get_total_candidates()}")
    print(f"Using device: {vector_store.device}")
    print("Search results:")
    for candidate, similarity in results:
        print(f"{candidate}: similarity = {similarity}")
