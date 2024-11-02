from typing import List, Dict
from openai import OpenAI
import os
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from excel_processor import CSVProcessor
from vector_store import VectorStore
import atexit

class AIMatcher:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AIMatcher, cls).__new__(cls)
        return cls._instance

    def __init__(self, csv_file_path: str):
        if self._initialized:
            return
            
        # Print CUDA diagnostic information
        self._print_cuda_info()
        
        # Configure OpenAI client
        self.client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="not-needed"
        )
        
        # Initialize device and model
        self.device = self._detect_optimal_device()
        print(f"Using device: {self.device}")
        
        self.model = self._initialize_model()
        self.csv_processor = CSVProcessor(csv_file_path)
        self.candidates = self.csv_processor.process_csv_file()
        
        # Initialize vector store
        self.vector_store = VectorStore(self.model.get_sentence_embedding_dimension())
        
        # Process embeddings
        self._process_embeddings()
        
        # Register cleanup
        atexit.register(self._cleanup)
        
        self._initialized = True

    def _print_cuda_info(self):
        """Print detailed CUDA diagnostic information"""
        print("\nCUDA Diagnostic Information:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
        else:
            print("CUDA is not available. Checking why:")
            try:
                import nvidia_smi
                nvidia_smi.nvmlInit()
                print("NVIDIA driver is installed")
            except:
                print("NVIDIA driver not found or not properly installed")
            
            try:
                torch.cuda.init()
            except Exception as e:
                print(f"CUDA initialization error: {str(e)}")

    def _detect_optimal_device(self) -> str:
        """Detect the optimal device with detailed logging"""
        if torch.cuda.is_available():
            # Verify CUDA is actually working
            try:
                # Try to create a small tensor on CUDA
                test_tensor = torch.cuda.FloatTensor(2, 2)
                del test_tensor  # Clean up
                print("CUDA test successful")
                return "cuda"
            except Exception as e:
                print(f"CUDA test failed: {str(e)}")
                print("Falling back to CPU")
                return "cpu"
        elif torch.backends.mps.is_available():
            print("Apple Metal (MPS) is available but disabled for stability")
            return "cpu"
        else:
            print("No GPU acceleration available")
            return "cpu"

    def _initialize_model(self) -> SentenceTransformer:
        """Initialize the model with device placement logging"""
        print(f"Initializing model on {self.device}")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if self.device == "cuda":
            try:
                model = model.to(self.device)
                print("Successfully moved model to CUDA")
            except Exception as e:
                print(f"Error moving model to CUDA: {str(e)}")
                print("Falling back to CPU")
                self.device = "cpu"
        
        return model

    def _process_embeddings(self):
        """Process embeddings in batches with proper resource management"""
        batch_size = 32
        all_embeddings = {}
        
        # Process in batches
        for i in range(0, len(self.candidates), batch_size):
            batch = self.candidates[i:i + batch_size]
            texts = []
            names = []
            
            for candidate in batch:
                text = f"{candidate['Position']} {candidate['Seniority']} {candidate['Period']} {candidate['Skills']}"
                texts.append(text)
                names.append(candidate['Name'])
            
            # Generate embeddings with proper device handling
            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            
            # Store embeddings
            for name, embedding in zip(names, batch_embeddings):
                all_embeddings[name] = embedding
        
        # Add to vector store
        self.vector_store.add_embeddings(all_embeddings)

    def _cleanup(self):
        """Cleanup resources properly"""
        if hasattr(self, 'model'):
            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Clear model
            self.model = None

    def match_candidates(self, request: Dict, top_k: int = 10) -> List[Dict]:
        # Generate request embedding
        with torch.no_grad():
            request_embedding = self._create_request_embedding(request)
        
        # Get similar candidates
        similar_candidates = self.vector_store.search(request_embedding, top_k * 2)
        
        matched_candidates = []
        batch_size = 5
        
        # Process scoring in batches
        for i in range(0, len(similar_candidates), batch_size):
            batch = similar_candidates[i:i + batch_size]
            batch_candidates = []
            
            for candidate_name, similarity in batch:
                candidate = next(c for c in self.candidates if c['Name'] == candidate_name)
                batch_candidates.append((candidate, similarity))
            
            scores = self._calculate_batch_scores(batch_candidates, request)
            
            for (candidate, similarity), score in zip(batch_candidates, scores):
                final_score = 0.3 * similarity + 0.7 * score
                matched_candidates.append({
                    "candidate": candidate,
                    "score": final_score
                })
        
        return sorted(matched_candidates, key=lambda x: x["score"], reverse=True)[:top_k]

    def _create_request_embedding(self, request: Dict) -> np.ndarray:
        text = f"{request['position']} {request['seniority']} {request['period']} {request.get('skills', '')}"
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding

    def _calculate_batch_scores(self, batch_candidates: List[tuple], request: Dict) -> List[float]:
        scores = []
        
        for candidate, _ in batch_candidates:
            prompt = f"""You are an AI assistant that calculates match scores between job requests and candidates.

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
                    model="local-model",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    n=1,
                    stop=None,
                    temperature=0.3
                )
                
                score_text = response.choices[0].message.content.strip()
                match = re.search(r'\d+(\.\d+)?', score_text)
                if match:
                    score = float(match.group())
                    scores.append(max(0, min(score, 1)))
                else:
                    scores.append(0.0)
            except Exception as e:
                print(f"Error calculating match score: {e}")
                scores.append(0.0)
        
        return scores

# Example usage
if __name__ == "__main__":
    matcher = AIMatcher("../data/candidates.csv")
    request = {"position": "Software Engineer", "seniority": "Senior", "period": "Immediate", "skills": "Python;React"}
    matches = matcher.match_candidates(request)
    print(matches)
