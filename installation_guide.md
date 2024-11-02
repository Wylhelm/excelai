# Installation Guide for Hardware-Accelerated RAG System

This guide provides instructions for setting up the RAG system with hardware acceleration support for both NVIDIA CUDA and Apple Metal devices.

## System Requirements

- Python 3.8 or higher
- One of the following:
  - NVIDIA GPU with CUDA support
  - Apple Silicon Mac (M1/M2/M3)
  - x86 CPU (fallback option)

## Installation Steps

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install base requirements:
```bash
pip install -r requirements.txt
```

3. Install hardware-specific dependencies:

### For NVIDIA CUDA Systems:
```bash
pip uninstall faiss-cpu
pip install faiss-gpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### For Apple Silicon (M1/M2/M3):
```bash
pip uninstall faiss-cpu
pip install faiss-cpu  # Special build will be selected for Apple Silicon
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Verification

To verify the installation and hardware acceleration:

1. Run the following Python code:
```python
import torch
print(f"PyTorch device available: {torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')}")
```

2. Run the vector store test:
```python
python src/vector_store.py
```

This will show which device is being used for acceleration and run a simple test of the vector store functionality.

## Performance Optimization

The system automatically detects and uses the best available hardware:

- On NVIDIA systems, it uses CUDA for both FAISS and PyTorch operations
- On Apple Silicon, it uses Metal for PyTorch operations and optimized CPU implementation for FAISS
- On CPU-only systems, it uses optimized CPU implementations with multi-threading

## Troubleshooting

1. If you encounter CUDA out-of-memory errors:
   - Reduce batch sizes in `ai_matcher.py`
   - Adjust `nlist` and `nprobe` parameters in `vector_store.py`

2. For Apple Silicon users:
   - Ensure you're using Python 3.8 or higher
   - If MPS (Metal) acceleration isn't working, verify macOS version is 12.3+

3. For NVIDIA users:
   - Ensure CUDA toolkit is installed
   - Verify GPU compatibility with installed CUDA version

## Additional Notes

- The system automatically batches operations for optimal performance
- Vector similarity search is optimized using IVFFlat index for better scaling
- The matching process combines vector similarity with LLM scoring for better results
