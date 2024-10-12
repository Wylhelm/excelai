# ExcelAI - AI-Powered Candidate Matching System

ExcelAI is an advanced web application designed to streamline the process of matching job candidates with job requests using artificial intelligence. The system processes candidate information from a CSV file and uses a combination of vector similarity search and language model inference to calculate match scores based on job requirements.

## Application Overview

The ExcelAI application consists of several key components:

1. Web Interface (Flask)
2. CSV Processing
3. Vector Database (FAISS)
4. AI-Powered Matching (Hybrid Approach)
5. Results Display

### Web Interface

The application uses Flask to create a web interface where users can input job requirements. The main route ('/') handles both GET and POST requests, allowing users to submit job requests via a form or URL parameters.

### CSV Processing

The `CSVProcessor` class (implemented in `excel_processor.py`) is responsible for reading and processing the CSV file containing candidate information. The CSV file is expected to be located at `data/candidates.csv` relative to the application root.

### Vector Database

The application uses FAISS (Facebook AI Similarity Search) for efficient similarity search and clustering of dense vectors. This is implemented in the `VectorStore` class (in `vector_store.py`), which provides methods for adding embeddings and performing similarity searches.

### AI-Powered Matching

The core of the application is the `AIMatcher` class, which uses a hybrid approach combining vector similarity search and language model inference to calculate match scores between job requests and candidates.

#### Matching Process

1. The application receives a job request with the following parameters:
   - Position
   - Seniority
   - Period
   - Skills

2. The matching process follows these steps:
   - The job request is converted into an embedding using SentenceTransformer.
   - A similarity search is performed using the FAISS index to find the most similar candidate embeddings.
   - For each similar candidate, a more detailed match score is calculated using a language model.
   - The language model considers the job request and candidate information to produce a score between 0 and 1.

3. Candidates are sorted based on their final match scores in descending order.

#### AI Models

The application uses two types of AI models:

1. SentenceTransformer ('all-MiniLM-L6-v2'): Used for generating embeddings of job requests and candidate information.
2. Local Language Model (via LM-Studio): Used for calculating detailed match scores. This model is accessed through the OpenAI API interface, allowing for fast, on-premise processing without relying on external API calls.

### Results Display

The matched candidates are displayed on the web interface, sorted by their match scores.

## Matching Criteria

The matching process takes into account several factors:

1. Position: The job title or role
2. Seniority: The level of experience required (e.g., Junior, Mid, Senior)
3. Period: The timeframe for starting the job (e.g., Immediate, 1 month)
4. Skills: Specific skills required for the position

The initial similarity search uses vector embeddings to find candidates with similar overall profiles. The language model then considers all these factors in detail when calculating the final match score, capturing nuanced relationships between different aspects of job requirements and candidate profiles.

## Technical Details

- The application is built using Python and Flask.
- Candidate data is stored in a CSV file.
- FAISS is used for efficient similarity search of candidate embeddings.
- SentenceTransformer is used for generating embeddings of job requests and candidate information.
- The detailed matching uses a local language model accessed through the OpenAI API interface.
- The application is designed to be run locally, with all AI models also running on the local machine for data privacy and faster processing.

## Running the Application

To run the application:

1. Ensure all dependencies are installed (see `requirements.txt`).
2. Make sure the local language model (LM-Studio) is running and accessible at the specified URL.
3. Run `python src/app.py` to start the Flask server.
4. Access the application through a web browser at the URL provided by Flask.

Note: The application requires a local setup of LM-Studio for the AI matching functionality to work.

## New Features and Improvements

- Hybrid matching approach: Combines the efficiency of vector similarity search with the nuanced understanding of a language model.
- FAISS integration: Enables fast and efficient similarity search for initial candidate filtering.
- SentenceTransformer: Provides high-quality embeddings for job requests and candidate information.
- Scalability: The vector database allows for efficient matching even with a large number of candidates.
- Flexibility: The system can be easily adapted to use different embedding models or language models as needed.
