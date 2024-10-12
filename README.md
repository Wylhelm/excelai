# ExcelAI - AI-Powered Candidate Matching System

ExcelAI is an advanced web application designed to streamline the process of matching job candidates with job requests using artificial intelligence. The system processes candidate information from a CSV file and uses a language model to calculate match scores based on job requirements.

## Application Overview

The ExcelAI application consists of several key components:

1. Web Interface (Flask)
2. CSV Processing
3. AI-Powered Matching
4. Results Display

### Web Interface

The application uses Flask to create a web interface where users can input job requirements. The main route ('/') handles both GET and POST requests, allowing users to submit job requests via a form or URL parameters.

### CSV Processing

The `CSVProcessor` class (implemented in `excel_processor.py`) is responsible for reading and processing the CSV file containing candidate information. The CSV file is expected to be located at `data/candidates.csv` relative to the application root.

### AI-Powered Matching

The core of the application is the `AIMatcher` class, which uses a language model to calculate match scores between job requests and candidates.

#### Matching Process

1. The application receives a job request with the following parameters:
   - Position
   - Seniority
   - Period
   - Skills

2. For each candidate in the CSV file, the `AIMatcher` calculates a match score using the following process:
   - It constructs a prompt for the language model, including both the job request and candidate information.
   - The prompt is sent to a local language model (LM-Studio) via the OpenAI API.
   - The language model returns a score between 0 and 1, where 1 is a perfect match and 0 is no match at all.

3. Candidates are sorted based on their match scores in descending order.

#### AI Model

The application uses a local language model through LM-Studio, which is accessed via the OpenAI API. This allows for fast, on-premise processing of matching requests without relying on external API calls.

### Results Display

The matched candidates are displayed on the web interface, sorted by their match scores.

## Matching Criteria

The matching process takes into account several factors:

1. Position: The job title or role
2. Seniority: The level of experience required (e.g., Junior, Mid, Senior)
3. Period: The timeframe for starting the job (e.g., Immediate, 1 month)
4. Skills: Specific skills required for the position

The AI model considers all these factors when calculating the match score. The exact weighting of each factor is determined by the language model's understanding of job matching, which can capture nuanced relationships between different aspects of job requirements and candidate profiles.

## Technical Details

- The application is built using Python and Flask.
- Candidate data is stored in a CSV file.
- The AI matching uses a local language model accessed through the OpenAI API interface.
- The application is designed to be run locally, with the AI model also running on the local machine for data privacy and faster processing.

## Running the Application

To run the application:

1. Ensure all dependencies are installed (see `requirements.txt`).
2. Make sure the local language model (LM-Studio) is running and accessible at the specified URL.
3. Run `python src/app.py` to start the Flask server.
4. Access the application through a web browser at the URL provided by Flask.

Note: The application requires a local setup of LM-Studio for the AI matching functionality to work.
