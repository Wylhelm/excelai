# ExcelAI Installation Guide

This guide will walk you through the process of setting up and running the ExcelAI project, a Flask-based web application that uses AI to match job candidates with job requests.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

1. Python 3.8 or higher
2. pip (Python package manager)
3. Git (optional, for cloning the repository)

## Installation Steps

1. Clone the repository (if you haven't already):
   ```
   git clone <repository_url>
   cd excelai
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration

1. Ensure you have a CSV file with candidate data located at `excelai/data/candidates.csv`.

2. Set up LM-Studio:
   - Download and install LM-Studio from [https://lmstudio.ai/](https://lmstudio.ai/)
   - Launch LM-Studio and load a compatible language model
   - Start the local API server in LM-Studio (usually runs on http://localhost:1234)

3. (Optional) Create a `.env` file in the project root directory to store any environment variables:
   ```
   OPENAI_API_BASE=http://localhost:1234/v1
   OPENAI_API_KEY=not-needed
   ```

## Running the Application

1. Make sure your virtual environment is activated.

2. Start the Flask application:
   ```
   python src/app.py
   ```

3. Open a web browser and navigate to `http://localhost:5000` to access the application.

## Usage

1. On the main page, you'll see a form where you can enter job request details:
   - Position
   - Seniority
   - Period
   - Skills (comma-separated)

2. Submit the form to see matched candidates based on your job request.

## Troubleshooting

- If you encounter any issues with dependencies, try updating them to the latest versions:
  ```
  pip install --upgrade -r requirements.txt
  ```

- Ensure that LM-Studio is running and the local API server is accessible before starting the Flask application.

- If you experience any errors related to file paths, double-check that the `candidates.csv` file is located in the correct directory (`excelai/data/`).

## Additional Notes

- This application uses a local language model through LM-Studio for candidate matching. Make sure LM-Studio is properly set up and running before using the application.

- The matching algorithm uses a combination of vector embeddings and AI-based scoring to find the best candidates for a given job request.

- For any further questions or issues, please refer to the project documentation or contact the development team.
