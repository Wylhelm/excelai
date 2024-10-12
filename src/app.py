from flask import Flask, render_template, request, flash
from excel_processor import CSVProcessor
from ai_matcher import AIMatcher
import os

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = os.urandom(24)  # Set a secret key for flash messages

# Initialize CSVProcessor and AIMatcher
csv_file_path = os.path.join(app.root_path, '..', 'data', 'candidates.csv')
csv_processor = CSVProcessor(csv_file_path)
ai_matcher = AIMatcher()

@app.route('/', methods=['GET', 'POST'])
def index():
    matches = []
    if request.method == 'POST' or request.args:
        # Process the form submission (POST) or URL parameters (GET)
        job_request = {
            'position': request.form.get('position') or request.args.get('position', ''),
            'seniority': request.form.get('seniority') or request.args.get('seniority', ''),
            'period': request.form.get('period') or request.args.get('period', ''),
            'skills': request.form.get('skills') or request.args.get('skills', '')
        }

        try:
            # Process CSV file and match candidates
            csv_data = csv_processor.process_csv_file()
            matches = ai_matcher.match_candidates(csv_data, job_request)
        except Exception as e:
            flash(f"An error occurred while matching candidates: {str(e)}", "error")

    return render_template('index.html', matches=matches)

if __name__ == '__main__':
    app.run(debug=True)
