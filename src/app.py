from flask import Flask, render_template, request, flash
from ai_matcher import AIMatcher
import os
import atexit

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = os.urandom(24)

# Initialize AIMatcher as singleton
csv_file_path = os.path.join(app.root_path, '..', 'data', 'candidates.csv')
ai_matcher = AIMatcher(csv_file_path)

def cleanup():
    """Ensure proper cleanup of resources"""
    global ai_matcher
    if hasattr(ai_matcher, '_cleanup'):
        ai_matcher._cleanup()

# Register cleanup handler
atexit.register(cleanup)

@app.route('/', methods=['GET', 'POST'])
def index():
    matches = []
    if request.method == 'POST' or request.args:
        job_request = {
            'position': request.form.get('position') or request.args.get('position', ''),
            'seniority': request.form.get('seniority') or request.args.get('seniority', ''),
            'period': request.form.get('period') or request.args.get('period', ''),
            'skills': request.form.get('skills') or request.args.get('skills', '')
        }

        try:
            matches = ai_matcher.match_candidates(job_request)
        except Exception as e:
            flash(f"An error occurred while matching candidates: {str(e)}", "error")

    return render_template('index.html', matches=matches)

if __name__ == '__main__':
    app.run(debug=False)  # Set debug=False to prevent duplicate resource initialization
