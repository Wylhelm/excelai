import random
import csv

# Lists of possible values for each column
positions = [
    "Software Engineer", "Data Scientist", "Product Manager", "UX Designer", "DevOps Engineer",
    "Frontend Developer", "Backend Developer", "Data Analyst", "QA Engineer", "Full Stack Developer",
    "Technology Analyst", "Software Architect", "Mobile Developer", "Cloud Engineer", "Business Analyst",
    "Security Engineer", "UI Designer", "Machine Learning Engineer", "Network Administrator"
]

seniority_levels = ["Junior", "Mid", "Senior"]

periods = ["Immediate", "1 week", "2 weeks", "1 month", "2 months"]

first_names = [
    "John", "Jane", "Mike", "Sarah", "Chris", "Emily", "David", "Lisa", "Tom", "Rachel",
    "Robert", "Emma", "Daniel", "Olivia", "Ethan", "Sophia", "Noah", "Ava", "Liam", "Mia",
    "William", "Isabella", "James", "Charlotte", "Benjamin", "Amelia", "Lucas", "Harper",
    "Henry", "Evelyn", "Alexander", "Abigail", "Mason", "Emily", "Michael", "Elizabeth"
]

last_names = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez",
    "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor",
    "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez",
    "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King", "Wright",
    "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green", "Adams", "Nelson", "Baker"
]

skills = [
    "Python", "JavaScript", "React", "R", "Machine Learning", "Agile", "Scrum", "User Stories",
    "Figma", "Sketch", "User Research", "Docker", "Kubernetes", "AWS", "HTML", "CSS", "Java",
    "Spring", "MySQL", "SQL", "Tableau", "Excel", "Selenium", "JUnit", "TestNG", "Node.js",
    "MongoDB", "Azure AI", "C#", ".NET", "Azure", "Swift", "Kotlin", "React Native", "Terraform",
    "CloudFormation", "Power BI", "Penetration Testing", "CISSP", "Wireshark", "Adobe XD",
    "Illustrator", "Prototyping", "TensorFlow", "PyTorch", "Scikit-learn", "Cisco", "CCNA",
    "Network Security"
]

# Function to generate a random candidate
def generate_candidate():
    position = random.choice(positions)
    seniority = random.choice(seniority_levels)
    period = random.choice(periods)
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    num_skills = random.randint(2, 5)
    candidate_skills = ";".join(random.sample(skills, num_skills))
    return [position, seniority, period, name, candidate_skills]

# Read existing candidates
with open('excelai/data/candidates.csv', 'r') as f:
    reader = csv.reader(f)
    existing_candidates = list(reader)

# Generate 100 new candidates
new_candidates = [generate_candidate() for _ in range(100)]

# Append new candidates to the existing file
with open('excelai/data/candidates.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(new_candidates)

print("Added 100 new candidates to the CSV file.")
