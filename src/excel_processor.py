import csv
from typing import List, Dict

class CSVProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def process_csv_file(self) -> List[Dict]:
        with open(self.file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            return list(csv_reader)

# Example usage
if __name__ == "__main__":
    processor = CSVProcessor("../data/candidates.csv")
    data = processor.process_csv_file()
    print(data)
