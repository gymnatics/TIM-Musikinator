import csv
import json

# Open the TSV file and read the data
with open('autotagging_moodtheme.tsv', 'r') as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t')
    data = [row for row in reader if row['PATH'].startswith('00/')]

# Open the JSON file and write the data
with open('data.json', 'w') as jsonfile:
    json.dump(data, jsonfile)