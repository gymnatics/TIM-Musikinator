import os
import numpy as np
import json
from scripts import commons

# Assuming you're in mtg-jamendo-dataset
input_file = "./data/autotagging_moodtheme.tsv"
tracks, tags, extra = commons.read_file(input_file)

path_list = []
labels_list = []

audio_folder = './audio_data'

# print(len(tracks))
for track_id, track_data in tracks.items():
    # if (tag in track_data['mood/theme'] for tag in ['energetic', 'relaxing', 'emotional', 'dark', 'love', 'sad']):
    if all(tag in ['energetic', 'relaxing', 'love', 'sad', 'dark', 'happy'] for tag in track_data['mood/theme']):
            # Load the audio file
            filename = track_data['path']
            # updated_filename = filename.rsplit('.mp3', 1)[0] + '.low.mp3'
            # filepath = os.path.join(audio_folder, updated_filename)
            path_list.append(filename)
            labels_list.append(track_data['mood/theme'])

# Convert sets in labels_list to lists
labels_list = [list(label_set) for label_set in labels_list]

# Combine track IDs and labels into a dictionary
combined_data = {path: label for path, label in zip(path_list, labels_list)}

# print(combined_data)

# Convert combined dictionary to JSON format
json_data = json.dumps(combined_data, indent = 4)

# Write JSON data to a file
with open('tsv_as_json.json', 'w') as json_file:
    json_file.write(json_data)

print("done")