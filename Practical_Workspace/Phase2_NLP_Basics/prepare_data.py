import os
import requests

# We will use "Alice's Adventures in Wonderland" as our creative story dataset!
# This will teach the model to write whimsical, creative story text.
data_url = 'https://gist.githubusercontent.com/phillipj/4944029/raw/75ba2243dd5ec2875f629bf5d79f6c1e4b5a8b46/alice_in_wonderland.txt'
data_path = 'story_data.txt'

if not os.path.exists(data_path):
    print("Downloading Creative Story Dataset (Alice in Wonderland)...")
    response = requests.get(data_url)
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print("Download complete!")
else:
    print("Dataset already exists.")

# Read the data to show you what it looks like
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Length of dataset in characters: {len(text):,}")

# Get all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size (Total unique characters): {vocab_size}")
print("Here are all the characters our model will learn to generate stories with:")
print("".join(chars))

# Show a tiny preview of the text
print("\n--- STORY PREVIEW ---")
print(text[:300] + "...\n---------------------")
