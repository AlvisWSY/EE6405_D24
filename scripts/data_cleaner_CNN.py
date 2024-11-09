import os
import re
from datasets import load_from_disk, Dataset
from bs4 import BeautifulSoup

# Set the base directory relative to the script's location
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '../data/cnn_dailymail/')
processed_data_dir = os.path.join(base_dir, '../data/Processed/cnn_dailymail/')
sub_dirs = ['train', 'test', 'validation']

# Make sure the processed directory exists
os.makedirs(processed_data_dir, exist_ok=True)

for sub_dir in sub_dirs:
    input_dir = os.path.join(data_dir, sub_dir)
    output_dir = os.path.join(processed_data_dir, sub_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset from disk
    dataset = load_from_disk(input_dir)

    def clean_html_and_illegal_chars(text):
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        # Remove illegal characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text

    def remove_location_and_author(text):
        # Remove location/author patterns at the start of the text, e.g., "LONDON, England (Reuters) -- "
        pattern = r'^[A-Z\s,]+\([A-Za-z]+\)\s*--\s*'
        text = re.sub(pattern, '', text).strip()
        # Remove patterns like "By . James Chapman for the Daily Mail ."
        pattern_byline = r'^By\s\.\s.*?\.'
        text = re.sub(pattern_byline, '', text).strip()
        return text

    def process_examples(example):
        # Remove samples where the length of highlights is greater than 752 characters
        if len(example['highlights']) > 752:
            return None
        # Clean HTML tags and illegal characters in all string columns
        example['highlights'] = clean_html_and_illegal_chars(example['highlights'])
        example['article'] = clean_html_and_illegal_chars(example['article'])
        # Remove location/author from the start of the article text
        example['article'] = remove_location_and_author(example['article'])
        # Rename 'highlights' column to 'abstract' and 'article' column to 'text'
        example['abstract'] = example.pop('highlights')
        example['text'] = example.pop('article')
        return example

    # Filter out and process dataset rows
    filtered_dataset = dataset.filter(lambda x: len(x['highlights']) <= 752)
    processed_dataset = filtered_dataset.map(process_examples, remove_columns=['article', 'id'])
    
    # Save the processed dataset
    processed_dataset.save_to_disk(output_dir)

print("Processing complete.")
