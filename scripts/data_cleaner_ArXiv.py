import os
import re
from datasets import load_from_disk, Dataset
from bs4 import BeautifulSoup

# Set the base directory relative to the script's location
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '../data/origin/ArXiv/')
processed_data_dir = os.path.join(base_dir, '../data/Processed/ArXiv/')
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

        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        
        # Remove LaTeX placeholders, commands, and math
        text = re.sub(r'@x\w+|\\[a-zA-Z]+|\$.*?\$|\\\[.*?\\\]|\\begin{.*?}\\.*?\\end{.*?}', ' ', text, flags=re.DOTALL)
        text = re.sub(r'\\[,{}\[\]]|\^\s*_\s*\}|\^\s*_\s*\^\s*_|fig\. \(\s*\)', ' ', text, flags=re.IGNORECASE)
        
        # Remove unresolved references and specific patterns
        text = re.sub(r'\b[a-zA-Z_]+\s*et\s*al\.?,?\b|\bclass\. grav\.?\s*,?', ' ', text, flags=re.IGNORECASE)

        # Remove content inside brackets and braces
        text = re.sub(r'\[.*?\]|\{.*?\}', ' ', text)

        # Remove repeated punctuation and unnecessary spaces
        text = re.sub(r'([.,!?])\1+|\s+([.,!?])|[.,!?]{2,}', r'\1', text)
        
        # Remove standalone numbers and special characters
        text = re.sub(r'\b\d+\b|[^\w\s.,]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def process_examples(example):
        # Clean HTML tags and illegal characters in all string columns
        example['abstract'] = clean_html_and_illegal_chars(example['abstract'])
        example['article'] = clean_html_and_illegal_chars(example['article'])
        # Rename 'article' column to 'text'
        example['text'] = example.pop('article')
        return example

    # First, filter out rows where the 'abstract' length is greater than 3000 characters
    filtered_dataset = dataset.filter(lambda x: len(x['abstract']) <= 3000)

    # Then, map the processing function without returning None
    processed_dataset = filtered_dataset.map(process_examples, remove_columns=['article'])
    
    # Save the processed dataset
    processed_dataset.save_to_disk(output_dir)

print("Processing complete.")
