import os
import re
from datasets import load_from_disk, Dataset
from bs4 import BeautifulSoup

# Set the base directory relative to the script's location
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '../data/ArXiv/')
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

    def clean_text(text):
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove LaTeX placeholders and citations
        text = re.sub(r'@xcite|@x\w+', ' ', text)  # Remove @xcite, @xmath-like placeholders
        
        # Remove LaTeX commands and environments
        text = re.sub(r'\\[a-zA-Z]+\b', ' ', text)  # Remove LaTeX commands like \text
        text = re.sub(r'\$.*?\$', ' ', text)  # Remove inline math ($...$)
        text = re.sub(r'\\\[.*?\\\]', ' ', text)  # Remove block math (\[...\])
        text = re.sub(r'\\begin{.*?}\\.*?\\end{.*?}', ' ', text, flags=re.DOTALL)  # Remove LaTeX environments
        
        # Remove brackets and contents in references or math
        text = re.sub(r'\[.*?\]', ' ', text)  # Remove contents in square brackets
        text = re.sub(r'\{.*?\}', ' ', text)  # Remove contents in curly braces
        
        # Remove references (assumes references start with common patterns like numbers or DOIs)
        text = re.split(r'(references|bibliography|^\d+\.\s|\[\d+\]\s|doi:|arxiv:|et al\.|vol\.)', text, flags=re.IGNORECASE)[0]

        # Remove excessive whitespace and newlines
        text = re.sub(r'\n', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text).strip()  # Remove multiple spaces
        
        return text

    def process_examples(example):
        # Clean HTML tags and unwanted content in 'abstract' and 'article'
        example['abstract'] = clean_text(example['abstract'])
        example['article'] = clean_text(example['article'])
        # Rename 'article' column to 'text'
        example['text'] = example.pop('article')
        return example

    # Filter rows where the 'abstract' length exceeds 3000 characters
    filtered_dataset = dataset.filter(lambda x: len(x['abstract']) <= 3000)

    # Apply cleaning and mapping
    processed_dataset = filtered_dataset.map(process_examples, remove_columns=['article'])
    
    # Save the processed dataset
    processed_dataset.save_to_disk(output_dir)

print("Processing complete.")
