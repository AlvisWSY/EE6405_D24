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
        # Remove HTML tags (if any)
        text = re.sub(r'<.*?>', ' ', text)
        
        # Remove LaTeX placeholders and commands
        text = re.sub(r'@xcite|@x\w+', ' ', text)  # Remove LaTeX-like placeholders
        text = re.sub(r'\\[a-zA-Z]+', ' ', text)  # Remove LaTeX commands like \text
        text = re.sub(r'\$.*?\$', ' ', text)  # Remove inline math ($...$)
        text = re.sub(r'\\\[.*?\\\]', ' ', text)  # Remove block math (\[...\])
        text = re.sub(r'\\begin{.*?}\\.*?\\end{.*?}', ' ', text, flags=re.DOTALL)  # Remove LaTeX environments
        
        # Remove LaTeX-like sequences
        text = re.sub(r'\\,|\\\]|\\\{', ' ', text)  # Remove escaped commas, brackets, and curly braces
        text = re.sub(r'\^ _ \}', ' ', text)  # Remove specific LaTeX patterns
        text = re.sub(r'\^ _ \^ _', ' ', text)  # Remove patterns like ^ _ ^ _
        text = re.sub(r'\^ _ \}', ' ', text)  # Remove patterns like ^ _ }

        # Remove fig. ( ) and similar patterns
        text = re.sub(r'fig\. \(\s*\)', ' ', text, flags=re.IGNORECASE)  # Remove fig. ( ) and variants
        
        # Remove unresolved references like "aasi _ et al."
        text = re.sub(r'[a-zA-Z_]+\s*et\s*al\.?\s*,?', ' ', text, flags=re.IGNORECASE)  # Remove "et al." references
        text = re.sub(r'class\. grav\.\s*\*?\s*\*?\s*,?', ' ', text, flags=re.IGNORECASE)  # Remove "class. grav."

        # Remove brackets and their contents
        text = re.sub(r'\[[^\]]*\]', ' ', text)  # Remove contents in square brackets
        text = re.sub(r'\{[^\}]*\}', ' ', text)  # Remove contents in curly braces

        # Remove sequences of repeated punctuation
        text = re.sub(r'([.,!?])\1+', r'\1', text)  # Replace repeated punctuation like "..." with "."
        text = re.sub(r'\s+([.,!?])', r'\1', text)  # Remove spaces before punctuation
        text = re.sub(r'[.,!?]{2,}', ' ', text)  # Remove repeated punctuation entirely
        
        # Remove numbers, standalone symbols, and extra spaces
        text = re.sub(r'\b\d+\b', ' ', text)  # Remove isolated numbers
        text = re.sub(r'[^\w\s.,]', ' ', text)  # Remove standalone special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces

        # Remove _ and ^ characters
        text = re.sub(r'[\^_]', ' ', text)  # Remove _ and ^ characters

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
