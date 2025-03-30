import pandas as pd
from transformers import pipeline
import re

# Load the cleaned complaints data.
complaints_df = pd.read_csv('data/AxionRay_assignment_cleaned.csv')
print(complaints_df.columns.dtype())
print(complaints_df.head(2))

# --- Step 1a: Identify Free Text Columns ---
#obj_cols = [col for col in complaints_df.columns if complaints_df[col].dtype == 'object']
#print(complaints_df[obj_cols].head(2))
free_text_columns = ["CAUSAL_VERBATIM","CORRECTION_VERBATIM","CUSTOMER_VERBATIM"]
print("Identified free text columns:", free_text_columns)
#
# --- Step 1b: Extract Meaningful Tags using Text Mining ---
#Using Hugging Face NER pipeline with the lightweight model 'dslim/bert-base-NER' for Named entity recognition
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

def extract_entities(text):
    """
    Given a string 'text', this function uses the NER pipeline
    to extract entities and returns a list of unique entity labels.
    """
    if pd.isnull(text):
        return []
    # Generate tags on text using the NER pipeline
    entities = ner_pipeline(text)
    # Extract the entity 'word' and 'entity_group'
    # (Here we can extract words as tags or focus on entity groups.)
    extracted_tags = {entity['word'] for entity in entities}
    return list(extracted_tags)

# For each free text column, extract entities and store the tags in new columns.
for col in free_text_columns:
    # Create a new column for the extracted tags from each free text column
    new_col = f"{col}_tags"
    complaints_df[new_col] = complaints_df[col].apply(extract_entities)

# Optionally, combine tags from multiple columns into a single 'tags' column.
def combine_tags(row):
    tags = []
    for col in free_text_columns:
        tags.extend(row[f"{col}_tags"])
    # Return unique tags (cleaned up, e.g., remove punctuation)
    unique_tags = set([re.sub(r'\W+', '', tag.lower()) for tag in tags])
    return list(unique_tags)

complaints_df['combined_tags'] = complaints_df.apply(combine_tags, axis=1)

# --- Step 2: Categorize "Type of Issues" based on Extracted Entities ---

def categorize_issue(tags):
    """
    Categorize the type of issue based on the presence of certain keywords in the extracted tags.
    You can adjust these rules based on domain knowledge.
    """
    # Define simple keyword mappings to categories:
    # Note: The keywords are assumed to be lower-case after cleaning.
    component_failure_keywords = {'radio', 'display', 'module', 'component', 'screen'}
    electrical_issue_keywords = {'electrical', 'wiring', 'circuit', 'voltage', 'power'}
    software_issue_keywords = {'software', 'update', 'program', 'firmware'}

    tags_set = set(tags)
    
    # Check for each category based on keyword overlap:
    if tags_set & component_failure_keywords:
        return "Component Failure"
    elif tags_set & electrical_issue_keywords:
        return "Electrical Issue"
    elif tags_set & software_issue_keywords:
        return "Software Issue"
    else:
        return "Other"

# Apply the categorization to the combined tags column.
complaints_df['issue_category'] = complaints_df['combined_tags'].apply(categorize_issue)

# --- Display Some Results ---
print("\nSample of extracted tags and assigned issue categories:")
print(complaints_df[['combined_tags', 'issue_category']].head())

# Optionally, you can save the enhanced DataFrame to a CSV for further analysis.
complaints_df.to_csv('data/complaints_enhanced.csv', index=False)
