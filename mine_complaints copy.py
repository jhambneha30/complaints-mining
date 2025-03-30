import pandas as pd
import ast
import re
from transformers import pipeline

# Load the cleaned complaints data.
complaints_df = pd.read_csv('data/AxionRay_assignment_cleaned.csv')
print(complaints_df.columns.dtype())
print(complaints_df.head(2))


###################### Function Definitions ######################
def extract_entities(text):
    """
    Given a string 'text', use the NER pipeline to extract entity words.
    Returns a list of unique tags (lowercased and stripped of punctuation).
    """
    if pd.isnull(text) or not isinstance(text, str) or text.strip() == "":
        return []
    entities = ner_pipeline(text)
    # Extract the word for each entity and clean up (lowercase and remove non-alphanumeric characters)
    tags = {re.sub(r'\W+', '', ent['word'].lower()) for ent in entities}
    # Remove empty strings if any
    return [tag for tag in tags if tag]

def parse_if_list(text):
    """
    Checks if the provided text is a string representation of a list.
    If yes, safely parse it using ast.literal_eval; otherwise return None.
    """
    if isinstance(text, str) and text.strip().startswith('['):
        try:
            return ast.literal_eval(text)
        except Exception:
            return None
    return None

def combine_tags(row):
    combined = []
    for col in tag_columns:
        combined.extend(row[f"{col}_tags"])
    # Remove duplicates
    return list(set(combined))

# -----------------------------
# Categorize "Type of Issues" based on the combined tags
# -----------------------------
def categorize_issue(tags):
    """
    Categorize the issue type based on keyword matches in the tags.
    The mapping below is an example and can be adjusted.
    """
    # Define simple keyword sets (all lower-case, no punctuation)
    component_failure_keywords = {'radio', 'display', 'module', 'component', 'screen'}
    electrical_issue_keywords = {'electrical', 'wiring', 'circuit', 'voltage', 'power'}
    software_issue_keywords = {'software', 'update', 'program', 'firmware'}
    
    tags_set = set(tags)
    
    # Check for matches in each category
    if tags_set & component_failure_keywords:
        return "Component Failure"
    elif tags_set & electrical_issue_keywords:
        return "Electrical Issue"
    elif tags_set & software_issue_keywords:
        return "Software Issue"
    else:
        return "Other"


# Define the columns to process for tags
tag_columns = [
    'Trigger', 'Failure Component', 'Failure Condition', 
    'Additional Context', 'Fix Component', 'Fix Condition'
]


# Initialize Hugging Face NER pipeline for Named Entity Recognition
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# -----------------------------
# Process each free-text column to generate tags
# -----------------------------
# We'll store the generated tags in new columns with suffix '_tags'
for col in tag_columns:
    new_col = f"{col}_tags"
    
    def get_tags(value):
        # First, check if the value is already a list (or string that looks like one)
        parsed = parse_if_list(value)
        if parsed is not None:
            # Clean tags: lower-case and remove punctuation
            return [re.sub(r'\W+', '', str(tag).lower()) for tag in parsed if tag]
        else:
            # Otherwise, extract entities from the free text
            return extract_entities(value)
    
    complaints_df[new_col] = complaints_df[col].apply(get_tags)


# Combine tags from the specified columns for categorization
complaints_df['combined_tags'] = complaints_df.apply(combine_tags, axis=1)
complaints_df['issue_category'] = complaints_df['combined_tags'].apply(categorize_issue)

# -----------------------------
# Display the enhanced DataFrame columns for stakeholder insights
# -----------------------------
# Columns of interest include the generated tag columns, combined_tags, and issue_category.
print("Enhanced DataFrame with generated tags and issue category:")
display_cols = tag_columns + [f"{col}_tags" for col in tag_columns] + ['combined_tags', 'issue_category']
print(complaints_df[display_cols].head())

# Save processed complaints data to csv
complaints_df.to_csv('data/processed_complaints.csv', index=False)
