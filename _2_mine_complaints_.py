import pandas as pd
import ast
import re
from transformers import pipeline

## Try models from higging face: 
MODEL_TO_MINE = 'google/flan-t5-small'

###################### Function Definitions ######################
def extract_entities(text):
    """
    Given a string 'text', use the NER pipeline to extract entity words.
    Returns a list of unique tags (lowercased and stripped of punctuation).
    """
    if pd.isnull(text) or not isinstance(text, str) or text.strip() == "":
        return []
    entities = ner_pipeline(text)
    tags = {re.sub(r'\W+', '', ent['word'].lower()) for ent in entities}
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

def combine_tags(row, columns):
    """
    Combine tags from the given list of columns (assumed to have '_tags' suffix)
    into a single list of unique tags.
    """
    combined = []
    for col in columns:
        combined.extend(row[f"{col}_tags"])
    return list(set(combined))

def categorize_issue(tags):
    """
    Categorize the issue type based on keyword matches in the tags.
    The mapping below is an example and can be adjusted.
    """
    component_failure_keywords = {'radio', 'display', 'module', 'component', 'screen'}
    electrical_issue_keywords = {'electrical', 'wiring', 'circuit', 'voltage', 'power'}
    software_issue_keywords = {'software', 'update', 'program', 'firmware'}
    
    tags_set = set(tags)
    if tags_set & component_failure_keywords:
        return "Component Failure"
    elif tags_set & electrical_issue_keywords:
        return "Electrical Issue"
    elif tags_set & software_issue_keywords:
        return "Software Issue"
    else:
        return "Other"

def generate_issue_details(row):
    """
    Generate values for the last six columns based on the free-text columns:
    CAUSAL_VERBATIM, CORRECTION_VERBATIM, CUSTOMER_VERBATIM.
    
    Uses a generation model (e.g. google/flan-t5-small) that accepts instructions.
    The output is expected in JSON format with keys:
    Trigger, Failure Component, Failure Condition, Additional Context, Fix Component, Fix Condition.
    
    Example prompt (with an example row) can be included as context.
    """
    prompt = (
        "Based on the following texts:\n"
        f"CAUSAL_VERBATIM: {row['CAUSAL_VERBATIM']}\n"
        f"CORRECTION_VERBATIM: {row['CORRECTION_VERBATIM']}\n"
        f"CUSTOMER_VERBATIM: {row['CUSTOMER_VERBATIM']}\n\n"
        "Generate the following details in JSON format with keys exactly as shown:\n"
        "{\n"
        '  "Trigger": "<value>",\n'
        '  "Failure Component": "<value>",\n'
        '  "Failure Condition": "<value>",\n'
        '  "Additional Context": "<value>",\n'
        '  "Fix Component": "<value>",\n'
        '  "Fix Condition": "<value>"\n'
        "}\n\n"
        "Example (for context):\n"
        '{ "Trigger": "No Additional Functionality", "Failure Component": "Audio Unit, Display", '
        '"Failure Condition": "Controls Irresponsive, Black Screen", "Additional Context": "No Additional Context", '
        '"Fix Component": "Radio, SPS", "Fix Condition": "Replaced, Programmed" }\n\n'
        "Now generate the values based on the provided texts."
    )
    generated = generation_pipeline(prompt, max_length=256)[0]['generated_text']
    try:
        details = eval(generated)
    except Exception:
        details = {}
    return details

###################### End Function Definitions ######################

# Load the cleaned complaints data.
complaints_df = pd.read_csv('data/AxionRay_assignment_cleaned.csv')
print(complaints_df.columns.dtype)
print(complaints_df.head(2))

# Define free-text columns for generating tags.
free_text_columns = ["CAUSAL_VERBATIM", "CORRECTION_VERBATIM", "CUSTOMER_VERBATIM"]

# Initialize Hugging Face pipelines.
# A NER pipeline (if you want to extract standard entities, not used in generation below).
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
# A generation pipeline using a small instruction-following model.
generation_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

# Process the free-text columns to generate tags using NER (if desired).
for col in free_text_columns:
    new_col = f"{col}_tags"
    
    def get_tags(value):
        parsed = parse_if_list(value)
        if parsed is not None:
            return [re.sub(r'\W+', '', str(tag).lower()) for tag in parsed if tag]
        else:
            return extract_entities(value)
    
    complaints_df[new_col] = complaints_df[col].apply(get_tags)

# Combine tags from free-text columns (optional, for additional insights)
complaints_df['combined_tags'] = complaints_df.apply(lambda row: combine_tags(row, free_text_columns), axis=1)
complaints_df['issue_category'] = complaints_df['combined_tags'].apply(categorize_issue)

# Now generate the last six columns for rows that do not have values.
# The target columns are:
target_columns = ["Trigger", "Failure Component", "Failure Condition", 
                  "Additional Context", "Fix Component", "Fix Condition"]

# For demonstration, assume that if "Trigger" is missing or empty, we generate the values.
def is_missing(cell):
    return pd.isnull(cell) or str(cell).strip() == ""

for idx, row in complaints_df.iterrows():
    # Check if the row is missing values in the target columns.
    if all(is_missing(row.get(col, None)) for col in target_columns):
        details = generate_issue_details(row)
        # Update the DataFrame with the generated values.
        for col in target_columns:
            complaints_df.at[idx, col] = details.get(col, "")

# Display enhanced DataFrame columns for stakeholder insights.
print("Enhanced DataFrame with generated details and issue category:")
display_cols = free_text_columns + [f"{col}_tags" for col in free_text_columns] + \
               ['combined_tags', 'issue_category'] + target_columns
print(complaints_df[display_cols].head())

# Save processed complaints data to CSV.
complaints_df.to_csv('data/processed_complaints.csv', index=False)
