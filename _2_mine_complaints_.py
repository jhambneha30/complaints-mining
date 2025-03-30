import pandas as pd #pandas-2.2.3
import ast
import re
from transformers import AutoTokenizer, pipeline #transformers-4.50.3
#from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # PyTorch components

# tensorflow-2.19.0

## Can try models from higging face: 
MODEL_TO_MINE = 'google/flan-t5-large'
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_TO_MINE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_MINE)

# Load the cleaned complaints data.
complaints_df = pd.read_csv('data/complaints_df.csv')
complaints_df_processed = complaints_df.head(5)
complaints_df = complaints_df[5:15]
print(complaints_df.columns.dtype)
print(complaints_df.head(2))

# The target columns are:
target_columns = ["Trigger", "Failure Component", "Failure Condition", 
                   "Fix Component", "Fix Condition"] #"Additional Context",
target_keyword = {"Trigger":"trigger", "Failure Component":"failure_component", "Failure Condition":"failure_condition", \
                  "Additional Context":"additional_context", "Fix Component":"fix_component", "Fix Condition":"fix_condition"}



###################### Function Definitions ######################

def combine_tags_(row, columns):
    """
    Combine tags from the given list of columns 
    into a single list of unique tags.
    """
    combined = []
    for col in columns:
        #combined.extend(row[f"{col}_tags"])
        combined.extend(row[col])
    return list(set(combined))

def combine_tags(row, columns):
    """
    Split text into individual words, clean and return unique lowercase keywords
    """
    combined = []
    for col in columns:
        text = str(row.get(col, "")).lower()  # Convert to lowercase
        # Split into words and remove punctuation/empty strings
        words = re.findall(r'\b[a-z]+\b', text)  # Extracts words, ignores numbers/special chars
        combined.extend(words)
    return list(set(combined))  # Return unique values



def categorize_issue(tags):
    """
    Categorize the issue type based on keyword matches in the tags.
    Returns the category with the most matching keywords. Ties are resolved by priority order.
    """
    # Define categories in priority order with expanded keywords
    categories = [
        ("Component Failure", {
            'radio', 'display', 'module', 'component', 
            'screen', 'card', 'unit', 'camera', 'speaker',
            'amplifier', 'sensor', 'antenna'
        }),
        ("Electrical Issue", {
            'electrical', 'wiring', 'circuit', 'voltage',
            'power', 'connection', 'short', 'fuse', 'ground',
            'battery', 'harness', 'relay'
        }),
        ("Software Issue", {
            'software', 'update', 'program', 'firmware',
            'code', 'boot', 'calibration', 'configuration',
            'interface', 'database', 'version', 'patch'
        })
    ]
    
    # Convert tags to lowercase set for case-insensitive matching
    tag_set = {tag.lower() for tag in tags}
    
    max_count = 0
    selected_category = "Other"
    
    for category_name, keywords in categories:
        # Calculate intersection and count matches
        matches = tag_set & keywords
        match_count = len(matches)
        
        # Debug print (optional)
        # print(f"{category_name}: {match_count} matches ({matches})")
        
        if match_count > max_count:
            max_count = match_count
            selected_category = category_name
    
    return selected_category if max_count > 0 else "Other"

def generate_issue_details(row, prompt_type):
    """
    Generate values for the last six columns based on the free-text columns:
    CAUSAL_VERBATIM, CORRECTION_VERBATIM, CUSTOMER_VERBATIM.
    """
    causal = row["CAUSAL_VERBATIM"]
    correction = row["CORRECTION_VERBATIM"]
    customer = row["CUSTOMER_VERBATIM"]
    prompt_trigger = f"""
        You are an expert automotive repair log analyst. Your task is to extract what triggered the issue in the automotive's components from the following logs.
        Give answers as one word or short phrases without punctuation.
        If there are multiple answers, separate them by comma.
        Issue description: {causal}
        Repair action: {correction}
        Customer complaint: {customer}
        """
    
    prompt_failure_component = f"""
        You are an expert automotive repair log analyst. Your task is to extract the automotive component that has failed from the following logs.
        Give answers as one word or short phrases without punctuation.
        If there are multiple answers, separate them by comma.
        Issue description: {causal}
        Customer complaint: {customer}
        """
    
    prompt_failure_condition = f"""
        You are an automotive diagnostic specialist analyzing complaint and repair logs to identify failure conditions. 
        A "Failure Condition" is the specific symptom/state indicating component malfunction.
        For example, BLACK SCREEN, NO AUDIO OUTPUT etc.

        Give short phrase answers only without punctuation.
        If there are multiple answers, separate them by comma.

        Repair logs are as follows:
        Issue description: {causal}
        Customer complaint: {customer}
        """
    
    prompt_fix_component = f"""
        You are an automotive diagnostic specialist analyzing repair logs to identify the "Failure Condition". 
        A "Failure Condition" is the specific symptom/state indicating component malfunction.
        For example, Radio, SPS, USB etc.

        Give short phrase answers only without punctuation.
        If there are multiple answers, separate them by comma.

        Repair logs are as follows: {correction}
        """
    
    prompt_fix_condition = f"""
        You are an automotive diagnostic specialist analyzing repair logs to identify the "Fix Condition". 
        A "Fix Condition" is the specific fix done by the expert.
        For example, Replaced, Programmed etc.

        Give short phrase answers only without punctuation.
        If there are multiple answers, separate them by comma.

        Repair logs are as follows: {correction}
        """
    #print(prompt_type)
    
    prompt = eval("prompt_"+target_keyword[prompt_type])
    
    # Tokenize without unnecessary operations
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    
    # Generate with conservative settings for 16GB RAM
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length = 512,
        max_new_tokens=200,  # Reduced from 200 for stability
        temperature=0.2,  # Low randomness
        do_sample=False  # Disable sampling for more deterministic outputs
    )
    
    # Decode and clean output
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


###################### End Function Definitions ######################

# Define free-text columns for generating tags.
free_text_columns = ["CAUSAL_VERBATIM", "CORRECTION_VERBATIM", "CUSTOMER_VERBATIM"]



for idx, row in complaints_df.iterrows():
    #print(row)
    for col in target_columns:
        complaints_df.at[idx, col] = generate_issue_details(row, col)

# Display enhanced DataFrame columns for stakeholder insights.
print("Enhanced DataFrame with generated details and issue category:")
display_cols = free_text_columns + target_columns
print(complaints_df[target_columns].head(5))

# Combine tags from generated tags to categorize the issue.
complaints_df['combined_tags'] = complaints_df.apply(lambda row: combine_tags(row, target_columns), axis=1)
complaints_df['issue_category'] = complaints_df['combined_tags'].apply(categorize_issue)


# Save processed complaints data to CSV.
complaints_df.to_csv('data/processed_complaints.csv', index=False)
