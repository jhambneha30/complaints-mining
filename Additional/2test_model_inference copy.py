from unittest.util import _MAX_LENGTH
from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration, AutoTokenizer  # PyTorch components

import torch
## torch-2.6.0
## Note: T5Tokenizer requires SentencePiece library: sentencepiece-0.2.0
# Load PyTorch model & tokenizer
#model_id = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
#model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
#tokenizer = AutoTokenizer.from_pretrained(model_id)
## Testing
causal = "INSPECTED FOR CONCERN UNABLE TO DUPLCIATE THE FIRST TIME SCANNED FOR CODES NONE STORED SEARCHED FOR BULLETINS AND FOUND PIT   PERFORMED   CALLED XM AND WAS TOLD NO REPAIR AT THIS TIME CUSTOMER CALLED AND WAS TOLD OVER THE AIR UPDATE HAD BEEN PERFORMEDALREADY CUSTOMER IS STILL HAVING CONCERN AFTER UPDATE OPENED TAC CASE     WHILE WAITING TO HEAR BACK FROM TAC CONTINUEDTO TRY TO DUPLICATE CONCERN DID DUPLICATE CONCERN AND SWAPPED XM ANTENNA BUT STILL HAD SAME RESULT TAC ESCALATE TO BRAND QUALITYNECESSARY TO PULL AND CHECK RADIO CONTROL INSPECTED CONNECTIONS LEADING FROM RADIO TO RADIO CONTROL ATTEMPTED TO PERFORM ANOTHERUPDATE STILL SAME CONCERN ESCALATED TO DMA TO TRY AND GET A QUICKER RESOLUTION AND WAS INSTRUCTED BY BRAND QUALITY TAC TO REPLACETHE RADIO RADIO WAS ON BACK ORDER FOR SEVERAL DAYS DID CALL SPECMO AND WAS TOLD THIS IS A NEW PART"
correction = "INSTALLED NEW RADIO AND PROGRAMMED PERFORMED REPAIR VERIFICATION AND FOUND WORKING AS DESIGNED AT THIS TIME PROGRAM         TECH"
customer = "C S INTERMITTENTLY WHEN STARTING VEHICLE CHANNEL NOT AVAILABLE MESSAGE WILL COME UP ON RADIO SCREEN PRESETS WILL TURN GRAY ANDBECOME UNAVAILABLE CUSTOMER CAN SHUT VEHICLE OFF AND RESTART VEHICLE FEW MINUTES LATER AND ALL IS WORKING NORMALLY THIS ONLYOCCURS WITH XM RADIO STATIONS ON CUSTOMERS LAST VISIT WE DID   CUSTOMER HAS SINCE CONTACTED XM AS OUTLINED IN   ANDWAS TOLD THAT THE OVER THE AIR UPDATE HAD BEEN PERFORMED BETWEEN             CUSTOMER IS STILL HAVING THE SAME ISSUES HE WASTOLD BY XM THAT IT WAS A MODULE ISSUE AND RADIO WOULD HAVE TO BE REPLACED PLEASE OPEN A TAC CASE AND SEE WHAT THEIR ANSWER IS"


def generate_tags(causal, correction, customer, ):
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

# Test
result = generate_tags(
    "SCREEN GOING BLACK WITH PICTURE OF CAMERA WITH LINE THOUGH IT, when reversing",
    "REPLACED RADIO",
    "BACKUP CAMERA getting switched on INTERMITTENTLY"
)
print(result)





prompt = f"""
        ### SYSTEM MESSAGE
        You are an expert automotive repair log analyst. Your task is to extract structured data from the following repair log input. Strictly follow the output format exactly without any extra text or formatting.

        ### INPUT DATA
        - CAUSAL: {causal}
        - CORRECTION: {correction}
        - CUSTOMER: {customer}

        ### DEFINITIONS
        - CAUSAL: Description of the issue or problem.
        - CORRECTION: The repair action taken.
        - CUSTOMER: Customer's description of the complaint.

        ### OUTPUT KEYS AND GUIDANCE
        - "Trigger": Describe the condition or scenario that triggers the issue. Example: "When Reversing"
        - "Failure Component": Identify the faulty component. Example: "Display"
        - "Failure Condition": Describe the observed failure. Example: "Black Screen"
        - "Additional Context": Note any extra information about how the problem occurs. Example: "While Driving"
        - "Fix Component": Identify which component was repaired or replaced. Example: "Radio"
        - "Fix Condition": Specify the repair action taken. Example: "Replaced"

        ### EXAMPLE
        Input:
            CAUSAL: "radio screen went black after software update"
            CORRECTION: "replaced radio module and reprogrammed"
            CUSTOMER: "radio stops working intermittently"
        Expected Output:
        {{
            "Trigger": ["After Software Update"],
            "Failure Component": ["Radio", "Display"],
            "Failure Condition": ["Black Screen", "Intermittent Operation"],
            "Additional Context": ["During Multimedia Use"],
            "Fix Component": ["Radio Module"],
            "Fix Condition": ["Replaced", "Reprogrammed"]
        }}

        ### CURRENT TASK
        Using the above definitions and examples, extract the data for the given input exactly in JSON format with the following keys:
        - Trigger
        - Failure Component
        - Failure Condition
        - Additional Context
        - Fix Component
        - Fix Condition

        ### INPUT DATA (for current task)
        CAUSAL: {causal}
        CORRECTION: {correction}
        CUSTOMER: {customer}

        Output the answer as valid JSON.
        """
