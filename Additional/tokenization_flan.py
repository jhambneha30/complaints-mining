from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the tokenizer and model for FLAN-T5-small.
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

# Define your prompt.
prompt = (
    "Based on the following texts:\n"
    "CAUSAL_VERBATIM: Unable to duplicate the concern at this time radio and display both working normally...\n"
    "CORRECTION_VERBATIM: Customer reports that radio and screen is intermittently inop Inspect and advise\n"
    "CUSTOMER_VERBATIM: ...\n\n"
    "Generate the following details in JSON format with keys exactly as shown:\n"
    '{ "Trigger": "<value>", "Failure Component": "<value>", "Failure Condition": "<value>", '
    '"Additional Context": "<value>", "Fix Component": "<value>", "Fix Condition": "<value>" }\n\n'
    "Now generate the values based on the provided texts."
)

# Tokenize the prompt.
inputs = tokenizer(prompt, return_tensors="pt")

# Generate output (with max_length adjusted as needed).
outputs = model.generate(inputs.input_ids, max_length=256)

# Decode the output.
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Output:")
print(generated_text)


'''
Explanation
Manual Tokenization:
In this code, we load the T5Tokenizer and T5ForConditionalGeneration from the FLAN-T5-small model. We then tokenize the prompt using the tokenizer() function with return_tensors="pt" to convert it into PyTorch tensors.

Model Inference:
The tokenized input is passed to the model’s generate method. The output tokens are then decoded back into a string using the tokenizer's decode method, skipping special tokens.

This approach gives you full control over tokenization and generation, even though the Hugging Face pipeline abstracts these steps by default.
'''
