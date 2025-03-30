# Mistral-7B-Instruct Example
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

prompt = """
[INST] Extract Failure Component and Fix Component:
Log: "Screen black when reversing. Replaced camera."
[/INST]
{"Failure Component": ["Camera"], "Fix Component": ["Camera"]}
[INST]
Log: "Radio unresponsive. Reprogrammed module."
[/INST] 
"""
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))