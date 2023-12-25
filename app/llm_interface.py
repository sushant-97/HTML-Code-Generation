# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "<username>/llama2-fine-tuned-dolly-15k"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Generate text
def query_llm(prompt):
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text
    output = model.generate(input_ids, do_sample=True, max_length=1000, top_p=0.95, top_k=60)

    # Decode output
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    return output