# Description: Interface for the llama2-fine-tuned-jawerty_html_dataset model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
from types import SimpleNamespace

model_id = "llama2-fine-tuned-jawerty_html_dataset"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Generate code
def generate(prompt, model_id, model, tokenizer):
    gen_config = GenerationConfig.from_pretrained(model_id)
    test_config = SimpleNamespace(
        max_new_tokens=256,
        gen_config=gen_config)

    tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
    with torch.inference_mode():
        output = model.generate(tokenized_prompt, 
                            max_new_tokens=test_config.max_new_tokens, 
                            generation_config=test_config.gen_config)
        
    return tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)
