import torch
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class ModelManager:
    def __init__(self, model_name='meta-llama/Llama-2-7b-hf'):
        self.model_name = model_name
        self.bnb_config = self.create_bnb_config()
        self.load_and_prepare_model()

    def create_bnb_config(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    def load_and_prepare_model(self):
        n_gpus = torch.cuda.device_count()
        max_memory = f'{40960}MB'

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            max_memory={i: max_memory for i in range(n_gpus)},
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Apply PEFT
        modules = self.find_all_linear_names(self.model)
        peft_config = self.create_peft_config(modules)
        self.model = get_peft_model(self.model, peft_config)

    def find_all_linear_names(self, model):
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def create_peft_config(self, modules):
        return LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
