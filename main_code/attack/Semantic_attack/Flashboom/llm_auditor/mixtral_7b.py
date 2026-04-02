from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch, os
from llm_auditor.base_auditor import BaseAuditor


class Mixtral(BaseAuditor):
    def __init__(self, max_new_tokens=300, **kwargs):
        super().__init__(
            max_new_tokens=max_new_tokens,
            begin_token="[INST]",
            end_token="[/INST]",
            support_system_prompt = True,
            linebreak='<0x0A>',
            **kwargs
        )
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.2"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )
                
        max_memory = {
            0:'30GiB',
        }


        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id
        )