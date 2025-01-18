from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
import torch
import logging

class Model:
    def __init__(self, model_id):
        self._model, self._tokenizer = self.initialize_model(model_id)

    def initialize_model(self, model_id):
        accelerator = Accelerator()

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # apparently this is needed for some reason
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
            quantization_config=quantization_config,  # requires CUDA
            device_map="auto",
        )

        logging.info(f"Accelerator device: {accelerator.device}")
        model = accelerator.prepare(
            model
        )  # automatically moves the model to the correct device

        model.eval()
        logging.info(f"Model device: {next(model.parameters()).device}")
        return model, tokenizer
    
    @property
    def model(self):
        return self._model
    
    @property
    def tokenizer(self):
        return self._tokenizer