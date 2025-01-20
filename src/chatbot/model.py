import torch
import logging
from accelerate import Accelerator
from chatbot.utils import timeit
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from chatbot.constants import (
    DEFAULT_STOP_SEQUENCE,
    DEFAULT_PROMPT_SUFFIX,
    DEFAULT_SYSTEM_PROMPT,
)


class Model:
    def __init__(self, model_id):
        self._model, self._tokenizer = self.initialize_model(model_id)
        self._prompt_suffix = DEFAULT_PROMPT_SUFFIX
        self._system_prompt = DEFAULT_SYSTEM_PROMPT
        self._stop_sequence = DEFAULT_STOP_SEQUENCE

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def prompt_suffix(self):
        return self._prompt_suffix

    @property
    def system_prompt(self):
        return self._system_prompt

    @property
    def stop_sequence(self):
        return self._stop_sequence

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

    @timeit
    def generate_response(
        self,
        prompt,
        max_length=512,
        max_sequences=1,
        temp=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
    ):
        accelerator = Accelerator()

        inputs = self.tokenizer(
            self.construct_prompt(prompt),
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        # Ensure inputs are moved to the same device as the model (mps or cuda)
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

        with torch.no_grad():
            response = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_return_sequences=max_sequences,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                tokenizer=self.tokenizer,
                stop_strings=[self.stop_sequence],
            )

        decoded_response = self.tokenizer.decode(response[0], skip_special_tokens=True)
        logging.debug(f"Full LLM response: {decoded_response}")

        return self.extract_response(decoded_response)

    def construct_prompt(self, user_input):
        return self.system_prompt + user_input + self.prompt_suffix

    def extract_response(self, response):
        return (
            response.split(self.prompt_suffix)[-1].split(self.stop_sequence)[0].strip()
        )
