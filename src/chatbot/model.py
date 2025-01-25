import torch
import json
import logging
from accelerate import Accelerator
from copy import deepcopy
from chatbot.utils import timeit
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from chatbot.tools import Tools
from chatbot.constants import (
    DEFAULT_STOP_SEQUENCE,
    DEFAULT_PROMPT_PREFIX,
    DEFAULT_PROMPT_SUFFIX,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT_TOOLS,
    TOOL_OUTPUT_PROCESS_PROMPT,
)


class Model:
    def __init__(self, model_id, use_tools=True):
        self._model, self._tokenizer = self.initialize_model(model_id)
        self._prompt_prefix = DEFAULT_PROMPT_PREFIX
        self._prompt_suffix = DEFAULT_PROMPT_SUFFIX
        self._system_prompt = DEFAULT_SYSTEM_PROMPT
        self._stop_sequence = DEFAULT_STOP_SEQUENCE
        if use_tools:
            self._tools = Tools()
            self._system_prompt = DEFAULT_SYSTEM_PROMPT_TOOLS.format(
                tools_description=self._tools.get_tools_description()
            )

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

    def get_tokenizer_info(self):
        vocab = self.tokenizer.get_vocab()
        added_vocab = self.tokenizer.get_added_vocab().keys()
        vocab_files_names = self.tokenizer.vocab_files_names
        logging.debug(f"Tokenizer files: {vocab_files_names}")
        logging.debug(f"Tokenizer added vocab: {list(added_vocab)}")

        return (vocab, added_vocab, vocab_files_names)

    @timeit
    def generate_response(
        self,
        prompt,
        max_length=512,
        max_sequences=1,
        temp=0.3,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        add_system_prompt=True,
    ):
        accelerator = Accelerator()

        # preserve unprocessed user prompt
        _prompt = deepcopy(prompt)
        if add_system_prompt:
            _prompt = self.prepare_prompt(_prompt)

        inputs = self.tokenizer(
            _prompt,
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
                max_new_tokens=max_length,
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
        decoded_response = self.extract_response(decoded_response)

        if hasattr(self, "_tools"):
            decoded_response = self.process_response(
                response=decoded_response, prompt=prompt
            )

        return decoded_response

    def process_response(self, response, prompt):
        # Try to process response as JSON in case a tool was requested
        try:
            # try to get rid of anything non-json
            r = response[response.find("{") : response.rfind("}") + 1]

            tool_command = json.loads(r)
            if (
                isinstance(tool_command, dict)
                and "tool" in tool_command
                and "parameters" in tool_command
            ):
                result = self._tools.execute_tool(
                    tool_command["tool"], **tool_command["parameters"]
                )
                return self.post_process_tool_out(
                    original_prompt=prompt,
                    original_response=response,
                    tool_name=tool_command["tool"],
                    tool_response=result,
                )
        except json.JSONDecodeError:
            pass

        return response

    def post_process_tool_out(
        self, original_prompt, original_response, tool_name, tool_response
    ):
        prompt = self.prepare_prompt(original_prompt)
        prompt += original_response + "\n"
        prompt += TOOL_OUTPUT_PROCESS_PROMPT.format(
            tool_name=tool_name,
            tool_output=tool_response,
            original_query=original_prompt,
        )
        prompt += self._prompt_suffix
        # do not add system prompt, since we are constructing one manually here
        return self.generate_response(prompt, add_system_prompt=False)

    def prepare_prompt(self, user_input):
        return (
            self.system_prompt + self._prompt_prefix + user_input + self.prompt_suffix
        )

    def extract_response(self, response):
        return (
            response.split(self.prompt_suffix)[-1].split(self.stop_sequence)[0].strip()
        )
