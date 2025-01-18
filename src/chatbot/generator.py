import torch
from src.chatbot.utils import timeit
from accelerate import Accelerator


class ChatbotGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

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
            prompt, return_tensors="pt", padding=True, return_attention_mask=True
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
            )

        return self.tokenizer.decode(response[0], skip_special_tokens=True)
