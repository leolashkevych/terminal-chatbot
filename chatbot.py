#!/usr/bin/env python3

"""
Chatbot playground on Llama 3.2 abliterated model
"""

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

def chatbot():
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct" # not spicy version
    model_id = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"

    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # apparently this is needed for some reason 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        device_map='auto' 
    )

    model = accelerator.prepare(model)  # automatically moves the model to the correct device

    model.eval()


    while True:
        try:
            user_input = input("> ")
            if user_input.lower() in ['quit', 'exit']:
                break
            
            inputs = tokenizer(
                user_input,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            )

            # Ensure inputs are moved to the same device as the model (mps or cuda)
            inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

            with torch.no_grad():
                response = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=1000,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )

            output_text = tokenizer.decode(response[0], skip_special_tokens=True)

            print("\nBot:", output_text, "\n")
            
        except KeyboardInterrupt:
            print("\nExiting chatbot...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chatbot()
