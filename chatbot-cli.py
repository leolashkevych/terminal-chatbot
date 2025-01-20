#!/usr/bin/env python3

"""
Chatbot playground on Llama 3.2 abliterated model
"""

import logging
from src.chatbot.model import Model
from src.chatbot.utils import print_cuda_setup


def chatbot(model_id):
    print_cuda_setup()
    model = Model(model_id)
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ["quit", "exit"]:
                break

            output_text = model.generate_response(user_input)
            print("\nUser:", output_text, "\n")

        except KeyboardInterrupt:
            logging.info("\nExiting chatbot...")
            break
        except Exception as e:
            logging.error(f"Error: {e}")


if __name__ == "__main__":
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct" # not spicy version
    model_id = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"

    chatbot(model_id=model_id)
