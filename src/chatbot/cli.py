#!/usr/bin/env python3

"""
Chatbot playground on Llama 3.2 abliterated model
"""

import logging
from chatbot.model import Model
from chatbot.utils import pargs, print_cuda_setup, set_logger_level


def chatbot(model_id):
    print_cuda_setup()
    model = Model(model_id)
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ["quit", "exit"]:
                exit(0)

            output_text = model.generate_response(user_input)
            print("\nBot:", output_text, "\n")

        except KeyboardInterrupt:
            logging.info("\nExiting chatbot...")
            break
        except Exception as e:
            logging.error(f"Error: {e}")


def main(*args, **kwargs):
    args = pargs(*args, **kwargs)
    if args.debug:
        set_logger_level(logging.DEBUG)
        logging.debug(f"Running with args: {args}")

    chatbot(model_id=args.model)


if __name__ == "__main__":
    main()