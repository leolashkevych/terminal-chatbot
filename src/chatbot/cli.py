#!/usr/bin/env python3

"""
Chatbot playground on Llama 3.2 abliterated model
"""

import logging
from chatbot.model import Model
from chatbot.utils import pargs, print_cuda_setup, set_logger_level


def chatbot(args):
    print_cuda_setup()
    model = Model(args.model, use_tools=args.agents)
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ["quit", "exit"]:
                exit(0)

            output_text = model.generate_response(user_input, temp=args.temp)
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

    chatbot(args)


if __name__ == "__main__":
    main()