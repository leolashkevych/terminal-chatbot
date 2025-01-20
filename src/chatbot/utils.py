import argparse
import logging
import time
import torch

from chatbot.constants import DEFAULT_MODEL_ID

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s",
    )

def set_logger_level(level):
    logging.getLogger().setLevel(level)


def pargs(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="HF Model ID to use for chatbot",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(*args, **kwargs)


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"Function {func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper


def check_mem(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print_gpu_memory()
        return result

    return wrapper


def print_cuda_setup():
    if torch.cuda.is_available():
        logging.info(f"CUDA Version: {torch.version.cuda}")
        logging.info(f"GPU Device: {torch.cuda.get_device_name()}")
        logging.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB"
        )


def print_gpu_memory():
    if torch.cuda.is_available():
        logging.info(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB")
        logging.info(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f}MB")


setup_logging()
