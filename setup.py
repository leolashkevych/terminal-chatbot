from setuptools import setup, find_packages

setup(
    name="terminal-chatbot",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "llama-stack",
        "flake8",
        "accelerate",
        "transformers",
        "torch",
        "bitsandbytes",
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "duckduckgo-search"
    ],
    entry_points={
        "console_scripts": [
            "chatbot=chatbot.cli:main",
        ],
    },
    description="A terminal-based chatbot using a refusal-vector ablated LLM",
    python_requires=">=3.11",
)
