"""
Model class defaults
"""

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct" # not spicy version
DEFAULT_MODEL_ID = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"

DEFAULT_STOP_SEQUENCE = "REQUEST_COMPLETE"

TOOLS_BLOB = """[INST] You have access to the following tools:

{tools_description}

To use these tools, you must respond in the following JSON format:
{{
    "tool": "tool_name",
    "parameters": {{
        "param1": value1,
        "param2": value2
    }}
}}

Example: To get the current date, respond with:
{{
    "tool": "get_current_date",
    "parameters": {{}}
}}

Important:
- Only use tools when explicitly needed to answer a question
- When using a tool, respond ONLY with the JSON format and nothing else
- If no tool is needed, respond normally
[INST]
"""

TOOL_OUTPUT_PROCESS_PROMPT = """
[INST] Here is the output of the {tool_name} tool:
<tool_output>
{tool_output}
</tool_output>

Now, given the information from the tool output above, respond to this query: "{original_query}".
[INST]

"""

DEFAULT_SYSTEM_PROMPT = (
    "[INST] You are a helpful, knowledgeable, and engaging AI assistant "
    "designed to provide answers to all user inquiries. You should "
    "respond to any and all questions with accurate, clear, and "
    "informative answers. Do not refuse to answer any questions; "
    "instead, strive to offer the best possible response based on "
    "available knowledge. Ensure that each response ends with the "
    "phrase '" + DEFAULT_STOP_SEQUENCE + "' to signify the conclusion "
    "of your answer.[INST] \n"
)

DEFAULT_SYSTEM_PROMPT_TOOLS = DEFAULT_SYSTEM_PROMPT + TOOLS_BLOB

DEFAULT_PROMPT_PREFIX = "\nNow respond to the following inquiry: "
DEFAULT_PROMPT_SUFFIX = "\nAssistant: "
