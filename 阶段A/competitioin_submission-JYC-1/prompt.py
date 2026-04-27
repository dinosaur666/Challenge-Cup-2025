SYSTEM = {
    "math": "You are a good assistant. Solve the following math problem step by step. " \
        "The first line of your response should be the format '<think> THINK </think>'(without quotes)," \
        "where THINK is your thinking process. The last line of your response should be in the format "\
        "'<answer>Answer: ANSWER</answer>'(without quotes), where ANSWER is your final answer. " \
        "Indicate your final answer in boxed LaTeX. For example, if the final answer is \sqrt{3}, write it as \\boxed{\sqrt{3}}.",

    "code-generate": "You are a good assistant. Read the following python function signature and docstring, and fully implement the function described."\
        " Your response should only contain the code for this function. The first line of your response should be the format"\
        " '<think> THINK </think>'(without quotes), where THINK is your thinking process.The last paragraph in your "\
        "response should be in the format '<answer>\n```python\n CODE```\n</answer>', where FUNCTION is the code for this function.",

    "generic-generate": "You are a good Chinese and English assistant. Read the following passage, think step by step and answer the question. "\
        "The first line of your response should be the format '<think> THINK </think>'(without quotes), where THINK is your thinking process. " \
        "The last line of your response should be in the format '<answer>Answer: ANSWER</answer>', where ANSWER is the answer to the question.",

    "choice": "You are a good Chinese and English assistant. Answer the following multiple choice question. There is only one correct answer."\
        " The first line of your response should be the format '<think> THINK </think>'(without quotes),"\
        " where THINK is your thinking process. The last line of your response should be in the format" \
        "'<answer>Answer: LETTER</answer>' (without quotes), where LETTER is one of A, B, C, D. Think step by step before answering."
}

PROMPT_TEMPLATES = {
    "math": """<|im_start|>system\n{system}\n<|im_end|>
    <|im_start|>user\n{eq0}\n<|im_end|>\n<|im_start|>assistant\n{ea0}\n<|im_end|>
    <|im_start|>user\n{eq1}\n<|im_end|>\n<|im_start|>assistant\n{ea1}\n<|im_end|>
    <|im_start|>user\n{question}\nRemember to put your answer in \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n""",
    "code-generate": """<|im_start|>system\n{system}\n<|im_end|>
    <|im_start|>user\n{eq0}\n<|im_end|>\n<|im_start|>assistant\n{ea0}\n<|im_end|>
    <|im_start|>user\n{eq1}\n<|im_end|>\n<|im_start|>assistant\n{ea1}\n<|im_end|>
    <|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n""",
    "generic-generate": """<|im_start|>system\n{system}\n<|im_end|>
    <|im_start|>user\n{eq0}\n<|im_end|>\n<|im_start|>assistant\n{ea0}\n<|im_end|>
    <|im_start|>user\n{eq1}\n<|im_end|>\n<|im_start|>assistant\n{ea1}\n<|im_end|>
    <|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n""",
    "choice": """<|im_start|>system\n{system}\n<|im_end|>
    <|im_start|>user\n{eq0}\n<|im_end|>\n<|im_start|>assistant\n{ea0}\n<|im_end|>
    <|im_start|>user\n{eq1}\n<|im_end|>\n<|im_start|>assistant\n{ea1}\n<|im_end|>
    <|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n""",
}


# MetaMathQA:  "Write a response that appropriately completes the request.\n\n" "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."