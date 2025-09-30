from openai import OpenAI
import time
import os
import sys
from simhash import Simhash
import re
current_dir = os.path.dirname(os.path.abspath(__file__))
mutipl_dir = os.path.join(current_dir, "../MutiPL-E", "evaluation/src")
sys.path.append(mutipl_dir)
from containerized_eval import eval_string_script

def extract_python_code(inputs):
    inputs = inputs.split("### Unit tests below ###")[0]
    inputs = inputs.replace("""
    \"\"\"\"\"\"
    ### Canonical solution below ###""","")
    return inputs.strip()

def extract_python_code_with_test(inputs):
    inputs = inputs.replace("""
    \"\"\"\"\"\"
    ### Canonical solution below ###""","")
    inputs = inputs.replace("\n### Unit tests below ###","")
    return inputs.strip()

def eval_code(lang, code):
    return eval_string_script(lang, code)

def replace_mark(code,lang):
    code = code.split(f"```{lang}\n")[-1]
    code = code.replace(f"```{lang}\n","")
    code = code.replace("```","")
    return code.rstrip()

def extract_code(code,lang,index):
    pattern = r'```{}\s*(.+?)\s*```'.format(lang)
    matches = re.findall(pattern, code, re.DOTALL)
    try:
        res = matches[index]
    except:
        res = ""
    return res

def output_gpt_prompt(message):
    s = ""
    for i in message:
        s += f"- role:{i['role']}\n"
        s += "  content: |\n"
        s += "    "+i["content"].replace("\n","\n    ")+"\n"
    return s

def is_similar(text1, text2, get_features, threshold=3):
    simhash1 = Simhash(get_features(text1))
    simhash2 = Simhash(get_features(text2))
    return simhash1.distance(simhash2) <= threshold

def get_examples(ds_right,ds_python,shot_num):
    data_type = {"Dict":[],"List":[],"str":[]}
    for sample in ds_right:
        cur_id = sample["id"]
        code = ds_python[cur_id]["code"]
        code = code.split("\n")
        for line in code:
            if "def f(" in line:
                for cur_type in data_type:
                    if cur_type in line:
                        data_type[cur_type].append(sample)
                        break
    examples = []
    examples_id = []
    tot_list = [sublist for sublist in data_type.values()]
    tot_data = [item for sublist in zip(*(tot_list)) for item in sublist]
    for index,sample in enumerate(tot_data):
        examples.append(sample["code"])
        examples_id.append(sample["id"])
    return examples[:shot_num],examples_id[:shot_num]

def gpt_response(message, api_key, model_name, tmp=0, stop=[], base_url="", max_completion_tokens=None,
                top_p=None, top_k=None, presence_penalty=None, repetition_penalty=None):
    """
    Call OpenAI-compatible API with inference parameters.

    Args:
        message: Chat messages
        api_key: API key
        model_name: Model name
        tmp: Temperature
        stop: Stop sequences
        base_url: Base URL for API
        max_completion_tokens: Max completion tokens (preferred over deprecated max_tokens)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        presence_penalty: Presence penalty
        repetition_penalty: Repetition penalty (mapped to frequency_penalty for OpenAI API)

    Returns:
        Generated text
    """
    if base_url == "":
        client = OpenAI(api_key=api_key)
    else:
        client = OpenAI(api_key=api_key, base_url=base_url)

    # Build kwargs for API call
    kwargs = {
        "model": model_name,
        "messages": message,
        "temperature": tmp,
        "stop": stop,
    }

    # Add max_completion_tokens if provided (preferred over deprecated max_tokens)
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens

    # Add optional parameters if provided
    if top_p is not None:
        kwargs["top_p"] = top_p
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    # OpenAI API uses frequency_penalty, not repetition_penalty
    if repetition_penalty is not None:
        kwargs["frequency_penalty"] = repetition_penalty

    # Note: top_k is not supported by OpenAI API, but may be supported by vLLM
    # vLLM supports it as an extra_body parameter
    if top_k is not None:
        kwargs["extra_body"] = {"top_k": top_k}

    flag = 0
    while flag != 1:
        try:
            completion = client.chat.completions.create(**kwargs)
            flag = 1

        except Exception as err:
            print("error : ", err)
            flag = 0
            time.sleep(1)
    return completion.choices[0].message.content

input_map = {
    "cpp":"candidate(????)",
    "cs":"F(????)",
    "d":"candidate(????)",
    "go":"candidate(????)",
    "java":"f(????)",
    "jl":"candidate(????)",
    "js":"candidate(????)",
    "lua":"candidate(????)",
    "php":"candidate(????)",
    "pl":"$candidate->(????)",
    "py":"candidate(????)",
    "r":"candidate(????)",
    "rb":"candidate.call(????)",
    "rkt":"(candidate ????)",
    "rs":"candidate(????)",
    "scala":"f(????)",
    "sh":"$(candidate ????)",
    "swift":"f(????)",
    "ts":"candidate(????)"
}
lang_comment = {
    "java": "    // \n",
    "cpp": "// \n",
    "cs": "    // \n    // ",
    "d":"\n/*\n\n*/",
    "go":"// \n",
    "jl":"\"\"\"\"\"\"\n",
    "js":"//\n",
    "lua":"",
    "php":"// \n",
    "pl":"# \n",
    "r":"",
    "rb":"# \n",
    "rkt":"",
    "rs":"/// \n",
    "scala":"    // \n",
    "sh":"\n# \n#",
    "swift":"\n/// ",
    "ts":"//\n",
    "py":"\"\"\"\"\"\"\n"
}

stop_tokens = {
    "java":[
    "    }\n    //",
    "    }\n    p",
    "    }\n\n",
    "    }\n}",
    ],
    "cpp": [
        "\n}"
    ],
    "go": [
        # "\nfunc",
        # "struct",
        # "\n// ",
        "\n```"
    ],
    "cs": [
        "\n    }\n"
    ],
    "d": [
        # "\n\n",
        # "\nvoid",
        # "\nint",
        "}\n```"
    ],
    "jl": [
        # "\nfunction",
        # "\nmacro",
        # "\n\n",
        "end\n```"
    ],
    "js": [
        # "\nfunction ",
        # "\n/*",
        # "\nconsole.log",
        "}\n```"
    ],
    "php": [
        # "\nfunction",
        # "\n#",
        # "\n?>",
        "}\n```"
    ],
    "pl": [
        # "\nsub",
        # "\n#",
        # "\n\n",
        "}\n```"
    ],
    "lua": [
        # "\nlocal",
        # "\nfunction",
        # "\n--",
        "end\n```"
    ],
    "r": [
        # "\n#",
        "}\n```"
    ],
    "rb": [
        # "\n#",
        # "\n\n",
        "end\n```"
    ],
    "rkt": [
        # "\n(define ",
        # "\n#|",
        # "\n(",
        ")\n```"
    ],
    "rs": [
        "\n}"
    ],
    "scala": [
    "\n    }\n"
    ],
    "sh": [
        "\n}"
    ],
    "swift": [
        "\n}"
    ],
    "ts": [
        # "\nfunction ",
        # "\n/*",
        # "\nclass",
        "}\n```"
    ],
}

lang_map = {
    "java": "java",
    "cpp": "cpp",
    "go": "go_test.go",
    "cs": "cs",
    "d": "humaneval_to_dlang.py",
    "jl": "julia",
    "js": "javascript",
    "php": "php",
    "pl": "pl",
    "py": "python",
    "r": "humaneval_to_r.py",
    "lua": "lua",
    "rb": "ruby",
    "rkt": "racket",
    "rs": "rust",
    "scala": "scala",
    "sh": "sh",
    "swift": "swift",
    "ts": "ts"
}

iterating_stops = {
    "cpp":["}\nint main()","}\n\nint main()"],
    "cs":["    }\n    public static void Main","    }\n\n    public static void Main"],
    "rkt":["(require rackunit)\n\n(define (test-humaneval)","(require rackunit)\n(define (test-humaneval)"],
    "d":["unittest"],
    "go":["func TestF(t *testing.T)"],
    "java":["    }\n    public static void main","    }\n\n    public static void main"],
    "jl":["using Test"],
    "js":["const assert"],
    "lua":["lu = require('luaunit')"],
    "php":["function candidate(...$args)"],
    "pl":["use Test::Deep;"],
    "py":["def check(candidate):"],
    "r":["test_humaneval"],
    "rb":["require 'test/unit'"],
    "rs":["}\n\nfn main()","}\nfn main()","}\n\n\nfn main()"],
    "scala":["    }\n    def main","    }\n\n    def main"],
    "sh":["}\n\ncandidate()","}\ncandidate()","}\n\n\ncandidate()"],
    "swift":["}\n\n\nfunc ==(left","}\n\nfunc ==(left","}\nfunc ==(left","}\n\n\n\nfunc ==(left"],
    "ts":["declare var require: any;"],
}