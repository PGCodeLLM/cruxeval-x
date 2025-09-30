import json
from transformers import AutoTokenizer
import argparse
from datasets import load_dataset
from prompt import crux_input_prompt_chat,crux_output_prompt_chat
from untils import eval_code,lang_map,input_map,iterating_stops,gpt_response
import json
from tqdm import tqdm
import os


def read_data(file_path):
    with open(file_path, 'r',encoding='utf-8') as f:
        data = json.load(f)
    return data

def gen_result(examples, model: str, args, progress_file, current_progress):
    """
    Generate results with progress tracking.

    Args:
        examples: List of examples
        model: Model name
        args: Arguments
        progress_file: Path to progress file
        current_progress: Current cumulative progress count

    Returns:
        examples with generations, updated progress count
    """
    stop = ["[/ANSWER]"]
    prompts = [ex["prompt"] for ex in examples]
    for i in tqdm(range(len(examples)),total= len(examples)):
        examples[i]['generation'] = gpt_response(
            prompts[i],
            api_key=args.api_key,
            model_name=args.model_name,
            tmp=args.tmp,
            stop=stop,
            base_url=args.base_url,
            max_completion_tokens=args.max_output_tokens if args.max_output_tokens else None,
            top_p=args.top_p if args.top_p else None,
            top_k=args.top_k if args.top_k else None,
            presence_penalty=args.presence_penalty if args.presence_penalty else None,
            repetition_penalty=args.repetition_penalty if args.repetition_penalty else None
        )

        # Write progress every 10 samples or on last sample
        if (i + 1) % 10 == 0 or i == len(examples) - 1:
            current_progress += 10 if (i + 1) % 10 == 0 else (i + 1) % 10
            with open(progress_file, 'w') as f:
                json.dump({
                    "done": current_progress,
                    "total": args.total_tasks,
                    "timestamp": time.time()
                }, f)

    return examples, current_progress

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--langs', type=str, default="['java', 'cpp', 'cs', 'd', 'go', 'jl', 'js', 'lua', 'php', 'pl', 'py', 'r', 'rb', 'rkt', 'rs', 'scala', 'sh', 'swift', 'ts']",
                       help='List of languages as string (e.g., "[\'py\', \'java\']") or comma-separated (e.g., "py,java") or "all"')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument('--tmp', type=float, default=0, help='Temperature')
    parser.add_argument('--tot_data_num', type=int, default=800)

    # Inference parameters
    parser.add_argument('--top_p', type=float, default=None, help='Top-p (nucleus sampling)')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling')
    parser.add_argument('--max_output_tokens', type=int, default=None, help='Max output tokens')
    parser.add_argument('--presence_penalty', type=float, default=None, help='Presence penalty')
    parser.add_argument('--repetition_penalty', type=float, default=None, help='Repetition penalty')

    parser.add_argument('--data_root', type=str, default=f'./datasets/cruxeval-x')
    parser.add_argument('--data_input_output', type=str, default=f'./datasets/cruxeval_preprocessed')
    parser.add_argument('--example_root', type=str, default=f'./datasets/examples')
    parser.add_argument('--example_input_output', type=str, default=f'./datasets/examples_preprocessed')
    parser.add_argument('--output_dir', type=str, default=f'./infer_results')
    parser.add_argument('--progress_file', type=str, default=None,
                       help='Path to progress JSON file for tracking')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse languages - support multiple formats
    ALL_LANGS = ['py', 'java', 'cpp', 'cs', 'd', 'go', 'jl', 'js', 'lua',
                 'php', 'pl', 'r', 'rb', 'rkt', 'rs', 'scala', 'sh', 'swift', 'ts']

    if args.langs.lower() == 'all':
        langs = ALL_LANGS
    elif ',' in args.langs:
        # Comma-separated format: "py,java,cpp"
        langs = [lang.strip() for lang in args.langs.split(',')]
    else:
        # Python list format: "['py', 'java']"
        try:
            langs = eval(args.langs)
        except:
            langs = [args.langs.strip()]

    # Calculate total tasks for progress tracking
    args.total_tasks = len(langs) * 800 * 2  # 2 tasks (input + output) per language
    current_progress = 0

    print(f"Running CruXEval-X for languages: {langs}")
    print(f"Total tasks: {args.total_tasks}")
    print(f"Progress file: {args.progress_file}")
    for lang in langs:
        ds_data = read_data(f"{args.data_root}/{lang}.json")
        ds_data_input_output = load_dataset("json",data_files=f"{args.data_input_output}/{lang}.jsonl")["train"]
        example_data = read_data(f"{args.example_root}/{lang}.json")
        example_data_input_output = load_dataset("json",data_files=f"{args.example_input_output}/{lang}.jsonl")["train"]
        ds_data_input_output = {"input":{i["id"]:i["input_reasoning"] for i in ds_data_input_output},
                                "output":{i["id"]:i["output_reasoning"] for i in ds_data_input_output}}
        example_data_input_output = {"input":{i["id"]:i["input_reasoning"] for i in example_data_input_output},
                                    "output":{i["id"]:i["output_reasoning"] for i in example_data_input_output}}
        prompt_func = {
            "input":crux_input_prompt_chat,
            "output":crux_output_prompt_chat
        }
        prompts = {}
        # construct prompt
        for task_type in ["input","output"]:
            examples = []
            for example in example_data:
                code = example["code"]
                flag = False
                for stop in iterating_stops[lang]:
                    if stop in code:
                        code = code.split(stop)[0]
                        flag = True
                        break
                if not flag: assert Exception
                code_right_part = example_data_input_output[task_type][example["id"]]
                examples.append({"code":code+code_right_part,"answer":example[f"{task_type}s"]})

            cur_prompt = [] # {"prompt":prompt,"task_id":task_id}
            for sample in ds_data:
                if "code" not in sample: continue
                code = sample["code"]
                flag = False
                for stop in iterating_stops[lang]:
                    if stop in code:
                        code = code.split(stop)[0]
                        flag = True
                        break
                if not flag: assert Exception
                code_right_part = ds_data_input_output[task_type][sample["id"]]
                code += code_right_part
                cur_prompt.append({"prompt":prompt_func[task_type](lang,examples,code,args.model_name),
                                   "task_id":sample["id"]})
            prompts[task_type] = cur_prompt
        # generate answer
        for task_type in ["input","output"]:
            print(f"\n--- Processing task: {task_type} ---")
            cur_prompt = prompts[task_type]
            gen_res, current_progress = gen_result(cur_prompt, args.model_name, args, args.progress_file, current_progress)
            for cur_res in gen_res:
                cur_res["generation"] = cur_res["generation"].split("[ANSWER]")[-1]
                cur_res["generation"] = cur_res["generation"].replace("[/ANSWER","")
            # judege whether the answer is right
            outputs = [{"id":i,"res":0} for i in range(800)]
            for index,cur_res in tqdm(enumerate(gen_res),total=len(gen_res)):
                answer = cur_res["generation"].strip()
                cur_id = cur_res["task_id"]
                code = cur_prompt[index]["prompt"][-1]["content"].split(f"```{lang}")[-1]
                code = code.replace(f"```","")
                if task_type == "output":
                    code = code.replace("????",answer)
                else:
                    code = code.replace(input_map[lang], answer)
                exec_output = eval_code(lang_map[lang], code)
                if exec_output["status"] != "OK":
                    outputs[cur_id]["res"] = False
                    outputs[cur_id]["error"] = exec_output["status"]
                    outputs[cur_id]["error_message"] = exec_output["stderr"]
                    outputs[cur_id]["code"] = code
                    outputs[cur_id]["answer"] = answer
                else:
                    outputs[cur_id]["res"] = True
                    outputs[cur_id]["code"] = code
                    outputs[cur_id]["answer"] = answer

            output_file = f"{args.output_dir}/{lang}_{task_type}.json"
            with open(output_file,"w",encoding="utf-8") as f:
                json.dump(outputs,f,indent=4,ensure_ascii=False)
            print(f"Results saved to: {output_file}")

    # Mark progress as complete
    if args.progress_file:
        with open(args.progress_file, 'w') as f:
            json.dump({
                "done": args.total_tasks,
                "total": args.total_tasks,
                "timestamp": time.time(),
                "status": "completed"
            }, f)

    print(f"\n{'='*70}")
    print(f"All evaluations complete!")
    print(f"{'='*70}")