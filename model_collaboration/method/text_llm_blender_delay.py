import json
import multiprocessing
import os
import re
from typing import List, Tuple

from model_collaboration.data import eval
from model_collaboration.method import distributed_generation
from model_collaboration.utils import distributed_sft

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


METHOD_NAME = "text_llm_blender_delay"
METHOD_LOG_DIR = os.path.join("model_collaboration/logs", METHOD_NAME)

import os
from pathlib import Path
script_path = Path(__file__).resolve()
script_dir = script_path.parent.parent.parent
os.chdir(script_dir)

gpu_id = 0 #only one gpu

# global hyperparameters for generation
MAX_RESPONSE_LENGTH = None
TEMPERATURE = None
TOP_P = None
BATCH_SIZE = None
BIG_MODEL_MODE = None

# TODO delay code!
import time
DELAY = True

list_of_model_name = ["Qwen/Qwen2.5-7B-Instruct","allenai/Llama-3.1-Tulu-3-8B-SFT","allenai/Llama-3.1-Tulu-3-8B-DPO","allenai/Llama-3.1-Tulu-3-8B",]
model_delays = {}
for model in list_of_model_name:
    model_delays[model] = 0 # 4 second delay


def update_generation_hyperparameters(max_response_length, temperature, top_p, batch_size, big_model_mode=False):
    global MAX_RESPONSE_LENGTH, TEMPERATURE, TOP_P, BATCH_SIZE, BIG_MODEL_MODE
    MAX_RESPONSE_LENGTH = max_response_length
    TEMPERATURE = temperature
    TOP_P = top_p
    BATCH_SIZE = batch_size
    BIG_MODEL_MODE = big_model_mode

update_generation_hyperparameters(50, 0.7, 0.9, 1) # changed batch size to 1

def preload_models_and_tokenizers(model_names):
    print("Preloading models")
    BIG_MODEL_MODE = False
    models = {}
    tokenizers = {}
    for model_name in model_names:
        if not BIG_MODEL_MODE:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=f"cuda:{gpu_id}", trust_remote_code=True)
        else:
            # ensure that gpu_id is a list
            if not isinstance(gpu_id, list):
                raise ValueError("In BIG_MODEL_MODE, gpu_id should be a list of GPU ids.")
            # set CUDA_VISIBLE_DEVICES
            gpu_id_str = ",".join([str(i) for i in gpu_id])
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        except:
            # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", use_fast=True)
            # tokenizer.pad_token = tokenizer.eos_token
            # tokenizer.padding_side = "left"
            raise ValueError("Tokenizer loading failed. Please check the model name. If it is a lora module, upload your tokenizer to the huggingface repo too.")

        models[model_name] = model
        tokenizers[model_name] = tokenizer

    print("Models loaded!")
        
    return models, tokenizers



def _build_ranking_prompt(
    question: str,
    candidate_answers: List[str],
    candidate_model_names: List[str],
) -> str:
    """
    Build a judging prompt for the ranker model.
    The model is asked to score each candidate answer from 1 to 10
    and respond with a JSON object: {"scores": [s1, s2, ..., sN]}.
    """
    prompt_lines = []
    prompt_lines.append(
        "You are an expert judge for assessing the quality of answers to a user's question."
    )
    prompt_lines.append(
        "Rate each answer on a scale from 1 to 10, where 10 is an excellent, fully correct answer and 1 is a very poor answer."
    )
    prompt_lines.append("")
    prompt_lines.append(f"Question: {question}")
    prompt_lines.append("")
    prompt_lines.append("Candidate answers:")
    for idx, answer in enumerate(candidate_answers):
        header = f"Answer {idx + 1}"
        if candidate_model_names is not None and idx < len(candidate_model_names):
            header += f" (from model {candidate_model_names[idx]}):"
        else:
            header += ":"
        prompt_lines.append(header)
        prompt_lines.append(answer)
        prompt_lines.append("")
    prompt_lines.append(
        "Now provide your judgment as a JSON object exactly in the following format (and nothing else):"
    )
    prompt_lines.append('{"scores": [s1, s2, ..., sN]}')
    prompt_lines.append(
        "Replace s1, s2, ..., sN with numeric scores for Answer 1, Answer 2, etc."
    )
    return "\n".join(prompt_lines)


def _build_pairwise_ranking_prompt(
    question: str,
    answer_a: str,
    answer_b: str,
) -> str:
    """
    Build a pairwise judging prompt for the ranker model.
    The model is asked to choose whether Answer A or Answer B is better
    and reply with a single character: "A" or "B".
    """
    prompt = (
        "You are a judge comparing two answers to a user's question.\n\n"
        f"Question:\n{question}\n\n"
        "Candidate A:\n"
        f"{answer_a}\n\n"
        "Candidate B:\n"
        f"{answer_b}\n\n"
        'Which answer is better, A or B? Reply with a single character: "A" or "B".'
    )
    return prompt


def _collect_dev_candidates_and_scores(task, task_type, gpu_ids, model_names):
    """
    Generate dev-set candidates and task scores for each model.
    Returns:
        dev_inputs: List[str]
        dev_candidates: List[List[str]]  # per example: list over models
        dev_scores: List[List[float]]    # per example: list over models
    """
    dev_inputs = eval.prepare_inputs(task, task_type, "dev")
    if not dev_inputs:
        return [], [], []

    list_of_input_list = [dev_inputs for _ in model_names]
    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids,
    )

    num_models = len(model_names)
    num_examples = len(dev_inputs)

    dev_scores_per_model = []
    for i in range(num_models):
        dev_outputs_i = list_of_output_list[i]
        scores_i = eval.get_scores(task, task_type, "dev", dev_outputs_i)
        dev_scores_per_model.append(scores_i)

    dev_candidates = []
    dev_scores = []
    for j in range(num_examples):
        cand_j = []
        score_j = []
        for i in range(num_models):
            cand_j.append(list_of_output_list[i][j])
            score_j.append(dev_scores_per_model[i][j])
        dev_candidates.append(cand_j)
        dev_scores.append(score_j)

    return dev_inputs, dev_candidates, dev_scores


def _train_ranker_on_dev(
    task,
    task_type,
    gpu_ids,
    model_names,
    hyperparameters,
    dev_inputs,
    dev_candidates,
    dev_scores,
):
    """
    Train a causal LM ranker on dev using pairwise preferences.
    Returns the path to the trained checkpoint, or None.
    """
    if not hyperparameters.get("train_ranker_on_dev", False):
        return None

    if not dev_inputs:
        print("[LLM-Blender] No dev inputs found; skip ranker training.")
        return None

    os.makedirs(METHOD_LOG_DIR, exist_ok=True)
    sft_path = os.path.join(
        METHOD_LOG_DIR,
        f"ranker_sft_{task}_{len(model_names)}.jsonl",
    )

    with open(sft_path, "w") as f:
        for q, cand_list, score_list in zip(dev_inputs, dev_candidates, dev_scores):
            n = len(cand_list)
            for i in range(n):
                for j in range(i + 1, n):
                    si, sj = score_list[i], score_list[j]
                    if si == sj:
                        continue
                    better = "A" if si > sj else "B"
                    prompt = _build_pairwise_ranking_prompt(
                        question=q,
                        answer_a=cand_list[i],
                        answer_b=cand_list[j],
                    )
                    record = {"prompt": prompt, "completion": better}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    ranker_base = hyperparameters.get(
        "ranker_model_name", "Qwen/Qwen2.5-7B-Instruct"
    )
    ranker_gpu_id = hyperparameters.get("ranker_gpu_id", gpu_ids[0])
    batch_size = hyperparameters.get("ranker_sft_batch_size", 1)
    grad_acc = hyperparameters.get(
        "ranker_sft_gradient_accumulation_steps", 16
    )
    lr = hyperparameters.get("ranker_sft_learning_rate", 1e-5)
    epoch = hyperparameters.get("ranker_sft_epoch", 3)

    output_dir = os.path.join(
        METHOD_LOG_DIR,
        f"ranker_model_{task}_{len(model_names)}",
    )

    # The previous auto-detection logic (checking if adapter_config.json exists) is removed
    # to allow explicit retraining when train_ranker_on_dev is True.
    # If the user wants to use an existing trained model without retraining,
    # they should set train_ranker_on_dev=False and point ranker_model_name to the directory.

    # Run SFT in a separate process to avoid polluting the main process environment (CUDA_VISIBLE_DEVICES)
    # and to avoid initializing CUDA in the main process before multiprocessing.Pool is used later.
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(
        target=distributed_sft.single_sft,
        kwargs={
            "model_name": ranker_base,
            "sft_data_path": sft_path,
            "gpu_id": ranker_gpu_id,
            "output_model_path": output_dir,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_acc,
            "learning_rate": lr,
            "epoch": epoch,
        }
    )
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"Ranker SFT process failed with exit code {p.exitcode}")

    print("[LLM-Blender] Ranker trained on dev, saved to:", output_dir)
    return output_dir


def _train_fuser_on_dev(
    task,
    task_type,
    gpu_ids,
    model_names,
    hyperparameters,
    dev_inputs,
    dev_candidates,
    dev_scores,
):
    """
    Train a causal LM fuser on dev.
    For each dev example, we take the top-k candidates (by dev score)
    as input and the gold answer from the dev set as the supervised target
    (when available); otherwise we fall back to the best candidate.
    """
    if not hyperparameters.get("train_fuser_on_dev", False):
        return None

    if not dev_inputs:
        print("[LLM-Blender] No dev inputs found; skip fuser training.")
        return None

    # Try to load gold dev outputs to use as supervision targets.
    dev_gold_outputs = None
    try:
        dataset_path = os.path.join(eval.DATA_DIR, f"{task}.json")
        with open(dataset_path, "r") as f:
            full_data = json.load(f)
        dev_data = full_data.get("dev", [])
        dev_gold_outputs = []
        if task_type == "multiple_choice":
            for item in dev_data:
                answer_letter = item.get("answer")
                choices = item.get("choices", {})
                if answer_letter in choices:
                    dev_gold_outputs.append(choices[answer_letter])
                else:
                    dev_gold_outputs.append(None)
        elif task_type == "exact_match":
            for item in dev_data:
                dev_gold_outputs.append(item.get("output"))
        elif task_type == "f1_match":
            for item in dev_data:
                raw_out = item.get("output")
                gold_text = None
                if isinstance(raw_out, str):
                    try:
                        parsed = json.loads(raw_out)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            gold_text = parsed[0]
                        else:
                            gold_text = raw_out
                    except Exception:
                        gold_text = raw_out
                else:
                    gold_text = raw_out
                dev_gold_outputs.append(gold_text)
        elif task_type == "reward_model" or task_type == "text_generation":
            for item in dev_data:
                dev_gold_outputs.append(item.get("output"))
        else:
            dev_gold_outputs = None

        if dev_gold_outputs is not None and len(dev_gold_outputs) < len(dev_inputs):
            dev_gold_outputs = dev_gold_outputs[: len(dev_inputs)]
    except Exception:
        dev_gold_outputs = None

    if dev_gold_outputs is None:
        print(
            "[LLM-Blender] No gold dev outputs available for fuser; "
            "falling back to best-candidate supervision."
        )

    os.makedirs(METHOD_LOG_DIR, exist_ok=True)
    sft_path = os.path.join(
        METHOD_LOG_DIR,
        f"fuser_sft_{task}_{len(model_names)}.jsonl",
    )

    top_k = hyperparameters.get("top_k", 3)
    with open(sft_path, "w") as f:
        for idx, (q, cand_list, score_list) in enumerate(
            zip(dev_inputs, dev_candidates, dev_scores)
        ):
            n = len(cand_list)
            if n == 0:
                continue
            sorted_idx = sorted(
                range(n), key=lambda idx: score_list[idx], reverse=True
            )
            k = min(top_k, n)
            top_indices = sorted_idx[:k]
            best_idx = sorted_idx[0]

            top_candidates = [(idx, cand_list[idx]) for idx in top_indices]
            prompt = _build_fusion_prompt(q, top_candidates, model_names)
            # Prefer gold dev answer when available; otherwise fall back to best candidate.
            completion = None
            if dev_gold_outputs is not None and idx < len(dev_gold_outputs):
                completion = dev_gold_outputs[idx]
            if not completion:
                completion = cand_list[best_idx]

            record = {"prompt": prompt, "completion": completion}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    fuser_base = hyperparameters.get("fuser_model_name") or (
        model_names[0] if len(model_names) > 0 else None
    )
    if fuser_base is None:
        raise ValueError(
            "fuser_model_name must be specified when training fuser."
        )

    fuser_gpu_id = hyperparameters.get("fuser_gpu_id", gpu_ids[0])
    batch_size = hyperparameters.get("fuser_sft_batch_size", 1)
    grad_acc = hyperparameters.get(
        "fuser_sft_gradient_accumulation_steps", 16
    )
    lr = hyperparameters.get("fuser_sft_learning_rate", 1e-5)
    epoch = hyperparameters.get("fuser_sft_epoch", 3)

    output_dir = os.path.join(
        METHOD_LOG_DIR,
        f"fuser_model_{task}_{len(model_names)}",
    )

    # The previous auto-detection logic (checking if adapter_config.json exists) is removed
    # to allow explicit retraining when train_fuser_on_dev is True.
    # If the user wants to use an existing trained model without retraining,
    # they should set train_fuser_on_dev=False and point fuser_model_name to the directory.

    # Run SFT in a separate process to avoid polluting the main process environment (CUDA_VISIBLE_DEVICES)
    # and to avoid initializing CUDA in the main process before multiprocessing.Pool is used later.
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(
        target=distributed_sft.single_sft,
        kwargs={
            "model_name": fuser_base,
            "sft_data_path": sft_path,
            "gpu_id": fuser_gpu_id,
            "output_model_path": output_dir,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_acc,
            "learning_rate": lr,
            "epoch": epoch,
        }
    )
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"Fuser SFT process failed with exit code {p.exitcode}")

    print("[LLM-Blender] Fuser trained on dev, saved to:", output_dir)
    return output_dir


def _parse_pairwise_preference(output_text: str) -> str:
    """
    Parse the ranker's pairwise preference output.
    Returns "A", "B", or "" if no clear preference is found.
    """
    text = output_text.strip().upper()
    for ch in text:
        if ch == "A":
            return "A"
        if ch == "B":
            return "B"
    return ""


def _parse_scores_from_ranker_output(
    output_text: str,
    num_candidates: int,
) -> List[float]:
    """
    Parse a list of scores from the ranker model output.
    Primary path: parse JSON with a `scores` field.
    Fallback: extract numbers via regex and truncate/pad.
    """
    # Try to extract JSON substring if extra text is present.
    json_candidate = output_text.strip()
    start_idx = json_candidate.find("{")
    end_idx = json_candidate.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_candidate = json_candidate[start_idx : end_idx + 1]

    # First attempt: JSON parsing.
    try:
        data = json.loads(json_candidate)
        scores_raw = (
            data.get("scores")
            or data.get("Scores")
            or data.get("score")
            or data.get("Score")
        )
        if isinstance(scores_raw, list):
            scores: List[float] = []
            for value in scores_raw[:num_candidates]:
                try:
                    scores.append(float(value))
                except Exception:
                    scores.append(0.0)
            if len(scores) < num_candidates:
                scores.extend([0.0] * (num_candidates - len(scores)))
            return scores
    except Exception:
        pass

    # Fallback: regex-based number extraction.
    number_strings = re.findall(r"-?\d+\.?\d*", output_text)
    scores: List[float] = []
    for value in number_strings[:num_candidates]:
        try:
            scores.append(float(value))
        except Exception:
            scores.append(0.0)
    if len(scores) < num_candidates:
        scores.extend([0.0] * (num_candidates - len(scores)))
    return scores


def _build_fusion_prompt(
    question: str,
    top_candidates: List[Tuple[int, str]],
    model_names: List[str],
) -> str:
    """
    Build a fusion prompt for the fuser model.
    top_candidates: list of (model_index, answer_text) pairs in descending rank order.
    """
    prompt_lines = []
    prompt_lines.append(
        "You are an AI assistant that combines multiple candidate answers into a single, high-quality answer."
    )
    prompt_lines.append(
        "Use the good parts of each candidate answer, ignore mistakes, and produce a final answer that is as accurate, helpful, and concise as possible."
    )
    prompt_lines.append("")
    prompt_lines.append(f"Question: {question}")
    prompt_lines.append("")
    prompt_lines.append("Candidate answers (from best to worse according to a judge model):")
    for rank, (model_idx, answer) in enumerate(top_candidates):
        model_name = model_names[model_idx] if 0 <= model_idx < len(model_names) else f"Model_{model_idx}"
        prompt_lines.append(f"Candidate {rank + 1} (from model {model_name}):")
        prompt_lines.append(answer)
        prompt_lines.append("")
    prompt_lines.append(
        "Now write the final answer to the question based on these candidates."
    )
    prompt_lines.append(
        "Final answer:"
    )
    return "\n".join(prompt_lines)


def _simple_rank_and_fuse(
    task: str,
    task_type: str,
    gpu_ids,
    model_names,
    hyperparameters,
    test_input_list,
    candidates_per_example: List[List[str]],
    ranker_model_override: str = None,
    fuser_model_override: str = None,
):
    """
    Backend 1: use causal LMs as zero-shot ranker and fuser.
    """
    max_response_length = hyperparameters.get("max_response_length")
    top_k = hyperparameters.get("top_k", 3)
    top_k = max(1, min(top_k, len(model_names)))

    # Ranker: causal LM as judge
    # Prefer override (trained on dev), otherwise use configured base.
    base_ranker = hyperparameters.get(
        "ranker_model_name", "Qwen/Qwen2.5-7B-Instruct"
    )
    actual_ranker_model = ranker_model_override or base_ranker
    ranker_gpu_id = hyperparameters.get("ranker_gpu_id", gpu_ids[0])
    ranker_max_response_length = hyperparameters.get(
        "ranker_max_response_length", 128
    )

    # Fuser: causal LM as summarizer
    base_fuser = hyperparameters.get("fuser_model_name") or (
        model_names[0] if len(model_names) > 0 else base_ranker
    )
    actual_fuser_model = fuser_model_override or base_fuser
    fuser_gpu_id = hyperparameters.get("fuser_gpu_id", gpu_ids[0])
    fuser_max_response_length = hyperparameters.get(
        "fuser_max_response_length", max_response_length
    )

    print("[LLM-Blender] Using ranker model:", actual_ranker_model)
    print("[LLM-Blender] Using fuser model:", actual_fuser_model)

    num_examples = len(test_input_list)
    num_models = len(model_names)

    # Step 2: pairwise-ranking prompts and call ranker model
    print("[LLM-Blender] Ranking candidate answers with pairwise comparisons...")
    pairwise_prompts: List[str] = []
    pairwise_metadata: List[Tuple[int, int, int]] = []
    for example_idx in range(num_examples):
        question = test_input_list[example_idx]
        candidate_answers = candidates_per_example[example_idx]
        n = len(candidate_answers)
        for i in range(n):
            for j in range(i + 1, n):
                prompt = _build_pairwise_ranking_prompt(
                    question=question,
                    answer_a=candidate_answers[i],
                    answer_b=candidate_answers[j],
                )
                pairwise_prompts.append(prompt)
                pairwise_metadata.append((example_idx, i, j))

    all_scores: List[List[float]] = [
        [0.0 for _ in range(num_models)] for _ in range(num_examples)
    ]
    if pairwise_prompts:
        ranker_outputs_nested = distributed_generation.distributed_generation(
            [actual_ranker_model],
            [pairwise_prompts],
            [ranker_gpu_id],
            max_response_length=ranker_max_response_length,
        )
        ranker_outputs = ranker_outputs_nested[0]

        for output_text, (example_idx, i, j) in zip(
            ranker_outputs, pairwise_metadata
        ):
            pref = _parse_pairwise_preference(output_text)
            if pref == "A":
                all_scores[example_idx][i] += 1.0
            elif pref == "B":
                all_scores[example_idx][j] += 1.0
            else:
                all_scores[example_idx][i] += 0.5
                all_scores[example_idx][j] += 0.5

    all_top_indices: List[List[int]] = []
    for example_idx in range(num_examples):
        sorted_indices = sorted(
            list(range(num_models)),
            key=lambda idx: all_scores[example_idx][idx],
            reverse=True,
        )
        top_indices = sorted_indices[:top_k]
        all_top_indices.append(top_indices)

    # Step 3: build fusion prompts and call fuser model
    print("[LLM-Blender] Fusing top-k candidate answers with the fuser model...")
    fuser_input_list: List[str] = []
    for example_idx in range(num_examples):
        question = test_input_list[example_idx]
        candidates = candidates_per_example[example_idx]
        top_indices = all_top_indices[example_idx]
        top_candidates: List[Tuple[int, str]] = [
            (model_idx, candidates[model_idx]) for model_idx in top_indices
        ]
        fusion_prompt = _build_fusion_prompt(
            question=question,
            top_candidates=top_candidates,
            model_names=model_names,
        )
        fuser_input_list.append(fusion_prompt)

    fuser_outputs_nested = distributed_generation.distributed_generation(
        [actual_fuser_model],
        [fuser_input_list],
        [fuser_gpu_id],
        max_response_length=fuser_max_response_length,
    )
    final_outputs = fuser_outputs_nested[0]

    return final_outputs, all_scores, all_top_indices

# TODO: changed back to run_method without trailing underscore
def run_method_(task, task_type, gpu_ids, model_names, hyperparameters):
    """
    LLM-Blender style text-level method.

    For each input:
      1) All candidate models generate their own answers.
      2) Optionally train a causal LM ranker and/or fuser on the dev set.
      3) Use the (possibly trained) ranker to score candidates.
      4) Use the (possibly trained) fuser to fuse top-k candidates.
    """

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    os.makedirs("model_collaboration/logs", exist_ok=True)
    os.makedirs(METHOD_LOG_DIR, exist_ok=True)

    # Optional training on dev set
    trained_ranker_path = None
    trained_fuser_path = None
    if hyperparameters.get("train_ranker_on_dev", False) or hyperparameters.get(
        "train_fuser_on_dev", False
    ):
        print("[LLM-Blender] Collecting dev candidates and scores for training...")
        dev_inputs, dev_candidates, dev_scores = _collect_dev_candidates_and_scores(
            task, task_type, gpu_ids, model_names
        )
        if hyperparameters.get("train_ranker_on_dev", False):
            trained_ranker_path = _train_ranker_on_dev(
                task,
                task_type,
                gpu_ids,
                model_names,
                hyperparameters,
                dev_inputs,
                dev_candidates,
                dev_scores,
            )
        if hyperparameters.get("train_fuser_on_dev", False):
            trained_fuser_path = _train_fuser_on_dev(
                task,
                task_type,
                gpu_ids,
                model_names,
                hyperparameters,
                dev_inputs,
                dev_candidates,
                dev_scores,
            )

    # Prepare test inputs
    test_input_list = eval.prepare_inputs(task, task_type, "test")

    # Step 1: generate candidate answers from each base model
    print(f"[LLM-Blender] Generating candidate answers from {len(model_names)} base models: {model_names}")
    list_of_input_list = [test_input_list for _ in model_names]

    # delay added to distributed generation
    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids,
    )

    num_examples = len(test_input_list)
    num_models = len(model_names)

    # Reorganize candidates per example for convenience
    candidates_per_example: List[List[str]] = []
    for example_idx in range(num_examples):
        example_candidates = []
        for model_idx in range(num_models):
            example_candidates.append(list_of_output_list[model_idx][example_idx])
        candidates_per_example.append(example_candidates)

    # Ranking and fusion
    final_outputs, all_scores, all_top_indices = _simple_rank_and_fuse(
        task,
        task_type,
        gpu_ids,
        model_names,
        hyperparameters,
        test_input_list,
        candidates_per_example,
        ranker_model_override=trained_ranker_path,
        fuser_model_override=trained_fuser_path,
    )

    # Evaluation
    print("[LLM-Blender] Evaluating fused outputs on the test set...")
    test_scores = eval.get_scores(task, task_type, "test", final_outputs)
    avg_test_score = sum(test_scores) / len(test_scores) if test_scores else 0.0
    print(
        f"[LLM-Blender] Final test {task} score with {len(model_names)} models: {avg_test_score}"
    )

    # Save logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "method": "text_llm_blender",
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_score,
        "logs": [],
    }
    for example_idx in range(num_examples):
        log_entry = {
            "input": test_input_list[example_idx],
            "candidate_answers": candidates_per_example[example_idx],
            "ranker_scores": all_scores[example_idx],
            "top_k_model_indices": all_top_indices[example_idx],
            "output": final_outputs[example_idx],
            "score": test_scores[example_idx],
        }
        experiment_logs["logs"].append(log_entry)

    os.makedirs("logs", exist_ok=True)
    log_filename = "model_collaboration/logs/{}_{}_{}_llm_blender.json".format(
        task, len(model_names), round(avg_test_score, 4)
    )
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0


if __name__ == "__main__":
    # This module is intended to be used via model_collaboration/main.py
    # with configuration-driven arguments.
    raise SystemExit(
        "Run this method through main.py with a JSON config, not as a standalone script."
    )

# DEPRECATED
def run_method______(task, task_type, gpu_ids, model_names, hyperparameters):
    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    os.makedirs("model_collaboration/logs", exist_ok=True)
    os.makedirs(METHOD_LOG_DIR, exist_ok=True)

    # Optional training on dev set
    trained_ranker_path = None
    trained_fuser_path = None
    if hyperparameters.get("train_ranker_on_dev", False) or hyperparameters.get(
        "train_fuser_on_dev", False
    ):
        print("[LLM-Blender] Collecting dev candidates and scores for training...")
        dev_inputs, dev_candidates, dev_scores = _collect_dev_candidates_and_scores(
            task, task_type, gpu_ids, model_names
        )
        if hyperparameters.get("train_ranker_on_dev", False):
            trained_ranker_path = _train_ranker_on_dev(
                task,
                task_type,
                gpu_ids,
                model_names,
                hyperparameters,
                dev_inputs,
                dev_candidates,
                dev_scores,
            )
        if hyperparameters.get("train_fuser_on_dev", False):
            trained_fuser_path = _train_fuser_on_dev(
                task,
                task_type,
                gpu_ids,
                model_names,
                hyperparameters,
                dev_inputs,
                dev_candidates,
                dev_scores,
            )
 

    test_scores = []
    # Prepare test inputs (one by one)
    for input_list in eval.prepare_inputs(task, task_type, "test"):
        test_input_list = [input_list] 


        list_of_input_list = [test_input_list for _ in model_names] 

        # delay added to distributed generation based on the model...
        list_of_output_list = distributed_generation.distributed_generation(
            model_names,
            list_of_input_list,
            gpu_ids,
        )

        num_examples = len(test_input_list)
        num_models = len(model_names)

        # Reorganize candidates per example for convenience
        candidates_per_example: List[List[str]] = []
        for example_idx in range(num_examples):
            example_candidates = []
            for model_idx in range(num_models):
                example_candidates.append(list_of_output_list[model_idx][example_idx])
            candidates_per_example.append(example_candidates)

        # Ranking and fusion
        final_outputs, all_scores, all_top_indices = _simple_rank_and_fuse(
            task,
            task_type,
            gpu_ids,
            model_names,
            hyperparameters,
            test_input_list,
            candidates_per_example,
            ranker_model_override=trained_ranker_path,
            fuser_model_override=trained_fuser_path,
        )

        test_scores.extend(eval.get_scores(task, task_type, "test", final_outputs))
        

    # Evaluation
    print("[LLM-Blender] Evaluating fused outputs on the test set...")
    avg_test_score = sum(test_scores) / len(test_scores) if test_scores else 0.0
    print(
        f"[LLM-Blender] Final test {task} score with {len(model_names)} models: {avg_test_score}"
    )

    # Save logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "method": "text_llm_blender",
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_score,
        "logs": [],
    }
    for example_idx in range(num_examples):
        log_entry = {
            "input": test_input_list[example_idx],
            "candidate_answers": candidates_per_example[example_idx],
            "ranker_scores": all_scores[example_idx],
            "top_k_model_indices": all_top_indices[example_idx],
            "output": final_outputs[example_idx],
            "score": test_scores[example_idx],
        }
        experiment_logs["logs"].append(log_entry)

    os.makedirs("logs", exist_ok=True)
    log_filename = "model_collaboration/logs/{}_{}_{}_llm_blender.json".format(
        task, len(model_names), round(avg_test_score, 4)
    )
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0

# run method without distributed generation and preloading models
def run_method(task, task_type, gpu_ids, model_names, hyperparameters):
    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    os.makedirs("model_collaboration/logs", exist_ok=True)
    os.makedirs(METHOD_LOG_DIR, exist_ok=True)

    # Optional training on dev set


    trained_ranker_path = None
    trained_fuser_path = None
    if hyperparameters.get("train_ranker_on_dev", False) or hyperparameters.get(
        "train_fuser_on_dev", False
    ):
        print("[LLM-Blender] Collecting dev candidates and scores for training...")
        dev_inputs, dev_candidates, dev_scores = _collect_dev_candidates_and_scores(
            task, task_type, gpu_ids, model_names
        )
        if hyperparameters.get("train_ranker_on_dev", False):
            trained_ranker_path = _train_ranker_on_dev(
                task,
                task_type,
                gpu_ids,
                model_names,
                hyperparameters,
                dev_inputs,
                dev_candidates,
                dev_scores,
            )
        if hyperparameters.get("train_fuser_on_dev", False):
            trained_fuser_path = _train_fuser_on_dev(
                task,
                task_type,
                gpu_ids,
                model_names,
                hyperparameters,
                dev_inputs,
                dev_candidates,
                dev_scores,
            )
        
    # initialize ranking and fusing params
    max_response_length = hyperparameters.get("max_response_length")
    top_k = hyperparameters.get("top_k", 3)
    top_k = max(1, min(top_k, len(model_names)))

    # Ranker: causal LM as judge
    # Prefer override (trained on dev), otherwise use configured base.
    base_ranker = hyperparameters.get(
        "ranker_model_name", "Qwen/Qwen2.5-7B-Instruct"
    )

    ranker_model_override=trained_ranker_path
    fuser_model_override=trained_fuser_path
    actual_ranker_model = ranker_model_override or base_ranker
    ranker_gpu_id = hyperparameters.get("ranker_gpu_id", gpu_ids[0])
    ranker_max_response_length = hyperparameters.get(
        "ranker_max_response_length", 128
    )

    # Fuser: causal LM as summarizer
    base_fuser = hyperparameters.get("fuser_model_name") or (
        model_names[0] if len(model_names) > 0 else base_ranker
    )
    actual_fuser_model = fuser_model_override or base_fuser
    fuser_gpu_id = hyperparameters.get("fuser_gpu_id", gpu_ids[0])
    fuser_max_response_length = hyperparameters.get(
        "fuser_max_response_length", max_response_length
    )

    


    prepared_inputs = eval.prepare_inputs(task, task_type, "test")

    num_examples = len(prepared_inputs) # number of models in pool times questions
    num_models = len(model_names)

    # only add locally!!
    models, tokenizers = preload_models_and_tokenizers(model_names + [actual_fuser_model] + [actual_ranker_model])

    final_outputs = []
    candidates_per_example: List[List[str]] = []

    all_scores: List[List[float]] = [
        [0.0 for _ in range(num_models)] for _ in range(num_examples)
    ]
    
    for example_idx in range(len(prepared_inputs)):
        input_list = prepared_inputs[example_idx]


        # list_of_input_list = [input_list for _ in model_names] 

        
        # # get output from pool of LLMs
        # list_of_output_list = distributed_generation.distributed_generation(
        #     model_names,
        #     list_of_input_list,
        #     gpu_ids,
        # ) # replaced by code below!

        test_input_list = []
        for model_name in models:
            test_input_list.append(generate_text_preload_llms(
                model_name,
                models,
                tokenizers,
                gpu_id,
                input_list,
                MAX_RESPONSE_LENGTH if max_response_length is None else max_response_length,
                TEMPERATURE,
                TOP_P
            ))

        # adapted to go one model at a time


        # make ranker prompt
        question = input_list
        candidate_answers = test_input_list
        n = len(candidate_answers)

        prompt = None
        metadata = None
        for i in range(n):
            for j in range(i + 1, n):
                prompt = _build_pairwise_ranking_prompt(
                    question=question,
                    answer_a=candidate_answers[i],
                    answer_b=candidate_answers[j],
                )
                metadata = (example_idx, i, j)
                pairwise_prompts.append(prompt)
                pairwise_metadata.append((example_idx, i, j))

        # ranker
        if prompt:
            # TODO: call ranker HERE
            ranker_output_nested = generate_text_preload_llms(actual_ranker_model, models, tokenizers, gpu_id, prompt, 
                MAX_RESPONSE_LENGTH if max_response_length is None else max_response_length,TEMPERATURE, TOP_P)


            ranker_output = ranker_output_nested[0]

            
            example_idx, i, j = metadata
            pref = _parse_pairwise_preference(ranker_output)
            import numpy as np
            print(np.array(all_scores).shape)
            if pref == "A":
                all_scores[example_idx][i] += 1.0
            elif pref == "B":
                all_scores[example_idx][j] += 1.0
            else:
                all_scores[example_idx][i] += 0.5
                all_scores[example_idx][j] += 0.5

        
        # this is wrong... should be only the one indice that we are interested in
        sorted_indices_index = sorted(
            list(range(num_models)),
            key=lambda idx: all_scores[example_idx][idx],
            reverse=True,
        )
        top_indices = sorted_indices_index[:top_k]
        print(top_indices)

        fuser_input_list: List[str] = []
    
        question = input_list
        

        candidates = candidates_per_example[example_idx]
        top_candidates: List[Tuple[int, str]] = [
            (model_idx, candidates[model_idx]) for model_idx in top_indices
        ]
        print("QUESTION")
        print(question)
        print("TOP_CANIDATES")
        print(top_canidates)
        fusion_prompt = _build_fusion_prompt(
            question=question,
            top_candidates=top_candidates,
            model_names=model_names,
        )
        fuser_input = fusion_prompt

        # call fuser
        fuser_output_nested = generate_text_preload_llms(actual_fuser_model, models, tokenizers, gpu_id, fusion_input, 
                MAX_RESPONSE_LENGTH if max_response_length is None else max_response_length,TEMPERATURE, TOP_P)
        final_outputs.append(fuser_outputs_nested[0])


    print("[LLM-Blender] Evaluating fused outputs on the test set...")
    test_scores = eval.get_scores(task, task_type, "test", final_outputs)
    avg_test_score = sum(test_scores) / len(test_scores) if test_scores else 0.0
    print(
        f"[LLM-Blender] Final test {task} score with {len(model_names)} models: {avg_test_score}"
    )

    # Save logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "method": "text_llm_blender",
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_score,
        "logs": [],
    }
    for example_idx in range(num_examples):
        log_entry = {
            "input": test_input_list[example_idx],
            "candidate_answers": candidates_per_example[example_idx],
            "ranker_scores": all_scores[example_idx],
            "top_k_model_indices": all_top_indices[example_idx],
            "output": final_outputs[example_idx],
            "score": test_scores[example_idx],
        }
        experiment_logs["logs"].append(log_entry)

    os.makedirs("logs", exist_ok=True)
    log_filename = "model_collaboration/logs/{}_{}_{}_llm_blender.json".format(
        task, len(model_names), round(avg_test_score, 4)
    )
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0


def generate_text_preload_llms(model_name,
            models,
            tokenizers,
            gpu_id,
            input_list,
            max_response_length, # if max_response_length is None else max_response_length,
            TEMPERATURE,
            TOP_P):
    # if not BIG_MODEL_MODE:
    #     model = models[model_name]
    # else:
    #     # ensure that gpu_id is a list
    #     if not isinstance(gpu_id, list):
    #         raise ValueError("In BIG_MODEL_MODE, gpu_id should be a list of GPU ids.")
    #     # set CUDA_VISIBLE_DEVICES
    #     gpu_id_str = ",".join([str(i) for i in gpu_id])
    #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str
    #     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    # try:
    #     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.padding_side = "left"
    # except:
    #     # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", use_fast=True)
    #     # tokenizer.pad_token = tokenizer.eos_token
    #     # tokenizer.padding_side = "left"
    #     raise ValueError("Tokenizer loading failed. Please check the model name. If it is a lora module, upload your tokenizer to the huggingface repo too.")

    # TODO: this replaces the above code
    model = models[model_name]
    tokenizer = tokenizers[model_name]


    # try to apply chat template
    try:
        chat_input = None

        if "<begin>" in input_list:
            question, partial_response = input_list.split("<begin>", 1)
            chat = [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ]
            chat_input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            chat_input += partial_response
            chat_inputs.append(chat_input)
        else:
            chat = [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_list}
            ]
            chat_input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            
    except:
        chat_input = input_list
    
    print(type(chat_input))
    print(chat_input)
    inputs = tokenizer(chat_input, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        # TODO: where the LLM is actually called... should invoke a delay here dependent of the LLM name
        # Should set the batch size to 1 in this case I think
        if DELAY:
            time.sleep(model_delays[model_name])
            print("delayed with model " + str(model_name))


        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_RESPONSE_LENGTH,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded_outputs = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    # decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # thinking model compatibility
    for idx in range(len(decoded_outputs)):
        if "</think>" in decoded_outputs[idx]:
            decoded_outputs[idx] = decoded_outputs[idx].split("</think>")[-1].strip()

    output_list = decoded_outputs
    return output_list

def simple_rank_and_fuse_preload(task: str,
    task_type: str,
    gpu_ids,
    model_names,
    models,
    tokenizers,
    hyperparameters,
    test_input_list,
    candidates_per_example: List[List[str]],
    ranker_model_override: str = None,
    fuser_model_override: str = None,):

    print("[LLM-Blender] Ranking candidate answers with pairwise comparisons...")
    pairwise_prompts: List[str] = []
    pairwise_metadata: List[Tuple[int, int, int]] = []
    for example_idx in range(num_examples):
        question = test_input_list[example_idx]
        candidate_answers = candidates_per_example[example_idx]
        n = len(candidate_answers)
        for i in range(n):
            for j in range(i + 1, n):
                prompt = _build_pairwise_ranking_prompt(
                    question=question,
                    answer_a=candidate_answers[i],
                    answer_b=candidate_answers[j],
                )
                pairwise_prompts.append(prompt)
                pairwise_metadata.append((example_idx, i, j))

    all_scores: List[List[float]] = [
        [0.0 for _ in range(num_models)] for _ in range(num_examples)
    ]
    if pairwise_prompts:
        ranker_outputs_nested = distributed_generation.distributed_generation(
            [actual_ranker_model],
            [pairwise_prompts],
            [ranker_gpu_id],
            max_response_length=ranker_max_response_length,
        )
        ranker_outputs = ranker_outputs_nested[0]

        for output_text, (example_idx, i, j) in zip(
            ranker_outputs, pairwise_metadata
        ):
            pref = _parse_pairwise_preference(output_text)
            if pref == "A":
                all_scores[example_idx][i] += 1.0
            elif pref == "B":
                all_scores[example_idx][j] += 1.0
            else:
                all_scores[example_idx][i] += 0.5
                all_scores[example_idx][j] += 0.5

    all_top_indices: List[List[int]] = []
    for example_idx in range(num_examples):
        sorted_indices = sorted(
            list(range(num_models)),
            key=lambda idx: all_scores[example_idx][idx],
            reverse=True,
        )
        top_indices = sorted_indices[:top_k]
        all_top_indices.append(top_indices)

    # Step 3: build fusion prompts and call fuser model
    print("[LLM-Blender] Fusing top-k candidate answers with the fuser model...")
    fuser_input_list: List[str] = []
    for example_idx in range(num_examples):
        question = test_input_list[example_idx]
        candidates = candidates_per_example[example_idx]
        top_indices = all_top_indices[example_idx]
        top_candidates: List[Tuple[int, str]] = [
            (model_idx, candidates[model_idx]) for model_idx in top_indices
        ]
        fusion_prompt = _build_fusion_prompt(
            question=question,
            top_candidates=top_candidates,
            model_names=model_names,
        )
        fuser_input_list.append(fusion_prompt)

    fuser_outputs_nested = distributed_generation.distributed_generation(
        [actual_fuser_model],
        [fuser_input_list],
        [fuser_gpu_id],
        max_response_length=fuser_max_response_length,
    )
    final_outputs = fuser_outputs_nested[0]

    return final_outputs, all_scores, all_top_indices