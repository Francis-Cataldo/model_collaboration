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
number_tested = 1
testing_input_nums = [1, 20, 50, 100, 200, 1000]
import sys
# if len(sys.argv) > 1:
#     number_tested = testing_input_nums[int(sys.argv[1])]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

update_generation_hyperparameters(4048, 0.7, 0.9, 1) # changed batch size to 1

def preload_models_and_tokenizers(model_names):
    # print(model_names)
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
            
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=f"cuda:{gpu_id}", trust_remote_code=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            # tokenizer.model_max_length = MAX_RESPONSE_LENGTH
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        except:
            # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", use_fast=True)
            # tokenizer.pad_token = tokenizer.eos_token
            # tokenizer.padding_side = "left"
            raise ValueError("Tokenizer loading failed. Please check the model name. If it is a lora module, upload your tokenizer to the huggingface repo too.")

        models[model_name] = model
        tokenizers[model_name] = tokenizer
        print(model_name, flush=True)

        # print(torch.cuda.memory_summary(), flush=True)
        # print(model.hf_device_map, flush=True)

    print("Models loaded!", flush=True)
        
    return models, tokenizers

def run_method(task, task_type, gpu_ids, pool_model_names, hyperparameters, num_datapoints=None):
    if num_datapoints != None:
        number_tested = testing_input_nums[num_datapoints]
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    os.makedirs("model_collaboration/logs", exist_ok=True)
    os.makedirs(METHOD_LOG_DIR, exist_ok=True)



    # get fuser and ranker names to preload
    ranker_model_override = None
    fuser_model_override = None

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
        pool_model_names[0] if len(pool_model_names) > 0 else base_ranker
    )
    actual_fuser_model = fuser_model_override or base_fuser
    fuser_gpu_id = hyperparameters.get("fuser_gpu_id", gpu_ids[0])
    fuser_max_response_length = hyperparameters.get(
        "fuser_max_response_length", MAX_RESPONSE_LENGTH
    )

    top_k = hyperparameters.get("top_k", 3)
    top_k = max(1, min(top_k, len(pool_model_names)))

    models, tokenizers = preload_models_and_tokenizers(pool_model_names + [actual_ranker_model])

    prepared_inputs = eval.prepare_inputs(task, task_type, "test")
    print(f"Length of prepared inputs is {len(prepared_inputs)}")
    import time
    start_time = time.time()
    # TODO: filter to amount desired
    outputs = []
    for i in range(1):
        print("test", flush=True)
        input_list = prepared_inputs[i]

        pool_reponses = get_pool_responses(input_list, pool_model_names, models, tokenizers)

        # for resp in pool_reponses:
        #     print(resp)

        # import time
        # time.sleep(1000)

        pairwise_ranking_output = pairwise_ranking(pool_reponses, input_list, models[actual_ranker_model], tokenizers[actual_ranker_model], actual_ranker_model)
        # print(f"Pairwise ranking output: {pairwise_ranking_output}")
        # print("Length of pairwise ranking output " +str(len(pairwise_ranking_output)))

        # fuser
        fuser_output = fuser(input_list, pairwise_ranking_output, pool_model_names, models[actual_fuser_model], tokenizers[actual_fuser_model], actual_fuser_model)
        outputs.append(fuser_output)

    test_scores = (eval.get_scores(task, task_type, "test", outputs))

    print("[LLM-Blender] Evaluating fused outputs on the test set...")
    avg_test_score = sum(test_scores) / len(test_scores) if test_scores else 0.0
    print(
        f"[LLM-Blender] Final test {task} score with {len(pool_model_names)} models: {avg_test_score}"
    )


    # import csv
    # with open(f'timing_graph_{len(pool_model_names)}.csv', 'a', newline='\n') as file:
    #     writer = csv.writer(file)
    #     # Write a single row (Headers)
    #     writer.writerow([number_tested, time.time()-start_time])


    # testing done here



def get_pool_responses(question, pool_model_names, models, tokenizers):
    
    responses = []
    for model_name in pool_model_names:
        tokenizer = tokenizers[model_name]
        model = models[model_name]
        try:
            chat_inputs = []
            
            if "<begin>" in question:
                question, partial_response = question.split("<begin>", 1)
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
                    {"role": "user", "content": question}
                ]
                chat_input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                chat_inputs.append(chat_input)
        except:
            chat_inputs = question
        
        inputs = tokenizer(chat_inputs, return_tensors="pt", padding=True, truncation=True,return_token_type_ids=False).to(model.device)
        #print("Should be the same as previous print: " + str(model_name))
        with torch.no_grad():
            # TODO: where the LLM is actually called... should invoke a delay here dependent of the LLM name
            # Should set the batch size to 1 in this case I think
            if DELAY:
                # time.sleep(model_delays[model_name])
                pass
                #print("delayed with model " + str(model_name))

            start_time_pool = time.time()
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_RESPONSE_LENGTH,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            # print("model name to added to csv: " + model_name)
            import csv
            with open('data.csv', 'a', newline='\n') as file:
                writer = csv.writer(file)
                # Write a single row (Headers)
                writer.writerow([model_name, time.time()-start_time_pool])
        # decoded_outputs = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        generated_tokens = output[:, inputs.input_ids.shape[1]:]
        decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # thinking model compatibility

        # if "</think>" in decoded_output:
        #     decoded_output = decoded_output.split("</think>")[-1].strip()

        responses.append(decoded_output[0])
        # del model
        # del tokenizer
        torch.cuda.empty_cache()
    
    return responses



def pairwise_ranking(candidate_answers, question, ranker_model, ranker_tokenizer, ranker_model_name):
    n = len(candidate_answers) # also number of models
    pairwise_prompts = []
    pairwise_metadata = []
    for i in range(n):
        for j in range(i + 1, n):
            prompt = _build_pairwise_ranking_prompt(
                question=question,
                answer_a=candidate_answers[i],
                answer_b=candidate_answers[j],
            )
            pairwise_metadata.append((i, j))
            pairwise_prompts.append(prompt)

    

    # print("PAIRWISE PROMPTS")
    # for prompt in pairwise_prompts:
    #     print(prompt)

    tokenizer = ranker_tokenizer
    tokenizer.padding_side = "right"
    model = ranker_model

   
    # print("PROMPT")
    # print("Type: " + str(type(pairwise_prompts)), flush=True)
    # print("Len: " + str(len(pairwise_prompts)), flush=True)
    # for prompt in pairwise_prompts:
    #     print(prompt)
    #     print()
    #     print()
    #     print()
    decoded_outputs = []
    for prompt in pairwise_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,return_token_type_ids=False).to(model.device)
        # TODO change with the following commented code
        # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        #print("Should be the same as previous print: " + str(model_name))
        with torch.no_grad():
            # TODO: where the LLM is actually called... should invoke a delay here dependent of the LLM name
            # Should set the batch size to 1 in this case I think
            if DELAY:
                # time.sleep(model_delays[model_name])
                pass
                #print("delayed with model " + str(model_name))

            start_time_ranker = time.time()
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_RESPONSE_LENGTH,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        import csv
        with open('data.csv', 'a', newline='\n') as file:
            writer = csv.writer(file)
            # Write a single row (Headers)
            writer.writerow([ranker_model_name, time.time()-start_time_ranker])
        # print("model name to added to csv: " + model_name)
        generated_tokens = output[0][inputs.input_ids.shape[1]:]
        decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # decoded_outputs.append(tokenizer.decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True))

        # just testing
        # print(f"Raw output from ranker including prompt: {tokenizer.batch_decode(output, skip_special_tokens=True)}\n END")

        decoded_outputs.append(decoded_output)

        # print(decoded_output)

    generated_texts = []

    # for i in range(output.shape[0]):
    #     input_len = inputs["attention_mask"][i].sum().item()

    #     generated = output[i, input_len:]

    #     text = tokenizer.decode(
    #         generated,
    #         skip_special_tokens=True
    #     )

    #     generated_texts.append(text)
    
    # generated_tokens = output[:, inputs.input_ids.shape[1]:]

    # print(len(generated_tokens))

    # decoded_output = tokenizer.batch_decode(
    #     generated_tokens,
    #     skip_special_tokens=True,
    #     max_new_tokens=1,
    # )
    print("decoded output")
    print(decoded_outputs)
    print(len(decoded_outputs))
    # print(decoded_output[1])
    
    all_scores = [0] * len(candidate_answers)

    for output_text, (i, j) in zip(
        decoded_outputs, pairwise_metadata
    ):
        pref = _parse_pairwise_preference(output_text)
        print("PREF")
        print(pref)
        if pref == "A":
            all_scores[i] += 1.0
        elif pref == "B":
            all_scores[j] += 1.0
        else:
            all_scores[i] += 0.5
            all_scores[j] += 0.5

    top_indices: List[int] = []
    sorted_indices = sorted(
        list(range(len(candidate_answers))),
        key=lambda idx: all_scores[idx],
        reverse=True,
    )
    top_k = 2 #TODO CHANGE
    print(all_scores)
    top_indices = sorted_indices[:top_k]

    return [candidate_answers[i] for i in top_indices]


    

def _parse_pairwise_preference(output_text: str) -> str:
    """
    Parse the ranker's pairwise preference output.
    Returns "A", "B", or "" if no clear preference is found.
    """
    text = output_text.strip().upper()
    print(text)
    for ch in text:
        if ch == "A":
            return "A"
        if ch == "B":
            return "B"
    return ""

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

def fuser(input_list, top_candidates, model_names, ranker_model, ranker_tokenizer, fuser_model_name):
    fusion_prompt_text = _build_fusion_prompt(
        question=input_list,
        top_candidates=top_candidates,
        model_names=model_names,
    )

    # print(fusion_prompt)

    tokenizer = ranker_tokenizer
    model = ranker_model
    print(fusion_prompt_text)

    chat = [
        {"role": "user", "content": fusion_prompt_text}
    ]
    formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True,return_token_type_ids=False).to(model.device)
    #print("Should be the same as previous print: " + str(model_name))
    with torch.no_grad():
        # TODO: where the LLM is actually called... should invoke a delay here dependent of the LLM name
        # Should set the batch size to 1 in this case I think
        if DELAY:
            # time.sleep(model_delays[model_name])
            pass
            #print("delayed with model " + str(model_name))

        start_time_fuser = time.time()
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_RESPONSE_LENGTH,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        import csv
        with open('data.csv', 'a', newline='\n') as file:
            writer = csv.writer(file)
            # Write a single row (Headers)
            writer.writerow([fuser_model_name, time.time()-start_time_fuser])
        # print("model name to added to csv: " + model_name)
        # import csv
        # with open('data.csv', 'a', newline='\n') as file:
        #     writer = csv.writer(file)
        #     # Write a single row (Headers)
        #     writer.writerow([model_name, time.time()-start_time])
    # decoded_outputs = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    generated_tokens = output[0][inputs.input_ids.shape[1]:]
    decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return decoded_output

    
        




    



def _build_fusion_prompt(
    question: str,
    top_candidates: List[str],
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
    for model_idx, answer in enumerate(top_candidates):
        # model_name = model_names[model_idx] if 0 <= model_idx < len(model_names) else f"Model_{model_idx}"
        prompt_lines.append(f"Candidate {model_idx + 1}):") # TODO model_idx -> rank if using ranker
        prompt_lines.append(answer)
        prompt_lines.append("")
    prompt_lines.append(
        "Now write the final answer to the question based on these candidates."
    )
    prompt_lines.append(
        "Final answer:"
    )
    return "\n".join(prompt_lines)