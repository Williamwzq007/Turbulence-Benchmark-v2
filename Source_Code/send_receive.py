import os
import time
import signal
import random
from run_llm import CallLLM
from termcolor import cprint
from auxiliary_functions import Auxiliary


HAS_SIGALRM = hasattr(signal, "SIGALRM") and (getattr(signal, "SIGALRM", None) is not None)
HAS_ALARM = hasattr(signal, "alarm")

def send_prompt_recieve_response(path, q, r, model_params, seed, num):
    auxiliary = Auxiliary()
    llm = CallLLM(model_params)
    model_name = model_params["model"]
    if not model_name:
        print("Please specify the model correctly in the config file.")
        exit()
    number_of_parameters = r.get("number_of_parameters")
    shuffle = True if r.get("shuffle").lower() == "true" else False

    openai_api = ["gpt-3.5-turbo", "gpt-4", "gpt-4o-2024-08-06", "deepseek-coder", "databricks/dbrx-instruct", "Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen2.5-Coder-32B-Instruct", "cognitivecomputations/dolphin-mixtral-8x22b", "google/gemma-2-27b-it"]
    cohere = ["command", "command-r-plus-08-2024"]
    claude = ["claude-3-5-sonnet-20240620", "claude-3-5-haiku-20241022"]
    google = [
    "models/gemini-pro-latest",
    "models/gemini-flash-latest",
    "models/gemini-flash-lite-latest",
    "models/gemini-2.0-flash",
    "models/gemini-2.5-pro",
    "models/gemini-2.5-flash",
    "codegemma-7b-it",
]
    mistral = ["codestral-latest", "mistral-large-latest"]
    azure_models = ["Meta-Llama-3-1-405B-Instruct", "Meta-Llama-3-1-70B-Instruct-ofoo", "Phi-3-medium-4k-instruct"]
    huggingface = ["starcoder2-15b-instruct-v0.1", "qwen2.5-coder-7b-instruct"]

    model = auxiliary.rename_model(model_name)

    import shutil

    try:
        columns = shutil.get_terminal_size(fallback=(120, 30)).columns
    except Exception:
        columns = 120


    auxiliary.create_model_folder(path + f"/Q{q}", model, num)
    params_list = auxiliary.get_params(q, seed)
    if shuffle:
        if seed != "default":
            random.seed(seed)
        random.shuffle(params_list)
    auxiliary.write_param_file(q, model, num, params_list)

    c = 1
    params_list = params_list[:number_of_parameters]
    inference_latency = []
    for params in params_list:
        auxiliary.create_result_folder(path + f"/Q{q}", c, model, num)
        if type(params) != tuple:
            params = (params,)

        question, function_name = auxiliary.parameterize_question(q, params)
        auxiliary.write_info_file(q, model, c, params, function_name, num)

        remainder = c % 10
        if remainder == 1:
            suffix = "st"
        elif remainder == 2:
            suffix = "nd"
        elif remainder == 3:
            suffix = "rd"
        else:
            suffix = "th"

        if q < 10:
            takes_space = 10
        elif 10 <= q < 100:
            takes_space = 11
        else:
            takes_space = 12

        print()
        cprint(
            f"Question {q}" + (" " * ((int(columns) // 2) - takes_space)),
            "black",
            "on_white",
            attrs=["bold"],
        )
        cprint(
            f"{c}{suffix} parameters" + (" " * ((int(columns) // 3) - takes_space)),
            "black",
            "on_light_grey",
            attrs=["bold"],
        )
        print(f"Waiting for {model_name} to respond...")
        sleep_duration = 1
        timeout = 150
        response, prompt = "", ""
        
        while True:
            try:
                start_time_real = time.time()

                if model_name in openai_api:
                    llm.set_prompt(question, True)
                    response, prompt = llm.openai_api()


                elif model_name in google:
                    if model_name == "codegemma-7b-it":
                        llm.set_prompt(question, False)
                        start_time_real = time.time()
                        response, prompt = llm.vertex_ai()
                    else:
                        llm.set_prompt(question, True)
                        start_time_real = time.time()
                        response, prompt = llm.gemini()


                elif model_name in azure_models:
                    llm.set_prompt(question, False)
                    response, prompt = llm.azure(model)

                elif model_name in huggingface:
                    llm.set_prompt(question, True)
                    response, prompt = llm.huggingface()

                else:
                    raise ValueError(f"Unknown model name: {model_name}")

                inference_latency.append(time.time() - start_time_real)
                break

            except Exception as e:
                cprint(e, "light_red")
                cprint(
                    f"{model_name} model did not respond. Retrying in {sleep_duration}(s)...",
                    "light_red",
                )
                time.sleep(sleep_duration)
                sleep_duration *= 2

            # finally:
            #     signal.alarm(0)

        cprint(f"The {model_name} response was received.", "light_green")
        cprint(
            f"Inference latency: {round(inference_latency[c-1], 3)}(s)\n", "light_blue"
        )

        if model_name == "gpt-3.5-turbo" or model_name == "gpt-4":
            extention = "json"
        else:
            extention = "txt"

        auxiliary.write_response(q, c, model, response, num, extention)
        model_params["question"] = question
        model_params["complete_prompt"] = prompt
        auxiliary.write_prompt_and_model_args(q, c, model, model_params, num)
        c += 1
