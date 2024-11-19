import os
import os.path as osp
import warnings
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import anthropic
import openai
MAX_OPENAI_RETRY, OPENAI_SLEEP_TIME = 5, 60
MAX_CLAUDE_RETRY, CLAUDE_SLEEP_TIME = 100, 5

registered_text_completion_llms = {
    "gpt-4-1106-preview",
    "gpt-4-0125-preview", "gpt-4-turbo-preview",
    "gpt-4-turbo", "gpt-4-turbo-2024-04-09"
    "gpt-4-turbo",
    "claude-2.1",
    "claude-3-opus-20240229", 
    "claude-3-sonnet-20240229", 
    "claude-3-haiku-20240307",
    "huggingface/codellama/CodeLlama-7b-hf"
}

def get_gpt_output(message, 
                   model="gpt-4-1106-preview", 
                   max_tokens=2048, 
                   temperature=1, 
                   max_retry=5,
                   sleep_time=60,
                   json_object=False,
                   history=None,
                   tools=[],
                   return_raw=False):
    if json_object:
        if isinstance(message, str) and not 'json' in message.lower():
            message = 'You are a helpful assistant designed to output JSON. ' + message

    if isinstance(message, str):
        messages = [{"role": "user", "content": message}] 
    else:
        messages = message
    if history:
        messages = history + messages
    kwargs = {"response_format": { "type": "json_object" }} if json_object else {}

    for cnt in range(max_retry):
        try:
            chat = openai.OpenAI().chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
                )
            result = chat.choices[0].message.content 
        except Exception as e:
            error = e
            print(cnt, "=>", e, f' [sleep for {sleep_time} sec]')
            time.sleep(sleep_time)
            continue
            
        try:
            if json_object:
                result = result[result.find("{"):result.rfind("}")+1]
                return json.loads(result)
            return result
        except Exception as e:
            print(result)
            print(cnt, "=> (json encode error) ", e)
    raise error

def complete_text_claude(message, 
                         model="claude-2.1",
                         json_object=False,
                         max_tokens=2048, 
                         temperature=1, 
                         max_retry=1,
                         sleep_time=0,
                         tools=[],
                         history=None,
                         return_raw=False,
                         **kwargs
                         ):
    try:   
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        anthropic_client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        print(e)
        print("Anthropic API key not found.")
    """ Call the Claude API to complete a prompt."""
    if isinstance(message, str):
        if json_object:
            message = "You are a helpful assistant designed to output in JSON format." + message
        messages = [{"role": "user", "content": message}]
    else:
        messages = message
    
    if history is not None:
        messages = history + messages

    result = None
    e = None
    for cnt in range(max_retry):
        try:
            result = anthropic_client.beta.tools.messages.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                **kwargs
            )
            if not return_raw:
                result = result.to_dict()["content"][0]['text']
        except Exception as e:
            print(cnt, "=>", e, f' [sleep for {sleep_time} sec]')
            time.sleep(sleep_time)
            continue
        
        try:
            if json_object and not return_raw:
                return json.loads(result)
            return result
        except Exception as e:
            print(result)
            print(cnt, "=> (json encode error) ", e)
    raise e

loaded_hf_models = {}

def complete_text_hf(message, 
                     model="huggingface/codellama/CodeLlama-7b-hf", 
                     max_tokens=2000, 
                     temperature=0.5, 
                     json_object=False,
                     max_retry=1,
                     sleep_time=0,
                     stop_sequences=[], 
                     **kwargs):
    if json_object:
        message = "You are a helpful assistant designed to output in JSON format." + message
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.split("/", 1)[1]
    if model in loaded_hf_models:
        hf_model, tokenizer = loaded_hf_models[model]
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(model).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model)
        loaded_hf_models[model] = (hf_model, tokenizer)
        
    encoded_input = tokenizer(message, 
                              return_tensors="pt", 
                              return_token_type_ids=False
                              ).to(device)
    for cnt in range(max_retry):
        try:
            output = hf_model.generate(
                **encoded_input,
                temperature=temperature,
                max_new_tokens=max_tokens,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs,
            )
            sequences = output.sequences
            sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]
            all_decoded_text = tokenizer.batch_decode(sequences)
            completion = all_decoded_text[0]
            return completion
        except Exception as e:
            print(cnt, "=>", e)
            time.sleep(sleep_time)
    raise e

def get_llm_output_tools(message,
                   tools=[],
                   model="gpt-4-0125-preview", 
                   max_tokens=2048, 
                   temperature=1, 
                   json_object=False,
                   history=None,
                   return_raw=False
                   ):
    '''
    A general function to complete a prompt using the specified model.
    '''
    if model not in registered_text_completion_llms:
        warnings.warn(f"Model {model} is not registered. You may still be able to use it.")
    kwargs = {'message': message, 
              'model': model, 
              'max_tokens': max_tokens, 
              'temperature': temperature, 
              'json_object': json_object,
              'history': history,
              'tools': tools,
              'return_raw': return_raw}
    
    if 'gpt-4' in model:
        kwargs.update({'max_retry': MAX_OPENAI_RETRY, 'sleep_time': OPENAI_SLEEP_TIME})
        return get_gpt_output(**kwargs)
    elif 'claude' in model:
        kwargs.update({'max_retry': MAX_CLAUDE_RETRY, 'sleep_time': CLAUDE_SLEEP_TIME})
        return complete_text_claude(**kwargs)
    elif 'huggingface' in model:
        return complete_text_hf(**kwargs)
    else:
        raise ValueError(f"Model {model} not recognized.")