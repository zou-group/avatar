import time
import base64
import json
import openai
import requests
import os
from typing import List, Dict, Any
from PIL import Image
from stark_qa.tools.api import complete_text_claude, parallel_func
from avatar.utils.image import image_to_base64

MAX_OPENAI_RETRY = int(os.getenv("MAX_OPENAI_RETRY", 5))
OPENAI_SLEEP_TIME = int(os.getenv("OPENAI_SLEEP_TIME", 60))
MAX_CLAUDE_RETRY = int(os.getenv("MAX_CLAUDE_RETRY", 10))
CLAUDE_SLEEP_TIME = int(os.getenv("CLAUDE_SLEEP_TIME", 0))
LLM_PARALLEL_NODES = int(os.getenv("LLM_PARALLEL_NODES", 5))


def complete_text_image_claude(image: Image.Image,
                               message: str, 
                               image_path: str = None,
                               model: str = "claude-3-opus-20240229", 
                               max_tokens: int = 1024, 
                               temperature: int = 1,
                               tools: List = [],
                               json_object: bool = False,
                               history: List[Dict[str, Any]] = None,
                               max_retry: int = 3, 
                               sleep_time: int = 0,
                               **kwargs) -> Dict:
    """ Call the Claude API to complete a prompt."""
    if image is not None:
        base64_image = image_to_base64(image)
    elif image_path is not None:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        raise ValueError("Either image_path or image_data must be provided.")
    if json_object:
        message = "You are a helpful assistant designed to output in JSON format." + message
    image_media_type = "image/jpeg"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": message,
                }
            ],
        }
    ]
    if history is not None:
        messages = history + [{"role": "user", "content": message}]
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    kwargs["max_retry"] = max_retry
    kwargs["sleep_time"] = sleep_time
    return complete_text_claude(messages,
                                model=model,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                json_object=json_object,
                                tools=tools,
                                **kwargs)


def get_gpt4v_output(image: Image.Image, 
                     message: str, 
                     model: str = "gpt-4-turbo", 
                     max_tokens: int = 1024, 
                     max_retry: int = 10, 
                     sleep_time: int = 10,
                     json_object: bool = False,
                     **kwargs) -> Dict:
    if json_object:
        if isinstance(message, str) and 'json' not in message.lower():
            message = 'You are a helpful assistant designed to output JSON. ' + message
    base64_image = image_to_base64(image)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }

    payload = {
        "model": model,
        "max_tokens": max_tokens, 
        "messages": [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": message},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
    }
    if json_object:
        payload['response_format'] = {"type": "json_object"}
    for cnt in range(max_retry):
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                     headers=headers, json=payload)
            result = response.json()['choices'][0]['message']['content']
        except Exception as e:
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
    raise e


def get_llm_vision_output(image: Image.Image,
                          message: str,
                          model: str = "claude-3-opus-20240229",
                          max_tokens: int = 1024,
                          temperature: int = 1,
                          json_object: bool = False) -> Dict:
    kwargs = {
        'message': message, 
        'image': image, 
        'model': model, 
        'max_tokens': max_tokens, 
        'temperature': temperature, 
        'json_object': json_object
    }
    
    if 'claude' in model:
        kwargs.update({'max_retry': MAX_CLAUDE_RETRY, 'sleep_time': CLAUDE_SLEEP_TIME})
        return complete_text_image_claude(**kwargs)
    if 'gpt-4' in model:
        kwargs.update({'max_retry': MAX_OPENAI_RETRY, 'sleep_time': OPENAI_SLEEP_TIME})
        return get_gpt4v_output(**kwargs)
    else:
        raise ValueError(f"Model {model} not recognized.")


get_llm_vision_outputs = parallel_func(get_llm_vision_output)
