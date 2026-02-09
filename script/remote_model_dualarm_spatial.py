import json
import sys
import os
import re
import base64
# import anthropic
# import google.generativeai as genai
from openai import OpenAI
# import typing_extensions as typing
# import lmdeploy
from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
from generation_guide import vlm_generation_guide
# from embodiedbench.planner.planner_config.generation_guide_manip import llm_generation_guide_manip, vlm_generation_guide_manip
from planner_utils import convert_format_2claude,convert_format_2gemini, fix_json

temperature = 0
max_completion_tokens = 4096
remote_url = os.environ.get('remote_url')

class RemoteModel:
    def __init__(
        self,
        model_name,
        model_type='remote',
        language_only=False,
        tp=1,
        task_type=None 
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.language_only = language_only
        self.task_type = task_type
        remote_url="http://localhost:8000/v1"
        if self.model_type == 'local':
            backend_config = PytorchEngineConfig(session_len=12000, dtype='float16', tp=tp)
            self.model = pipeline(self.model_name, backend_config=backend_config)
        else:
            if "claude" in self.model_name:
                self.model = OpenAI(
                    api_key="YOUR_API_KEY",
                    base_url="YOUR_API_URL"
                )
            elif "gemini" in self.model_name:
                self.model = OpenAI(
                    api_key="YOUR_API_KEY",
                    base_url="YOUR_API_URL"
                )
            elif "gpt" in self.model_name or "o4-mini" in self.model_name:
                self.model = OpenAI(
                    api_key="YOUR_API_KEY",
                    base_url="YOUR_API_URL"
                )
            elif 'qwen3' in self.model_name:
                self.model = OpenAI(
                    api_key="YOUR_API_KEY",
                    base_url="YOUR_API_URL"
                )
            elif "glm-4.5v" in self.model_name:
                self.model = OpenAI(
                    api_key="YOUR_API_KEY",
                    base_url="YOUR_API_URL"
                )
            elif "Qwen2-VL" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "Qwen2.5-VL" in self.model_name or "Qwen3" in self.model_name:
                self.model = OpenAI(base_url = remote_url,api_key="00000001")
            elif "Ovis2" in self.model_name:
                self.model = OpenAI(base_url = remote_url,api_key="00000002")
            elif "Llama-3.2-90B-Vision-Instruct" in self.model_name:
                self.model = OpenAI(base_url = remote_url,api_key="00000002")
            elif "Llama-4-Scout-17B-16E-Instruct" in self.model_name:
                self.model = OpenAI(base_url = remote_url,api_key="00000002")
            elif "LLM-Research" in self.model_name:
                self.model = OpenAI(base_url = remote_url,api_key="00000002")
            elif "OpenGVLab/InternVL" in self.model_name:
                if "78B" in self.model_name:
                    self.model = OpenAI(base_url = remote_url,api_key="00000000")
                else:
                    self.model = OpenAI(base_url = remote_url,api_key="00000000")
            elif "meta-llama/Llama-3.2-90B-Vision-Instruct" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "90b-vision-instruct" in self.model_name: # you can use fireworks to inference
                self.model = OpenAI(base_url='https://api.fireworks.ai/inference/v1',
                                    api_key=os.environ.get("firework_API_KEY"))
            else:
                try:
                    self.model = OpenAI(base_url = remote_url)
                except:
                    raise ValueError(f"Unsupported model name: {model_name}")


    def respond(self, message_history: list):
        if self.model_type == 'local':
            return self._call_local(message_history)
        else:
            if "claude" in self.model_name:
                return self._call_claude(message_history)
            elif "gemini-2.5-pro" in self.model_name:
                return self._call_gemini_2_5_pro(message_history)
            elif "gemini" in self.model_name:
                return self._call_gemini(message_history)
            elif "gpt-5" in self.model_name:
                return self._call_gpt5(message_history)
            elif "gpt" in self.model_name or "o4-mini" in self.model_name:
                return self._call_gpt(message_history)
            elif 'qwen3' in self.model_name:
                return self._call_gpt(message_history)
            elif "glm-4.5v" in self.model_name:
                return self._call_gpt(message_history)
            elif "/data3/wuxin/model" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Qwen2-VL-7B-Instruct" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Qwen2.5-VL-7B-Instruct" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Qwen2.5-VL-32B-Instruct" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Ovis2" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "LLM-Research" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Qwen3-VL" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Qwen2.5-VL-72B-Instruct" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Llama-3.2-11B-Vision-Instruct" in self.model_name:
                return self._call_llama11b(message_history)
            elif "meta-llama/Llama-3.2-90B-Vision-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "Llama-4-Scout-17B-16E-Instruct" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "OpenGVLab/InternVL" in self.model_name:
                return self._call_intern38b(message_history)
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")

    def _call_local(self, message_history: list):
        response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "embodied_planning",
                    "schema": vlm_generation_guide
                }
        }
        response = self.model(
            message_history,
            gen_config=GenerationConfig(
                temperature=temperature,
                response_format=response_format,
                max_new_tokens=max_completion_tokens,
            )
        )
        out = response.text
        out = fix_json(out)
        return out

    def _call_claude(self, message_history: list):

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            # response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        
        out = response.choices[0].message.content
        out = fix_json(out,fix_mode="Claude")
        return out

    def _call_gemini(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        response = self.model.beta.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            reasoning_effort="none",
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        
        msg = response.choices[0].message
        out = msg.content
        out = fix_json(out)
        return out
    
    def _call_gemini_2_5_pro(self, message_history: list):
        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        response = self.model.beta.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            reasoning_effort="low",  
            extra_body={
                "google": {
                    "thinking_config": {
                        "thinking_budget": 128,   
                        "include_thoughts": False 
                    }
                }
            },
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        msg = response.choices[0].message
        out = msg.content
        out = fix_json(out)
        return out


    def _call_gpt(self, message_history: list):
        # response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            # response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        msg = response.choices[0].message
        out = msg.content
        if not out:  
            if hasattr(msg, "refusal") and msg.refusal:
                out = msg.refusal
            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                out = str(msg.tool_calls)  
            else:
                out = ""
        out = fix_json(out)
        return out
    
    def _call_gpt5(self, message_history: list):

        response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            # response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens,
            reasoning_effort="minimal",
        )
        msg = response.choices[0].message

        out = msg.content
        if not out:  
            if hasattr(msg, "refusal") and msg.refusal:
                out = msg.refusal
            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                out = str(msg.tool_calls)  
            else:
                out = ""
        out = fix_json(out)
        return out
    
    def _call_qwen7b(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        # response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            # response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )

        out = response.choices[0].message.content
        out = fix_json(out)
        return out

    def _call_ovis9b(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )

        out = response.choices[0].message.content
        out = fix_json(out)
        return out
    
    def _call_llama11b(self, message_history):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            # response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        out = response.choices[0].message.content
        out = fix_json(out)
        return out
    

    def _call_qwen72b(self, message_history):
        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            # response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )

        out = response.choices[0].message.content
        out = fix_json(out)
        return out
    
    def _call_intern38b(self, message_history):

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            # response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens,
        )

        out = response.choices[0].message.content
        out = fix_json(out)
        print("response.usage:", response.usage)
        return out

def save_base64_to_txt(base64_data, output_file):
    with open(output_file, "w") as file:
        file.write(base64_data)

if __name__ == "__main__":

    model = RemoteModel(model_name="claude-sonnet-4-20250514",model_type="remote")

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        

    base64_image = encode_image("test.png")
    messages=[
        {
            "role": "user",
            "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                }
            },
            {
                "type": "text",
                "text":f"What do you think for this picture??"
            },
            ],
        }
    ]
    response = model.respond(messages)
    print(response)

