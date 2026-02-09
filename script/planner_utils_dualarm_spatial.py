import os
import re
import base64
import copy
from mimetypes import guess_type
# import google.generativeai as genai
# from openai import OpenAI, AzureOpenAI
# import typing_extensions as typing
# from pydantic import BaseModel, Field
import json

template_spatial = '''
You are a dual-arm robot manipulation assistant. You are designed to finish dual-arm manipulation task. However, now you just need to analyze the given observation image and decide which arm (left or right) should perform the grasping action.\n
The observation image will contain two robotic arms and some objects on a table. You need to analyze the observation image and determine which robotic arm should be used to grasp some given objects.\n
Your output should be json format and clearly indicate:

VISUAL_STATE_DESCRIPTION(describe what you see) and RESULTS. RESULTS should be a array. Every item in array is a json object which contain OBJECT and its USE_ARM.

!!! Json format is strict. When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
!!! And do not output any ""(quotation marks) in JSON text content(in visual_state_description no quote!!!) like "red_block", you can use RED_BLOCK or (red_block) instead. Else json object can't be loaded successfully. Pay attention to make your output json identifiable!

# This is a output example for this task((LEFT OR RIGHT) means you should choose one):
'''
def clean_json_markdown(text: str) -> str:
    text = re.sub(r"^```(?:json)?\n?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return text.strip()

def fix_json(json_str,fix_mode="Default"):
    """
    Locates the substring between the keys "reasoning_and_reflection" and "language_plan"
    and escapes any inner double quotes that are not already escaped.
    
    The regex uses a positive lookahead to stop matching when reaching the delimiter for the next key.
    """
    # first fix common errors
    json_str = json_str.replace("'",'"') 
    json_str = json_str.replace('\"s ', "\'s ")
    json_str = json_str.replace('\"re ', "\'re ")
    json_str = json_str.replace('\"ll ', "\'ll ")
    json_str = json_str.replace('\"t ', "\'t ")
    json_str = json_str.replace('\"d ', "\'d ")
    json_str = json_str.replace('\"m ', "\'m ")
    json_str = json_str.replace('\"ve ', "\'ve ")
    json_str = json_str.replace('```json', '').replace('```', '')

    # Then fix some situations. Pattern explanation:
    # 1. ("reasoning_and_reflection"\s*:\s*") matches the key and the opening quote.
    # 2. (?P<value>.*?) lazily captures everything in a group named 'value'.
    # 3. (?=",\s*"language_plan") is a positive lookahead that stops matching before the closing quote
    #    that comes before the "language_plan" key.
    pattern = r'("reasoning_and_reflection"\s*:\s*")(?P<value>.*?)(?=",\s*"language_plan")'
    
    def replacer(match):
        prefix = match.group(1)            # Contains the key and the opening quote.
        value = match.group("value")         # The raw value that might contain unescaped quotes.
        # Escape any double quote that is not already escaped.
        fixed_value = re.sub(r'(?<!\\)"', r'\\"', value)
        return prefix + fixed_value
    
    def extract_first_json(text: str):
      brace_count = 0
      start = None
      for i, ch in enumerate(text):
          if ch == '{':
              if brace_count == 0:
                  start = i
              brace_count += 1
          elif ch == '}':
              brace_count -= 1
              if brace_count == 0 and start is not None:
                  return text[start:i+1] 
      return None
    
    def clean_json_top_tail(text):
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or start > end:
            return text
        return text[start:end+1]
    # Use re.DOTALL so that newlines in the value are included.
    fixed_json = re.sub(pattern, replacer, json_str, flags=re.DOTALL)
    fixed_json = re.sub(r'\\\"', '\'', fixed_json)
    fixed_json = fixed_json.replace("True", "true").replace("False", "false").replace("None", "null")
    try:
        json_object = json.loads(fixed_json)
    except:
        fixed_json = clean_json_top_tail(fixed_json)
        try:
            json_object = json.loads(fixed_json)
            print("json error fix success!!!")
        except:
            if fix_mode == "Claude":
                try:
                  jsontext=extract_first_json(fixed_json)
                  json_object = json.loads(jsontext)
                  fixed_json=jsontext
                  print("json error fix success!!!")
                except:
                  print("json error fix failed!!!")
            else:
                try:
                  jsontext=extract_first_json(fixed_json)
                  json_object = json.loads(jsontext)
                  fixed_json=jsontext
                  print("json error fix success!!!")
                except:
                  try:
                    fixed_json = fixed_json+"]}"
                    json_object = json.loads(fixed_json)
                    print("json error fix success!!!")
                  except:
                    print("json error fix failed!!!")
    
    return fixed_json


import re


def convert_format_2claude(messages):
    new_messages = []
    
    for message in messages:
        if message["role"] == "user":
            new_content = []
    
            for item in message["content"]:
                if item.get("type") == "image_url":
                    base64_data = item["image_url"]["url"][22:]
                    new_item = {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_data
                        }
                    }
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            new_message = message.copy()
            new_message["content"] = new_content
            new_messages.append(new_message)

        else:
            new_messages.append(message)

    return new_messages

def convert_format_2gemini(messages, mode="image_url"):
    new_messages = []
    
    for message in messages:
        if message["role"] == "user":
            new_content = []
            
            for item in message["content"]:
                if item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    
                    if mode == "image_url":
                        new_item = item
                    
                    elif mode == "inline_data":
                        if url.startswith("data:"):
                            base64_data = url.split(",")[1]
                        else:
                            base64_data = url
                        new_item = {
                            "type": "input_image",
                            "image_data": base64_data
                        }
                    
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            new_message = {**message, "content": new_content}
            new_messages.append(new_message)

        else:
            new_messages.append(message)
    
    return new_messages


# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"