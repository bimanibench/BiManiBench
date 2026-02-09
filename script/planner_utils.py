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

template = '''
The images in every observation are the current state of the environment, involving the image from head_camera, front_camera, third_view_camera, and the example image of the task to help you make decisions.
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan': str}
The fields in above JSON follows the purpose below:
1. visual_state_description is for description of current state from the visual image, and you should describe the gripper state by your eyes. 
2. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, 
3. language_plan is for describing the following action to achieve the user instruction.
4. executable_plan is an json array that contains the next action to achieve the user instruction. Every item of the json array is a json object that contains the action id (2.2 to 2.9) and the according action name. Additionally, you must provide the parameters for the action function. The instruction for the action function and parameters is as follows:

Action list is following, action id from 2.2 to 2.9. For some ambiguous parameters, please refer to the assistant info for settings; otherwise, don't output them and leave them as default:
{
  "2.2": {
    "name": "grasp_actor",
    "description": "Pick up a specified object using the selected arm.",
    "parameters": {
      "actor": "The object to grasp.",
      "arm_tag": "Which arm to use.",
      "pre_grasp_dis": "Pre-grasp distance (default 0.1 meters), the arm will move to this position first.",
      "grasp_dis": "Grasping distance (default 0 meters), the arm moves from the pre-grasp position to this position and then closes the gripper.",
      "gripper_pos": "Gripper closing position (default 0, fully closed).",
      "contact_point_id": "Optional list of contact point IDs; if not provided, the best grasping point is selected automatically."
    },
    "returns": "tuple[ArmTag, list[Action]]",
    "example": "self.move(self.grasp_actor(self.cup, arm_tag=arm_tag, pre_grasp_dis=0.1, contact_point_id=[0, 2][int(arm_tag=='left')]))"
  },
  "2.3": {
    "name": "place_actor",
    "description": "Places a currently held object at a specified target pose.",
    "parameters": {
      "actor": "The currently held object.",
      "arm_tag": "The arm holding the object.",
      "target_pose": "Target position/orientation, length 3 or 7 (xyz + optional quaternion). Please use 7 length rather than 3 length to avoid ambiguity.",
      "functional_point_id": "Optional ID of the functional point; aligns this point to the target if provided. Please don't provide unless it's specified in task-specific assistant info. It will be ignored. Use target_pose to finish this action task.",
      "pre_dis": "Pre-place distance (default 0.1 meters).",
      "dis": "Final placement distance (default 0.02 meters).",
      "is_open": "Whether to open the gripper after placing (default True).",
      "kwargs": {
        "constrain": "Alignment strategy: 'free', 'align', or 'auto' (default auto).",
        "align_axis": "Vectors in world coordinates to align with. (a optional parameter, typically, it is not necessary to specify a specific value.).",
        "actor_axis": "Second object axis used for alignment (default [1, 0, 0]).",
        "actor_axis_type": "Whether actor_axis is relative to 'actor' or 'world' (default 'actor').",
        "pre_dis_axis": "Direction of pre-placement offset: 'grasp', 'fp', or custom vector. (default 'grasp')."
      }
    },
    "returns": "tuple[ArmTag, list[Action]]",
    "example": "self.move(self.place_actor(actor=self.current_actor, target_pose=target_pose, arm_tag=arm_tag, functional_point_id=0, pre_dis=0.1, dis=0.02, pre_dis_axis='fp'))"
  },
  "2.4": {
    "name": "move_by_displacement",
    "description": "Moves the end-effector of the specified arm along relative directions and sets its orientation.",
    "parameters": {
      "arm_tag": "The arm to control.",
      "x": "Displacement along x-axis (meters).",
      "y": "Displacement along y-axis (meters).",
      "z": "Displacement along z-axis (meters).",
      "quat": "Optional quaternion specifying the target orientation.",
      "move_axis": "'world' or 'arm'; defines coordinate system (default 'world')."
    },
    "returns": "tuple[ArmTag, list[Action]]",
    "example": "self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis='world'))"
  },
  "2.5": {
    "name": "move_to_pose",
    "description": "Moves the end-effector of the specified arm to a specific absolute pose.",
    "parameters": {
      "arm_tag": "The arm to control.",
      "target_pose": "Absolute pose (xyz + optional quaternion)."
    },
    "returns": "tuple[ArmTag, list[Action]]",
    "example": "self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=target_pose))"
  },
  "2.6": {
    "name": "close_gripper",
    "description": "Closes the gripper of the specified arm.",
    "parameters": {
      "arm_tag": "Which arm's gripper to close.",
      "pos": "Gripper position (0 = fully closed)."
    },
    "returns": "tuple[ArmTag, list[Action]]",
    "example": "self.move(self.close_gripper(arm_tag=arm_tag))"
  },
  "2.7": {
    "name": "open_gripper",
    "description": "Opens the gripper of the specified arm.",
    "parameters": {
      "arm_tag": "Which arm's gripper to open.",
      "pos": "Gripper position (1 = fully open)."
    },
    "returns": "tuple[ArmTag, list[Action]]",
    "example": "self.move(self.open_gripper(arm_tag=arm_tag))"
  },
  "2.8": {
    "name": "back_to_origin",
    "description": "Returns the specified arm to its predefined initial position.",
    "parameters": {
      "arm_tag": "The arm to return to origin."
    },
    "returns": "tuple[ArmTag, list[Action]]",
    "example": "self.move(self.back_to_origin(arm_tag=ArmTag('right')))"
  },
  "2.9": {
    "name": "get_arm_pose",
    "description": "Gets the current pose of the end-effector of the specified arm.",
    "parameters": {
      "arm_tag": "Which arm to query."
    },
    "returns": "list[float]",
    "example": "pose = self.get_arm_pose(ArmTag('left'))"
  }
}

!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
!!! And do not output any ""(quotation marks) in JSON content like "can_right", you can use can_right or (can_right) instead. Else json object can't be loaded successfully and the action will be empty. 
# This is a proper output template example: 

'''

template_vla = '''
You are a robot manipulation assistant. Your task is to follow the instruction and move objects using **End-Effector Pose Control mode** for a dual-arm robot.
The action format is:
```
[left_end_effector_pose (xyz + quaternion) + left_gripper + right_end_effector_pose (xyz + quaternion) + right_gripper]
```
!!!NOTE: Please control the float number in output to be within 5 digits.
### Parameter Explanation:
1. **xyz**
   The position of the end-effector in the world coordinate system (unit: meters). Example: `[0.2, 0.3, 0.1]`.
2. **quaternion (qx, qy, qz, qw)**
   The orientation of the end-effector, represented as a quaternion. The first three values `(qx, qy, qz)` describe the rotation axis (a unit vector), and the last value `qw` is the cosine of half the rotation angle:
   [
   q = [\sin(\\theta/2) \cdot u_x,; \sin(\\theta/2) \cdot u_y,; \sin(\\theta/2) \cdot u_z,; \cos(\\theta/2)]
   ]
**Example**: A 90° rotation around the z-axis is:
[
[0.0, 0.0, \sin(90°/2), \cos(90°/2)] = [0.0, 0.0, 0.7071, 0.7071]
]
We use a **right-handed coordinate system**: if you point your right thumb along the positive axis, the curl of your fingers indicates the positive rotation direction.
Thus, looking **from the positive z-axis downward**, a +90° rotation is **counterclockwise**.

3. **Coordinate System Convention**
   In the environment visualization:
* Right side = positive x-axis
* Into the screen = positive y-axis
* Upward = positive z-axis
4. **gripper (open/close state)**
   Range `[0, 1]`:

* `0` = fully closed
* `1` = fully open
* e.g., `0.5` = half open.
---
### Action Example:
Scenario: An apple is located at `[0.2, 0.3, 0.0]`.
* **Action 1**
```
[0.2, 0.3, 0.15, 0.5, -0.5, 0.5, 0.5, 1.0, 0.5, 0.3, 0.1, 0.0, 0.5, -0.5, 0.5, 0.5]
```
Explanation: Move the left hand above the apple.
* **Action 2**
```
[0.2, 0.3, 0.08, 0.5, -0.5, 0.5, 0.5, 1.0, 0.5, 0.3, 0.1, 0.0, 0.5, -0.5, 0.5, 0.5]
```
Explanation: Lower the left hand to grasp the apple.

* **Action 3**
```
[0.2, 0.3, 0.08, 0.5, -0.5, 0.5, 0.5, 0, 0.5, 0.3, 0.1, 0.0, 0.5, -0.5, 0.5, 0.5]
```
Explanation: Close the left gripper to pick up the apple.
---
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
!!! And do not output any ""(quotation marks) in JSON content like "can_right", you can use can_right or (can_right) instead. Else json object can't be loaded successfully and the action will be empty. 
!!! you must control your output length within 3000 tokens. Else the output will be cut off and the action will be empty. And output nothing else except the json format below.
!!! The max step you can manipulate is limited, so try your best to use dual arm at the same time to finish the task. And try to output more than an action in every output.
# This is a proper output template example: 
'''
def clean_json_markdown(text: str) -> str:
    """移除 ```json ``` 或 ``` 包裹的代码块"""
    # 去掉 Markdown 包裹
    text = re.sub(r"^```(?:json)?\n?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return text.strip()

def clean_json_quote(json_string):
    def replacer(match):
        key = match.group(1)
        value = match.group(2)
        
        cleaned_value = value.replace('"', '')
        cleaned_value = cleaned_value.replace('{', '').replace('}', '')
        cleaned_value = cleaned_value.replace('\\', '')
        return f'"{key}": "{cleaned_value}"'
    pattern = re.compile(
        r'"(visual_state_description|reasoning_and_reflection|language_plan)"\s*:\s*"(.*?)"(?=\s*,\s*"\w+"|\s*})',
        re.DOTALL
    )
    
    # 使用 replacer 函数替换所有匹配到的部分
    cleaned_string = pattern.sub(replacer, json_string)
    return cleaned_string

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
    fixed_json = clean_json_quote(fixed_json)
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