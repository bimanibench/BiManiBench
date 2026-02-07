import torch
import re
import os
import time
import glob
import numpy as np
import cv2
import json
from generation_guide import vlm_generation_guide
from planner_utils_dualarm_spatial import local_image_to_data_url, template_spatial, fix_json
from remote_model_dualarm_spatial import RemoteModel
# from embodiedbench.planner.custom_model import CustomModel
# from embodiedbench.main import logger
writeIt = 1

def write_print(content,file_path):  
    global writeIt
    with open(file_path, 'a',encoding="utf-8") as f:
        print(f"step {writeIt}: ***********************************************************************************",file=f)
        writeIt+=1
        print(content,file=f)

class VLMPlanner():
    def __init__(self, model_name, model_type, obs_key='head_rgb', 
                chat_history=False, language_only=False, use_feedback=True, multistep=0, tp=1, kwargs={},prompt_out_path = "prompt_output.txt",user_instruction=''):
        self.model_name = model_name
        self.obs_key = obs_key
        # self.system_prompt = system_prompt
        self.chat_history = chat_history # whether to includ all the chat history for prompting
        self.model_type = model_type
        self.user_instruction = user_instruction
        self.model = RemoteModel(model_name, model_type, language_only, tp=tp)

        self.use_feedback = use_feedback
        self.multistep = multistep
        self.planner_steps = 0
        self.output_json_error = 0
        self.language_only = language_only
        self.kwargs = kwargs

        self.prompt_out_path = prompt_out_path
    

    def get_message(self, image, prompt,example_image_path=None,messages=[],assitantImage=None):
            print(image)
            if type(image) == str:
                image_path = image 
            else:
                image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                cv2.imwrite(image_path, image)
            data_url = local_image_to_data_url(image_path=image_path)
            # example_data_url = local_image_to_data_url(image_path=example_image_path)
            if assitantImage is not None:
                content = [{
                            "type": "image_url",
                            "image_url": {
                                        "url": data_url,
                            }
                    }]
                for assitantImage_path in assitantImage:
                    assitant_data_url = local_image_to_data_url(image_path=assitantImage_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": assitant_data_url
                        }  
                    })
                content.append({
                        "type": "text", 
                        "text": prompt
                    })
                if example_image_path is not None:
                    example_data_url = local_image_to_data_url(image_path=example_image_path)
                    content.append({
                        "type": "image_url",
                        "image_url":{
                            "url": example_data_url
                        }
                    })
            else:
                content = [{
                            "type": "image_url",
                            "image_url": {
                                        "url": data_url,
                            }
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    }
                    ]
                if example_image_path is not None:
                    example_data_url = local_image_to_data_url(image_path=example_image_path)
                    content.append({
                        "type": "image_url",
                        "image_url":{
                            "url": example_data_url
                        }
                    })
            return messages + [
                {
                    "role": "user",
                    "content": content,
                }
            ]

    def set_task(self,task_name):
        self.task_name = task_name

    def set_template_example(self,template_example):
        self.template_example = template_example

    def reset(self):
        # at the beginning of the episode
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0
        global writeIt
        writeIt = 1

    def language_to_action(self, output_text):
        pattern = r'\*\*\d+\*\*'
        match = re.search(pattern, output_text)
        if match:
            action = int(match.group().strip('*'))
        else:
            print('random action')
            action = np.random.randint(len(self.actions))
        return action
    
    def json_to_results(self, output_text, json_key='results'):
        try:
            json_object = json.loads(output_text)
            resultsList = json_object.get(json_key, {})
            if isinstance(resultsList, list):
                if len(resultsList) == 0:
                    print("actionList is empty, return a zero array")
                    self.output_json_error += 1
                    return {}
            return resultsList
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            self.output_json_error += 1
            return {}
        except Exception as e:
            print("An unexpected error occurred:", e)
            return {}

    def act(self, observation,assistant_obs=None, assistant_info=None):
        # print("act_feedback:", self.episode_act_feedback)
        if type(observation) == dict:
            obs = observation[self.obs_key]
        else:
            obs = observation # input image path
        # print(obs)
        if assistant_info is None:
            prompt =  template_spatial + self.template_example
        else:
            prompt = '!!!!! Assistant info (Very Important to provide some key info): '+ assistant_info + '\n'
            prompt = prompt + template_spatial + self.template_example + '\n\n'

        write_print(prompt,self.prompt_out_path)
        # example_image_path=f"script/example_img/{self.task_name}.png"
        self.episode_messages = self.get_message(obs,prompt=prompt)
        # print()
        try: 
                out = self.model.respond(self.episode_messages)
        except Exception as e:
                print("An unexpected error occurred:", e)

                if self.model_type != 'local':
                    time.sleep(60)
                else:
                    time.sleep(20)
                out = self.model.respond(self.episode_messages)

        results = self.json_to_results(out)
        self.planner_steps += 1
        return results, out

    def update_info(self, info):
        """Update episode feedback history."""
        self.episode_act_feedback.append([
            info['action_list'],
            info['action_feedback'],
        ])
    def set_prompt_output_path(self, prompt_out_path):
        """Set the path to save the prompt output."""
        self.prompt_out_path = prompt_out_path
    
    def set_task_instruction(self, user_instruction):
        """Set the task instruction."""
        self.user_instruction = user_instruction
        print("set_task_instruction:", self.user_instruction)
        

