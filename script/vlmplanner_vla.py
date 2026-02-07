import torch
import re
import os
import time
import glob
import numpy as np
import cv2
import json
from generation_guide import vlm_generation_guide
from planner_utils import local_image_to_data_url, template_vla, fix_json
from remote_model import RemoteModel
writeIt = 1

def write_print(content,file_path):  
    global writeIt
    with open(file_path, 'a',encoding="utf-8") as f:
        print(f"step {writeIt}: ***********************************************************************************",file=f)
        writeIt+=1
        print(content,file=f)

class VLMPlanner():
    def __init__(self, model_name, model_type, system_prompt, obs_key='head_rgb', 
                chat_history=False, language_only=False, use_feedback=True, multistep=0, tp=1, kwargs={},prompt_out_path = "prompt_output.txt",user_instruction=''):
        self.model_name = model_name
        self.obs_key = obs_key
        self.system_prompt = system_prompt
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

    def process_prompt(self,prev_act_feedback=[]):
        user_instruction = self.user_instruction
        n = len(prev_act_feedback)
        if n == 0:
            prompt = ""
        elif self.chat_history:
            prompt = 'The 3-steps action history:'
            for i in range(max(0,n-3),n):
                action_feedback = prev_act_feedback[i]
                prompt += '\nStep {}, actionList {}, action_feedback:{}'.format(i, action_feedback[0], action_feedback[1])

            prompt += f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the actions.'''
        else:
            print("???? vlmplanner.py: error")
        return prompt
    

    def get_message(self, image, prompt,example_image_path,messages=[],assitantImage=None):
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
    
    def json_to_action(self, output_text, json_key='executable_plan'):
        try:
            json_object = json.loads(output_text)
            actionList = json_object.get(json_key, {})
            if isinstance(actionList, list):
                if len(actionList) == 0:
                    print("actionList is empty, return a zero array")
                    self.output_json_error += 1
                    # action = -1
                    return {}
            return actionList
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            self.output_json_error += 1
            # action = -1
            return {}
        except Exception as e:
            # Catch-all for any other unexpected errors not handled specifically
            print("An unexpected error occurred:", e)
            # self.output_json_error += 1
            # action = -1
            return {}

    def act(self, observation,assistant_obs=None, assistant_info=None, actionChunkNum=0,vic_mode=False):
        # print("act_feedback:", self.episode_act_feedback)
        def list_task_images(task_name, base_dir="script/example_img"):
            pattern = os.path.join(base_dir, f"{task_name}_vic*.png")
            files = glob.glob(pattern)
            def extract_num(fname):
                base = os.path.basename(fname)
                num = base.replace(f"{task_name}_vic", "").replace(".png", "")
                return int(num) if num.isdigit() else 0  # 没数字的记作0

            files_sorted = sorted(files, key=extract_num)
            return files_sorted
    
        if type(observation) == dict:
            obs = observation[self.obs_key]
        else:
            obs = observation # input image path
        # print(obs)
        if assistant_info is None:
            prompt = self.process_prompt(prev_act_feedback=self.episode_act_feedback)
            prompt = prompt + template_vla + self.template_example
        else:
            prompt = self.process_prompt(prev_act_feedback=self.episode_act_feedback) + '\n' + '!!!!! Assistant info (Very Important to provide some key info): '+ assistant_info + '\n'
            prompt = prompt + template_vla + self.template_example + '\n\n'

        write_print(prompt,self.prompt_out_path)
        example_image_path=f"script/example_img/{self.task_name}.png"
        #check if the example image exists
        if not os.path.exists(example_image_path):
            example_image_path=None
        if assistant_obs is not None:
            if vic_mode:
                assistant_obs_new = list_task_images(self.task_name)
                print("Test: vic_mode is open. The num of vic images:", len(assistant_obs_new))
                if(isinstance(assistant_obs,list)):
                    # assistant_obs.append(f"script/example_img/{self.task_name}.png")
                    assistant_obs.extend(assistant_obs_new)
                else:
                    assistant_obs = [assistant_obs] + assistant_obs_new
                self.episode_messages = self.get_message(obs,assitantImage=assistant_obs,prompt=prompt+'\n'+"Note: The example I provide is very helpful, please check the example carefully and try to imitate this example to plan and output.",example_image_path=example_image_path)
            else:
                self.episode_messages = self.get_message(obs,assitantImage=assistant_obs,prompt=prompt,example_image_path=example_image_path)
        else:
            if vic_mode:
                assistant_obs = list_task_images(self.task_name)
                print("Test: vic_mode is open. The num of vic images:", len(assistant_obs_new))
                self.episode_messages = self.get_message(obs,assitantImage=assistant_obs,prompt=prompt+'\n'+"Note: The example I provide is very helpful, please check the example carefully and try to imitate this example to plan and output.",example_image_path=example_image_path)
            else:
                self.episode_messages = self.get_message(obs,prompt=prompt,example_image_path=example_image_path)
        try: 
                out = self.model.respond(self.episode_messages)
        except Exception as e:
                print("An unexpected error occurred:", e)

                if self.model_type != 'local':
                    time.sleep(60)
                else:
                    time.sleep(20)
                out = self.model.respond(self.episode_messages)

        actionList = self.json_to_action(out)
        self.planner_steps += 1
        if(actionChunkNum!=0 and len(actionList)>0):
            actionList = actionList[0:min(actionChunkNum, len(actionList))]
        return actionList, out

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
        

