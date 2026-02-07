vlm_generation_guide={
    "type": "object",
    'properties': {
        "visual_state_description": {
            "type": "string",
            "description": "Description of current state from the visual image",
        },
        "reasoning_and_reflection": {
            "type": "string",
            "description": "summarize the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task",
        },
        "language_plan": {
            "type": "string",
            "description": "The action plan in natural language, which is a detailed description of the next action to achieve the user instruction. It should be a single sentence that describes the plan and the actions to be taken.",
        },
        "executable_plan": {
            "type": "array",
            "description": "The action plan in executable format, which contains some action objects. They will be exeucuted one by one. The action function name should be one of the functions defined in the action space, and the parameters should be the parameters required by the action function. ",
            "items": {
                "type": "object",
                "description": "An action to be executed by the robot, which contains the action function id, action function name and its parameters.",
                "properties": {
                    "action_id": {
                        "type": "string",
                        "description": "The action function id, which should be one of the functions defined in action space",
                        "enum": ["2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9"]
                    },
                    "action_name":{
                        "type": "string",
                        "description": "The action function name, which should be one of the functions defined in action space",
                        "enum": ["grasp_actor", "place_actor", "move_by_displacement", "move_to_pose", "close_gripper", "open_gripper", "back_to_origin", "get_arm_pose"]
                    },
                    "parameters": {
                        "type": "object",
                        "description": "The parameters required by the action function.",
                        "properties": {
                            "target_pose": {
                                "type": "array",
                                "description": "7-dim pose [x, y, z, qx, qy, qz, qw]",
                                "items": {"type": "number"},
                                "minItems": 7,
                                "maxItems": 7
                            }
                        },
                        "additionalProperties": {
                            "type": ["string", "number", "boolean", "array", "object"],
                            "description": "The value of other parameters."
                        }
                    }
                },
                "required": ["action_id", "action_name", "parameters"]
            },
        }
    },
    "required": ["visual_state_description", "reasoning_and_reflection","language_plan", "executable_plan"]
}
