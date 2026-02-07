from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
import random
import ast

class stack_and_group_blocks(Base_Task):
    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)
        self.playtestcount = 0

    def load_actors(self):
        # 六个方块颜色
        color_map = {
            "red": (1, 0, 0),
            "yellow": (1, 1, 0),
            "blue": (0, 0, 1),
            "green": (0, 1, 0),
            "black": (0, 0, 0),
            "white": (1, 1, 1),
        }
        self.block_colors = list(color_map.keys())

        # 六个方块大小随机，但不同
        self.halfsize_map = {
            "red": 0.025,
            "yellow": 0.02,
            "blue": 0.024,
            "green": 0.019,
            "black": 0.022,
            "white": 0.018,
        }

        # 随机生成初始位姿
        self.blocks = {}
        used_positions = []

        for name in self.block_colors:
            block_half_size = self.halfsize_map[name]
            pose = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=[-0.13, 0.1],
                zlim=[0.741 + block_half_size],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
            )

            # 避免初始位置过近
            while any(np.sum((pose.p[:2] - p[:2]) ** 2) < 0.01 for p in used_positions):
                pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[-0.13, 0.1],
                    zlim=[0.741 + block_half_size],
                    qpos=[1, 0, 0, 0],
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )

            used_positions.append(pose.p)
            self.blocks[name] = create_box(
                scene=self,
                pose=pose,
                half_size=(block_half_size, block_half_size, block_half_size),
                color=color_map[name],
                name=name
            )
            self.blocks[name].halfsize = block_half_size
            self.add_prohibit_area(self.blocks[name], padding=0.05)

        # 保存每个 block 的引用
        self.block_red = self.blocks["red"]
        self.block_yellow = self.blocks["yellow"]
        self.block_blue = self.blocks["blue"]
        self.block_green = self.blocks["green"]
        self.block_black = self.blocks["black"]
        self.block_white = self.blocks["white"]

                # Generate random y position for all blocks
        y_pose = np.random.uniform(-0.2, -0.13)

        # Define target poses for each block with random x positions
        self.block1_target_pose = [
            np.random.uniform(-0.1, -0.09),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]
        self.block2_target_pose = [
            np.random.uniform(0.01, 0.02),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]
        self.block3_target_pose = [
            np.random.uniform(0.08, 0.09),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]


    def play_once(self):
        # 默认抓取顺序可以按列堆叠
        # 列1：红->黄, 列2：蓝->绿, 列3：黑->白
        columns = [
            [self.block_red, self.block_yellow],
            [self.block_blue, self.block_green],
            [self.block_black, self.block_white]
        ]
        last_gripper = None

        # 左到右放置列
        x_positions = [-0.1, 0.0, 0.1]
        y_position = np.random.uniform(-0.2, -0.1)

        for col_idx, col in enumerate(columns):
            base_x = x_positions[col_idx]
            for i, block in enumerate(col):
                target_pose = [
                    base_x,
                    y_position,
                    0.74 + self.table_z_bias + i * 0.05,  # 堆叠
                    0, 1, 0, 0
                ]
                arm_tag = ArmTag("left" if block.get_pose().p[0] < 0 else "right")

                if last_gripper is not None and (last_gripper != arm_tag):
                    self.move(
                        self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09),
                        self.back_to_origin(arm_tag=arm_tag.opposite),
                    )
                else:
                    self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))

                self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))
                self.move(self.place_actor(block, target_pose=target_pose, arm_tag=arm_tag, functional_point_id=0,
                                           pre_dis=0.09, dis=0.02, constrain="align"))
                self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis="arm"))

                last_gripper = arm_tag

    def play_test(self):
        input("Press Enter to start the test...")

        # 定义动作序列，每个元素是一个函数调用
        actions = [
            # Red column
            lambda: self.move(self.grasp_actor(self.block_red, arm_tag="left", pre_grasp_dis=0.09)),
            lambda: self.move(self.move_by_displacement(arm_tag="left", z=0.15)),
            lambda: self.move(self.place_actor(self.block_red, target_pose=self.block1_target_pose,
                                            arm_tag="left", functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align")),
            lambda: self.move(self.move_by_displacement(arm_tag="left", z=0.15, move_axis="arm")),
            lambda: self.move(self.back_to_origin(arm_tag="left")),

            # Blue column
            lambda: self.move(self.grasp_actor(self.block_blue, arm_tag="right", pre_grasp_dis=0.09)),
            lambda: self.move(self.move_by_displacement(arm_tag="right", z=0.15)),
            lambda: self.move(self.place_actor(self.block_blue, target_pose=self.block2_target_pose,
                                            arm_tag="right", functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align")),
            lambda: self.move(self.move_by_displacement(arm_tag="right", z=0.15, move_axis="arm")),
            lambda: self.move(self.back_to_origin(arm_tag="right")),

            # Black column
            lambda: self.move(self.grasp_actor(self.block_black, arm_tag="right", pre_grasp_dis=0.09)),
            lambda: self.move(self.move_by_displacement(arm_tag="right", z=0.15)),
            lambda: self.move(self.place_actor(self.block_black, target_pose=self.block3_target_pose,
                                            arm_tag="right", functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align")),
            lambda: self.move(self.move_by_displacement(arm_tag="right", z=0.15, move_axis="arm")),
            lambda: self.move(self.back_to_origin(arm_tag="right")),

            # Yellow on red
            lambda: self.move(self.grasp_actor(self.block_yellow, arm_tag="left", pre_grasp_dis=0.09)),
            lambda: self.move(self.move_by_displacement(arm_tag="left", z=0.15)),
            lambda: self.move(self.place_actor(self.block_yellow, target_pose=self.block_red.get_functional_point(1),
                                            arm_tag="left", functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align")),
            lambda: self.move(self.move_by_displacement(arm_tag="left", z=0.15, move_axis="arm")),
            lambda: self.move(self.back_to_origin(arm_tag="left")),

            # Green on blue
            lambda: self.move(self.grasp_actor(self.block_green, arm_tag="left", pre_grasp_dis=0.09)),
            lambda: self.move(self.move_by_displacement(arm_tag="left", z=0.15)),
            lambda: self.move(self.place_actor(self.block_green, target_pose=self.block_blue.get_functional_point(1),
                                            arm_tag="left", functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align")),
            lambda: self.move(self.move_by_displacement(arm_tag="left", z=0.15, move_axis="arm")),
            lambda: self.move(self.back_to_origin(arm_tag="left")),

            # White on black
            lambda: self.move(self.grasp_actor(self.block_white, arm_tag="right", pre_grasp_dis=0.09)),
            lambda: self.move(self.move_by_displacement(arm_tag="right", z=0.15)),
            lambda: self.move(self.place_actor(self.block_white, target_pose=self.block_black.get_functional_point(1),
                                            arm_tag="right", functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align")),
            lambda: self.move(self.move_by_displacement(arm_tag="right", z=0.15, move_axis="arm")),
            lambda: self.move(self.back_to_origin(arm_tag="right")),
        ]

        if self.playtestcount < len(actions):
            actions[self.playtestcount]()  # 执行一步
        else:
            print("Test completed. No more actions to perform.")

        self.playtestcount += 1

    def get_assistantInfo(self):
        assistant_functional_points = {
            "red_block": self.block_red.get_functional_point(1),
            "yellow_block": self.block_yellow.get_functional_point(1),
            "blue_block": self.block_blue.get_functional_point(1),
            "green_block": self.block_green.get_functional_point(1),
            "black_block": self.block_black.get_functional_point(1),
            "white_block": self.block_white.get_functional_point(1),
        }
        return(
            "This is a dual-arm task. You must use both arms (left and right) to finish the task. There are six blocks on the table. You should place six blocks in 3 columns(red and yellow in a column, blue and green in a column, white and black in a column). A column means two blocks are aligned in z axis and stacked. You should pay attention to make the smaller one on the larger one. And then arrange there columns in order from left to right, following the rule: (red and yellow blocks column) → (blue and green blocks column) → (white and black blocks column).\n"
"The main steps of this task: 1. judge the size of blocks from largest to smallest 2. grasp each block with the nearest arm 3. raise the arm slightly after grasping to avoid collision 4. place the block at the correct target position 5. always make the arm back to origin before using the other arm\n"
"In parameter (actor), you must output red_block, green_block, blue_block, yellow_block, white_block, black_block to represent blocks in different colors.\n"
f"!!!! Now {self.block1_target_pose} is the far left position, {self.block2_target_pose} is the middle position and {self.block3_target_pose} is the far right position. These are the three target poses of blocks from left to right. You can use them *directly* in the target_pose of place_actor() function. It's 7-dim. And the functional point of different blocks are as followed, you will use it when you try to place other blocks on them. The functional pointss of different colors: {assistant_functional_points}\n"
f"The size map of blocks in different colors is {self.halfsize_map}. You should use this map to judge block sizes.\n"
"You must raise the robot arm a certain distance to move the object or execute place command, otherwise it may hit the table. But you should not raise it too high, otherwise the object may fall off or the raise action will fail.\n"
"You can use the action 'back_to_origin' to return the arm to the origin position.\n"
"!!! Besides, if you don't need an arm right now, *please make it back to origin* in dual-arm tasks. Otherwise it may block the other arm.\n"
"NOTE: In parameters of PLACE_ACTOR() function, you must specify functional_point_id=0, constrain='align', else the place action will fail. You can also set pre_dis=0.09 and dis=0.02 to make it easier to succeed. In GRASP_ACTOR() function, you are recommended to set pre_grasp_dis=0.09.\n"
"And from your perspective: the front (inside) is +y, the right is +x, and the top is +z. The target pose is 7-dim, the first three are (x, y, z).\n"
        )

    def check_success(self):
        eps_stack = [0.025, 0.025, 0.012]
        eps_col = 0.05

        columns = [
            [self.block_red, self.block_yellow],
            [self.block_blue, self.block_green],
            [self.block_black, self.block_white]
        ]

        # 检查堆叠
        success_stack = True
        for col in columns:
            bottom_pose = col[0].get_pose().p
            top_pose = col[1].get_pose().p
            if not (np.all(abs(top_pose[:2] - bottom_pose[:2]) < eps_stack[:2]) and
                    abs(top_pose[2] - (bottom_pose[2] + 2 * col[0].halfsize)) < eps_stack[2]):
                success_stack = False
                break

        # 检查列顺序
        left_x = columns[0][0].get_pose().p[0]
        middle_x = columns[1][0].get_pose().p[0]
        right_x = columns[2][0].get_pose().p[0]
        success_order = (left_x < middle_x - eps_col) and (middle_x < right_x - eps_col)

        # 检查 gripper
        success_gripper = self.is_left_gripper_open() and self.is_right_gripper_open()

        return success_stack and success_order and success_gripper

    def take_action_by_dict(self,action_object):
        try:
            parameters = action_object.get("parameters", {})
            arm_tag = parameters.get("arm_tag","")
            arm_tag = arm_tag.lower()
            if arm_tag not in ["left", "right"]:
                    # raise ValueError(f"Invalid arm tag: {arm_tag}. Must be 'left' or 'right'.")
                if "both" in arm_tag and action_object["action_name"] == "back_to_origin":
                    self.move(self.back_to_origin(arm_tag="left"),self.back_to_origin(arm_tag="right"))
                    return True
                elif action_object["action_name"] != "move_by_displacement":
                    print(f"Invalid arm tag: {arm_tag}. Must be 'left' or 'right' or 'both', 'both' only can be used in action MOVE_BY_DISPLACEMENT.")
                    return f"Invalid arm tag: {arm_tag}. Must be 'left' or 'right' or 'both', 'both' only can be used in action MOVE_BY_DISPLACEMENT."
            actor = parameters.get("actor", None)
            target_object = None
            if isinstance(actor, str):
                actor = actor.lower()
                if "red" in actor:
                    target_object = self.block_red
                elif "green" in actor:
                    target_object = self.block_green
                elif "blue" in actor:
                    target_object = self.block_blue
                elif "yellow" in actor:
                    target_object = self.block_yellow
                elif "black" in actor:
                    target_object = self.block_black
                elif "white" in actor:
                    target_object = self.block_white
                else:
                    print(f"Invalid actor: {actor}. Must be 'red_block', 'green_block', 'blue_block', 'yellow_block', 'white_block', 'black_block' or some object with color tag.")
                    return f"Invalid actor: {actor}. Must be 'red_block', 'green_block', 'blue_block', 'yellow_block', 'white_block', 'black_block' or some object with color tag."
            
            if action_object["action_name"] == "grasp_actor":
                # print(action)
                if target_object is None:
                    print(f"Action Failed: No target object specified for grasping: {arm_tag}. Please specify the target object in the parameters.")
                    return f"No target object specified for grasping: {arm_tag}. Please specify the target object in the parameters."
                pre_grasp_dis = parameters.get("pre_grasp_dis", 0.1)
                if isinstance(pre_grasp_dis,str):
                    pre_grasp_dis = float(pre_grasp_dis)
                grasp_dis = parameters.get("grasp_dis", 0.0)
                if isinstance(grasp_dis,str):
                    grasp_dis = float(grasp_dis)
                gripper_pos = parameters.get("gripper_pos", 0.0)
                if isinstance(gripper_pos,str):
                    gripper_pos = float(gripper_pos)
                contact_point_id = parameters.get("contact_point_id", None)
                if isinstance(contact_point_id,str):
                    contact_point_id = int(contact_point_id)
                try:
                    if arm_tag == "left":
                        if(target_object.get_pose().p[0]>0.1):
                            print(f"Action Failed: target {actor} is too far, left arm can not finish this 'grasp' action! Please use another arm!")
                            return f"Action Failed: target {actor} is too far, left arm can not finish this 'grasp' action! Please use another arm!"
                        self.move(self.grasp_actor(target_object,arm_tag=arm_tag,pre_grasp_dis=pre_grasp_dis,
                                        grasp_dis=grasp_dis, gripper_pos=gripper_pos,
                                        contact_point_id=contact_point_id))
                        # self.left_grasped = True
                    elif arm_tag == "right":
                        if(target_object.get_pose().p[0]<-0.1):
                            print(f"Action Failed: target {actor} is too far, right arm can not finish this 'grasp' action! Please use another arm!")
                            return f"Action Failed: target {actor} is too far, right arm can not finish this 'grasp' action! Please use another arm!"
                        self.move(self.grasp_actor(target_object,arm_tag=arm_tag,pre_grasp_dis=pre_grasp_dis,
                                        grasp_dis=grasp_dis, gripper_pos=gripper_pos,
                                        contact_point_id=contact_point_id))
                        # self.right_grasped = True
                    return True
                except AssertionError as e:
                    print("Action Failed! This action can't be planned!!!")
                    return "Action Failed! This action can't be planned!!!"
                except Exception as e:
                    print(f"Grasping failed for {arm_tag} arm: {e}")
                    return f"Grasping failed for {arm_tag} arm: {e}"
                except:
                    print('unknown error when doing grasping!')
                    return 'unknown error when doing grasping!'

            elif action_object["action_name"] == "place_actor":
                if target_object is None:
                    print(f"Action Failed: No target object specified for placing:{arm_tag}. Please specify the target object in the parameters.")
                    return f"No target object specified for placing:{arm_tag}. Please specify the target object in the parameters."
                target_pose = parameters.get("target_pose", None)
                try:
                    target_pose = ast.literal_eval(target_pose) if isinstance(target_pose, str) else target_pose
                    target_pose = [float(i) for i in target_pose] if target_pose else None
                    if(len(target_pose) != 7):
                        print(f"Invalid target_pose length: {len(target_pose)}. Must be a list or tuple of 7 numbers.")
                        return f"Invalid target_pose length: {len(target_pose)}. Must be a list or tuple of 7 numbers."
                except:
                    print(f"Invalid target_pose format: {target_pose}. Must be a list or tuple of numbers.")
                    return f"Invalid target_pose format: {target_pose}. Must be a list or tuple of numbers."
                
                functional_point_id = parameters.get("functional_point_id",None) #functional_point_id 默认调为9
                function_point_id = parameters.get("function_point_id",None)
                if(function_point_id!=None):
                    print("Wrong arg name: function_point_id. It may be functional_point_id. Replanning!!!")
                    return("Wrong arg name: function_point_id. It may be functional_point_id. Replanning!!!")
                if isinstance(functional_point_id,str):
                    functional_point_id = int(functional_point_id)
                try:
                    kwargs = parameters.get("kwargs", None)
                    if kwargs != None:
                        parameters.update(kwargs)
                except:
                    print("kwargs recognize failed!")
                    return("kwargs recognize failed!")
                
                pre_dis = parameters.get("pre_dis", 0.1)
                pre_dis = float(pre_dis) if isinstance(pre_dis, str) else pre_dis
                dis = parameters.get("dis", 0.02)
                dis = float(dis) if isinstance(dis, str) else dis
                is_open = parameters.get("is_open", True)
                align_axis = parameters.get("align_axis", None)
                actor_axis = parameters.get("actor_axis", [1, 0, 0])
                actor_axis_type = parameters.get("actor_axis_type", "actor")
                constrain = parameters.get("constrain", "auto")
                pre_dis_axis = parameters.get("pre_dis_axis", "grasp")

                try:
                    self.move(self.place_actor(target_object, arm_tag=arm_tag, target_pose=target_pose,
                                            functional_point_id=functional_point_id, pre_dis=pre_dis, dis=dis,
                                            is_open=is_open, align_axis=align_axis, actor_axis=actor_axis,
                                            actor_axis_type=actor_axis_type, constrain=constrain,
                                            pre_dis_axis=pre_dis_axis))
                    return True
                except Exception as e:
                    print(f"Placing failed for {arm_tag} arm: {e}")
                    return f"Placing failed for {arm_tag} arm: {e}"

            elif action_object["action_name"] == "move_by_displacement":
                # print(action)
                x = parameters.get("x", 0.0)
                y = parameters.get("y", 0.0)
                z = parameters.get("z", 0.0)
                try:
                    if(isinstance(x, str)):
                        x = float(x)
                    if(isinstance(y, str)):
                        y = float(y)
                    if(isinstance(z, str)):
                        z = float(z)
                except Exception as e:
                    print(f"Invalid value of x or y or z:{e}")
                    return f"Invalid value of x or y or z:{e}"
                quat = parameters.get("quat", None)
                move_axis = parameters.get("move_axis", "world")
                try:
                    if(arm_tag == "left"):
                            self.move(self.move_by_displacement(arm_tag="left", x=x, y=y, z=z, quat=quat,move_axis=move_axis))
                    elif(arm_tag == "right"):
                            self.move(self.move_by_displacement(arm_tag="right", x=x, y=y, z=z,quat=quat, move_axis=move_axis))
                    else:
                            self.move(self.move_by_displacement(arm_tag="left", x=x, y=y, z=z, quat=quat,move_axis=move_axis),
                                    self.move_by_displacement(arm_tag="right", x=x, y=y, z=z, quat=quat,move_axis=move_axis))
                    # print("move_up, now z:",self.roller.get_pose().p[2])
                    return True
                except Exception as e:
                    print(f"Moving by displacement failed for {arm_tag} arm: {e}")
                    return f"Moving by displacement failed for {arm_tag} arm: {e}"

                
            elif action_object["action_name"] == "move_to_pose":
                target_pose = parameters.get("target_pose", None)
                try:
                    target_pose = ast.literal_eval(target_pose) if isinstance(target_pose, str) else target_pose
                    target_pose = [float(i) for i in target_pose] if target_pose else None
                except:
                    print(f"Invalid target_pose format: {target_pose}. Must be a list or tuple of numbers.")
                    return f"Invalid target_pose format: {target_pose}. Must be a list or tuple of numbers."
                if(len(target_pose) != 7):
                    print(f"Invalid target_pose length: {len(target_pose)}. Must be a list or tuple of 7 numbers.")
                    return f"Invalid target_pose length: {len(target_pose)}. Must be a list or tuple of 7 numbers."
                try:
                    self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=target_pose))
                    return True
                except Exception as e:
                    print(f"Moving to pose failed for {arm_tag} arm: {e}")
                    return f"Moving to pose failed for {arm_tag} arm: {e}"
                
            elif action_object["action_name"] == "close_gripper":
                # print(action)
                pos = parameters.get("pos", 0.0)
                try:
                    self.move(self.close_gripper(arm_tag=arm_tag, pos=pos))
                    return True
                except Exception as e:
                    print(f"Closing gripper failed for {arm_tag} arm: {e}")
                    return f"Closing gripper failed for {arm_tag} arm: {e}"
            elif action_object["action_name"] == "open_gripper":
                # print(action)
                pos = parameters.get("pos", 1.0)
                try:
                    self.move(self.open_gripper(arm_tag=arm_tag, pos=pos))
                    if(arm_tag == "left"):
                        self.left_grasped = False
                    elif(arm_tag == "right"):
                        self.right_grasped = False
                    return True
                except Exception as e:
                    print(f"Opening gripper failed for {arm_tag} arm: {e}")
                    return f"Opening gripper failed for {arm_tag} arm: {e}"
            elif action_object["action_name"] == "back_to_origin":
                # print(action)
                try:
                    self.move(self.back_to_origin(arm_tag=arm_tag))
                    return True
                except Exception as e:
                    print(f"Returning to origin failed for {arm_tag} arm: {e}")
                    return f"Returning to origin failed for {arm_tag} arm: {e}"
            elif action_object["action_name"] == "get_arm_pose":
                # print(action)
                return self.get_arm_pose(arm_tag=arm_tag)
            else:
                print(f"Invalid action name: {action_object['action_name']}. Must be 'grasp_actor', 'place_actor', 'move_by_displacement', 'move_to_pose', 'close_gripper', 'open_gripper', 'back_to_origin' or 'get_arm_pose'.")
                return f"Invalid action name: {action_object['action_name']}. Must be 'grasp_actor', 'place_actor', 'move_by_displacement', 'move_to_pose', 'close_gripper', 'open_gripper', 'back_to_origin' or 'get_arm_pose'."
        except Exception as e:
            print("Error, please check the content: ",e)
            return("Error, please check the content: ",e)