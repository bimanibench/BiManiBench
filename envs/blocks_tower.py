from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
import random
import ast

class blocks_tower(Base_Task):
    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)
        self.playtestcount = 0
        self.vic_mode = False
    def load_actors(self):
    # 颜色
        color_map = {
            "red": (1, 0, 0),
            "yellow": (1, 1, 0),
            "blue": (0, 0, 1),
            "green": (0, 1, 0),
        }
        self.block_colors = list(color_map.keys())

        # 半尺寸（可按需改为随机，但要保证不同）
        self.halfsize_map = {
            "red": 0.032,
            "yellow": 0.024,
            "blue": 0.028,
            "green": 0.02,
        }

        # 成对叠放：两列（列1、列2），每列底块+顶块
        # 你也可以把配对改成别的组合
        pair_columns = [
            ("red", "yellow"),  # 底: red，顶: yellow
            ("blue", "green"),  # 底: blue，顶: green
        ]

        self.blocks = {}

        # 选两列的 x 位置：左右两侧，保证间隔足够大，避免一开始就接近目标
        x_slots = [np.random.uniform(-0.26, -0.20), np.random.uniform(0.20, 0.26)]
        np.random.shuffle(x_slots)  # 随机决定哪一对在左/右

        # 同一条 y 基准线，便于稳定叠放和抓取
        y_line = np.random.uniform(-0.05, 0.1)

        # 桌面基准高度（与其他任务保持一致）
        table_z = 0.74 + self.table_z_bias

        # 创建两列、每列两个方块（底块先创建，再创建顶块）
        for col_idx, (bottom_name, top_name) in enumerate(pair_columns):
            x = x_slots[col_idx]

            # 底块参数
            hb = self.halfsize_map[bottom_name]
            bottom_z = table_z + hb  # 底块中心高度 = 桌面 + 底块半高

            bottom_pose = rand_pose(
                xlim=[x, x],
                ylim=[y_line, y_line],
                zlim=[bottom_z, bottom_z],
                qpos=[1, 0, 0, 0],      # 无倾斜，保证稳定
                rotate_rand=False       # 只要稳定，不做滚转/俯仰/偏航随机
            )
            bottom_block = create_box(
                scene=self,
                pose=bottom_pose,
                half_size=(hb, hb, hb),
                color=color_map[bottom_name],
                name=bottom_name
            )
            bottom_block.halfsize = hb
            self.blocks[bottom_name] = bottom_block
            self.add_prohibit_area(bottom_block, padding=0.05)

            # 顶块参数
            ht = self.halfsize_map[top_name]
            # 顶块中心高度 = 底块中心 + (底块半高 + 顶块半高)
            top_z = bottom_z + hb + ht

            top_pose = rand_pose(
                xlim=[x, x],
                ylim=[y_line, y_line],
                zlim=[top_z, top_z],
                qpos=[1, 0, 0, 0],
                rotate_rand=False
            )
            top_block = create_box(
                scene=self,
                pose=top_pose,
                half_size=(ht, ht, ht),
                color=color_map[top_name],
                name=top_name
            )
            top_block.halfsize = ht
            self.blocks[top_name] = top_block
            self.add_prohibit_area(top_block, padding=0.05)

        # 保存引用（按名称）
        self.block_red = self.blocks["red"]
        self.block_yellow = self.blocks["yellow"]
        self.block_blue = self.blocks["blue"]
        self.block_green = self.blocks["green"]

        y_center = np.random.uniform(-0.2, -0.15)  # y 可随机一点
        # 左边缓存位置
        self.left_cache_pose = [
            -0.15,               # x 左边
            y_center,            # 同一 y
            0.74 + self.table_z_bias + 0.01,  # 放在桌面上，稍微抬一点
            0, 1, 0, 0
        ]

        # 右边缓存位置
        self.right_cache_pose = [
            0.15,                # x 右边
            y_center,
            0.74 + self.table_z_bias + 0.01,
            0, 1, 0, 0
        ]
        self.middle_pose = [0,y_center,0.75 + self.table_z_bias, 0, 1, 0, 0]


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
            # yellow 
            lambda: self.move(self.grasp_actor(self.block_yellow, arm_tag="left", grasp_dis=0.03)),
            lambda: self.move(self.move_by_displacement(arm_tag="left", z=0.15)),
            lambda: self.move(self.place_actor(self.block_yellow, target_pose=self.left_cache_pose,
                                            arm_tag="left", functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align")),
            lambda: self.move(self.move_by_displacement(arm_tag="left", z=0.15, move_axis="arm")),
            lambda: self.move(self.back_to_origin(arm_tag="left")),
            # green
            lambda: self.move(self.grasp_actor(self.block_green, arm_tag="right", grasp_dis=0.03)),
            lambda: self.move(self.move_by_displacement(arm_tag="right", z=0.15)),
            lambda: self.move(self.place_actor(self.block_green, target_pose=self.right_cache_pose,
                                            arm_tag="right", functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align")),
            lambda: self.move(self.move_by_displacement(arm_tag="right", z=0.15, move_axis="arm")),
            lambda: self.move(self.back_to_origin(arm_tag="right")),

            # red
            lambda: self.move(self.grasp_actor(self.block_red, arm_tag="left", grasp_dis=0.03)),
            lambda: self.move(self.move_by_displacement(arm_tag="left", z=0.15)),
            lambda: self.move(self.place_actor(self.block_red, target_pose=self.middle_pose,
                                            arm_tag="left", functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align")),
            lambda: self.move(self.move_by_displacement(arm_tag="left", z=0.15, move_axis="arm")),
            lambda: self.move(self.back_to_origin(arm_tag="left")),

            # blue
            lambda: self.move(self.grasp_actor(self.block_blue, arm_tag="right", grasp_dis=0.03)),
            lambda: self.move(self.move_by_displacement(arm_tag="right", z=0.15)),
            lambda: self.move(self.place_actor(self.block_blue, target_pose=self.block_red.get_functional_point(1),
                                            arm_tag="right", functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align")),
            lambda: self.move(self.move_by_displacement(arm_tag="right", z=0.15, move_axis="arm")),
            lambda: self.move(self.back_to_origin(arm_tag="right")),

            # yellow
            lambda: self.move(self.grasp_actor(self.block_yellow, arm_tag="left", grasp_dis=0.03)),
            lambda: self.move(self.move_by_displacement(arm_tag="left", z=0.15)),
            lambda: self.move(self.place_actor(self.block_yellow, target_pose=self.block_blue.get_functional_point(1),
                                            arm_tag="left", functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align")),
            lambda: self.move(self.move_by_displacement(arm_tag="left", z=0.15, move_axis="arm")),
            lambda: self.move(self.back_to_origin(arm_tag="left")),

            # green
            lambda: self.move(self.grasp_actor(self.block_green, arm_tag="right", grasp_dis=0.03)),
            lambda: self.move(self.move_by_displacement(arm_tag="right", z=0.15)),
            lambda: self.move(self.place_actor(self.block_green, target_pose=self.block_yellow.get_functional_point(1),
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
            return("This is a two arm task. You must use both arms(left and right) to finish the task. There are four blocks in two piles on table. You should place them in a pile(tower) on the center of table, from bottom to top, largest to smallest.\n"
                   "The main steps of this task: 1. judge the order from largest to smallest and this the order to place blocks 2. grasp blocks with nearest arm. 3. raise your hand in case of collision with other objects 4. place one block to ideal position, raise your arm in case of collision and make arm back to origin, and then place another blocks to ideal positions."
                   "In parameter (actor), you must output red_block, green_block or blue_block to represent blocks in different colors. \n"
                   f"!!!! Now {self.middle_pose} is the center target pose for placing final block tower. And there are two temporary cache postion in two side for temporary placement of upper blocks. Left one is {self.left_cache_pose} and right one is {self.right_cache_pose}. Note: Left one can only be used with left arm and right one can only be used with right arm, else the place action will fail and env can not respond any more. Besides, when you plan to place some blocks on a bottom block, you should set target_pose = functional_point of this bottom block. The functional point of red block is {self.block_red.get_functional_point(1)}. The functional point of blue block is {self.block_blue.get_functional_point(1)}. The functional point of yellow block is {self.block_yellow.get_functional_point(1)}. The functional point of green block is {self.block_green.get_functional_point(1)}. You can use them *directly* in the target_pose of place_actor() function. It's 7-dim. The smallest one should be top and the largest one should be bottom. Else the blocks tower is not stable. You should judge their size with you observation and place them to proper position. And you should make proper plan and make good use of cache position on two sides of table.\n"
                   f"The size of red, yellow, green, blue blocks are {self.halfsize_map}. You should place blocks in a blocks tower in order of size from largest to smallest, from bottom to top."
                   "You must raise the robot arm a certain distance to move the object or execute place command, otherwise it may hit something if it moves directly close to the table surface. But you should not raise the arm too high, otherwise the object may fall off or the raise action will fail. \n You can use the action 'back_to_origin' to return the arm to the origin position. \n !!! Besides, if you don't need arm right now, *please make arm back to origin* in dual arm tasks. Else it may block actions of another arm. \n !!!"
                   "NOTE: In this task you should try to grasp the top block and place it to temporary cache position before grasping the bottom block. And don't directly make arm back_to_origin when you finished placing. You should raise arm a little first and then make arm back_to_origin. Otherwise the arm will hit something when back_to_origin if you don't raise the arm a little first. And there is a problem: if you move some blocks in an output, the functional point of the moved block will be different(Assistant info can only provide the information about last observation. For example, you try to move the red block to center and then move other block on red block in your this plan. But when you have moved the red block, the functional point of red block will change and the assistant info I provide you is out-dated.) So if you have moved the block and will use the functional point information in this same action series, you should not output PLACE_ACTOR() function in your output, stop here and wait next more information. I will provide you with the latest observation and assistant info soon. Don't be hurry. And when you place or grasp object, pay atttention to the left arm can only be used to grasp object on the left part of table and place to left part of table(x<=0.05, the first dim in target_pose) and the right arm can only be used to grasp object on right part of the table and place to right part of table(x>=-0.05, the first dim in target_pose). And when you place the smallest one to the top, you should raise your arm high enough(maybe about 0.2m) and then place to the top. Otherwise the movement of arm will collide with the blocks tower. And in parameters of PLACE_ACTOR() function, you must specify the functional_point_id = 0 and constrain='align', else the place action will be failed!!! You can also set pre_dis=0.09 and dis=0.02 to make it easier to success in PLACE_ACTOR() function. And you are recommended to set grasp_dis=0.03 in parameters of GRASP_ACTOR() function to make it easier to success.\n"
                   "And from your perspective, the front(inside) is the positive direction of y, the right is the positive direction of x, and the top is the positive direction of z. The target pose is 7-dim, first three is x, y, z.\n"
                   "This task is complex. Make sure your output is in the correct format. Your output must be a recognizable json format. You must a json object(dict in python) every time. If you output more than a json object, there will be error and no action will be executed.\n")
    # def play_test(self):
        # input("Enter Press to continue.")
    def check_success(self):
        block_pose_red = self.block_red.get_pose().p
        block_pose_yellow = self.block_yellow.get_pose().p
        block_pose_green = self.block_green.get_pose().p
        block_pose_blue = self.block_blue.get_pose().p
        # blocks_pose_lst = [block_pose_red[2],block_pose_yellow[2],block_pose_green[2],block_pose_blue[2]]
        
        # self.sorted_indices[1]
        eps = [0.05, 0.05]
        return (np.all(abs(block_pose_red[:2] - block_pose_blue[:2]) < eps) and np.all(abs(block_pose_blue[:2] - block_pose_yellow[:2]) < eps) and np.all(abs(block_pose_yellow[:2] - block_pose_green[:2]) < eps) and abs(block_pose_red[0])<0.03 and self.is_left_gripper_open() and self.is_right_gripper_open())

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
                    if isinstance(pos,str):
                        pos = float(pos)
                    self.move(self.close_gripper(arm_tag=arm_tag, pos=pos))
                    return True
                except Exception as e:
                    print(f"Closing gripper failed for {arm_tag} arm: {e}")
                    return f"Closing gripper failed for {arm_tag} arm: {e}"
            elif action_object["action_name"] == "open_gripper":
                # print(action)
                pos = parameters.get("pos", 1.0)
                try:
                    if isinstance(pos,str):
                        pos = float(pos)
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