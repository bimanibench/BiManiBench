from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
import ast

class blocks_cross_shape(Base_Task):
    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)
        self.playtestcount = 0
        self.vic_mode = False

    def load_actors(self):
        # 五个方块颜色
        color_map = {
            "red": (1, 0, 0),
            "yellow": (1, 1, 0),
            "blue": (0, 0, 1),
            "green": (0, 1, 0),
            "black": (0, 0, 0),
        }
        self.block_colors = list(color_map.keys())

        # 随机生成初始位姿（靠近左右两侧）
        self.blocks = {}
        used_positions = []
        for name in self.block_colors:

            block_half_size = 0.025
             # 生成 pose 范围
            if name == "red":
                x_range = [-0.3, -0.15]   # 左半边
            elif name == "blue":
                x_range = [0.15, 0.3]     # 右半边
            else:
                # xlim = [-0.28, 0.28]    # 整个桌面
                # 左右两侧分布：x < -0.15 或 x > 0.15
                if np.random.rand() < 0.5:
                    x_range = [-0.3, -0.15]  # 左侧
                else:
                    x_range = [0.15, 0.3]   # 右侧

            pose = rand_pose(
                xlim=x_range,
                ylim=[-0.16, 0.1],
                zlim=[0.741 + block_half_size],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
            )
            it = 0
            while any(np.sum((pose.p[:2] - p[:2]) ** 2) < 0.01 for p in used_positions) and it < 100:
                pose = rand_pose(
                    xlim=x_range,
                    ylim=[-0.16, 0.1],
                    zlim=[0.741 + block_half_size],
                    qpos=[1, 0, 0, 0],
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )
                it+=1
            if it >= 100:
                raise UnStableError("Failed to sample non-colliding block positions.")
             # 记录已使用位置
            used_positions.append(pose.p)
            self.blocks[name] = create_box(
                scene=self,
                pose=pose,
                half_size=(block_half_size, block_half_size, block_half_size),
                color=color_map[name],
                name=name,
            )
            self.blocks[name].halfsize = block_half_size
            self.add_prohibit_area(self.blocks[name], padding=0.05)

        # 保存 block 引用
        self.block_red = self.blocks["red"]
        self.block_yellow = self.blocks["yellow"]
        self.block_blue = self.blocks["blue"]
        self.block_green = self.blocks["green"]
        self.block_black = self.blocks["black"]

        # --- 定义目标十字型 ---
        center_x, center_y = 0.0, -0.15  # 中心点
        base_z = 0.74 + self.table_z_bias

        self.target_poses = {
            "red": [center_x - 0.12, center_y, base_z, 0, 1, 0, 0],    # 左
            "blue": [center_x + 0.12, center_y, base_z, 0, 1, 0, 0],   # 右
            "black": [center_x, center_y, base_z, 0, 1, 0, 0],         # 中
            "green": [center_x, center_y + 0.12, base_z, 0, 1, 0, 0],  # 上
            "yellow": [center_x, center_y - 0.12, base_z, 0, 1, 0, 0], # 下
        }
    
    def get_assistantInfo(self):
            return("This is a two arm task. You must use both arms(left and right) to finish the task. there are five blocks on the table, the color of the blocks is random, move the blocks to the center of the table, and arrange them to a cross shape. The red block should be on the left. The black block should be on the center. The blue block should be on the right. The green should be inside(ahead). The yellow should be the pposite direction to green block, closest to you.\n"
                   "The main steps of this task: 1. grasp a block with nearest arm. 2. raise your hand in case of collision with other objects 3. determine the orientation(target_pose) among the five target coordinates to obtain the most ideal placement position for the block 4. place the block to the ideal position, raise your arm in case of collision and make arm back to origin, and then determine and place another blocks to ideal positions."
                   "In parameter (actor), you must output red_block, green_block, yellow_block, black_block, blue_block to represent blocks in different colors. \n"
                   f'!!!! Now there are five target_pose for you to determine: {self.target_poses["red"]}, {self.target_poses["yellow"]}, {self.target_poses["black"]}, {self.target_poses["blue"]}, {self.target_poses["green"]}. They are 7-dim and first 3-dim represents x, y, z.'
                   "And from your perspective, the front(inside) is the positive direction of y, the right is the positive direction of x, and the top is the positive direction of z. You can use them *directly* in the target_pose of place_actor() function. Imagine a cross with five points in your mind and determine the different orientation in this five points.\n"
                   "You must raise the robot arm a certain distance to move the object or execute place command, otherwise it may hit something if it moves directly close to the table surface. But you should not raise the arm too high, otherwise the object may fall off or the raise action will fail. \n You can use the action 'back_to_origin' to return the arm to the origin position. \n !!! Besides, if you don't need arm right now, *please make arm back to origin* in dual arm tasks. Else it may block actions of another arm. \n !!!"
                   "NOTE: Don't directly make arm back_to_origin when you finished placing. You should raise arm a little first and then make arm back_to_origin. Otherwise the arm will hit something when back_to_origin if you don't raise the arm a little first. And when you place or grasp object, pay atttention to the left arm can only be used to grasp object on the left part of table and place to left part of table(x<=0.05, the first dim in target_pose) and the right arm can only be used to grasp object on right part of the table and place to right part of table(x>=-0.05, the first dim in target_pose). And in parameters of PLACE_ACTOR() function, you must specify the functional_point_id = 0 and constrain='align', else the place action will be failed!!! You can also set pre_dis=0.09 and dis=0.02 to make it easier to success in PLACE_ACTOR() function. And you are recommended to set grasp_dis=0.03 in parameters of GRASP_ACTOR() function to make it easier to success.\n"
                   "And from your perspective, the front(inside) is the positive direction of y, the right is the positive direction of x, and the top is the positive direction of z. The target pose is 7-dim, first three is x, y, z.\n"
                   "This task is complex. Make sure your output is in the correct format. Your output must be a recognizable json format. You must a json object(dict in python) every time. If you output more than a json object, there will be error and no action will be executed.\n")
        
    def check_success(self):
        block_pose_red = self.block_red.get_pose().p
        block_pose_yellow = self.block_yellow.get_pose().p
        block_pose_blue = self.block_blue.get_pose().p
        block_pose_green = self.block_green.get_pose().p
        block_pose_black = self.block_black.get_pose().p
        eps = [0.02, 0.02]

        return (np.all(abs(block_pose_red[:2] - self.target_poses["red"][:2]) < eps)
                and np.all(abs(block_pose_yellow[:2] - self.target_poses["yellow"][:2]) < eps)
                and np.all(abs(block_pose_blue[:2] - self.target_poses["blue"][:2]) < eps)
                and np.all(abs(block_pose_green[:2] - self.target_poses["green"][:2]) < eps)
                and np.all(abs(block_pose_black[:2] - self.target_poses["black"][:2]) < eps)
                and self.is_left_gripper_open() and self.is_right_gripper_open())
    
    def play_test(self):
        input("Press Enter to start the test...")
        # 定义动作序列，每个元素是一个函数调用
        actions = [
            # red
            lambda: self.move(self.grasp_actor(self.block_red, arm_tag="left", grasp_dis=0.03)),
            lambda: self.move(self.move_by_displacement(arm_tag="left", z=0.15)),
            lambda: self.move(self.place_actor(self.block_red, target_pose=self.target_poses["red"],
                                            arm_tag="left", functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align")),
            lambda: self.move(self.move_by_displacement(arm_tag="left", z=0.15, move_axis="arm")),
            lambda: self.move(self.back_to_origin(arm_tag="left")),
            # black
            lambda: self.move(self.grasp_actor(self.block_black, arm_tag="right", grasp_dis=0.03)),
            lambda: self.move(self.move_by_displacement(arm_tag="right", z=0.15)),
            lambda: self.move(self.place_actor(self.block_black, target_pose=self.target_poses["black"],
                                            arm_tag="right", functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align")),
            lambda: self.move(self.move_by_displacement(arm_tag="right", z=0.15, move_axis="arm")),
            lambda: self.move(self.back_to_origin(arm_tag="right")),

            # yellow
            lambda: self.move(self.grasp_actor(self.block_yellow, arm_tag="left", grasp_dis=0.03)),
            lambda: self.move(self.move_by_displacement(arm_tag="left", z=0.15)),
            lambda: self.move(self.place_actor(self.block_yellow, target_pose=self.target_poses["yellow"],
                                            arm_tag="left", functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align")),
            lambda: self.move(self.move_by_displacement(arm_tag="left", z=0.15, move_axis="arm")),
            lambda: self.move(self.back_to_origin(arm_tag="left")),

            # blue
            lambda: self.move(self.grasp_actor(self.block_blue, arm_tag="right", grasp_dis=0.03)),
            lambda: self.move(self.move_by_displacement(arm_tag="right", z=0.15)),
            lambda: self.move(self.place_actor(self.block_blue, target_pose=self.target_poses["blue"],
                                            arm_tag="right", functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align")),
            lambda: self.move(self.move_by_displacement(arm_tag="right", z=0.15, move_axis="arm")),
            lambda: self.move(self.back_to_origin(arm_tag="right")),

            # green
            lambda: self.move(self.grasp_actor(self.block_green, arm_tag="left", grasp_dis=0.03)),
            lambda: self.move(self.move_by_displacement(arm_tag="left", z=0.15)),
            lambda: self.move(self.place_actor(self.block_green, target_pose=self.target_poses["green"],
                                            arm_tag="left", functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align")),
            lambda: self.move(self.move_by_displacement(arm_tag="left", z=0.15, move_axis="arm")),
            lambda: self.move(self.back_to_origin(arm_tag="left")),

        ]

        if self.playtestcount < len(actions):
            actions[self.playtestcount]()  # 执行一步
        else:
            print("Test completed. No more actions to perform.")

        self.playtestcount += 1

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
                else:
                    print(f"Invalid actor: {actor}. Must be 'red_block', 'green_block', 'blue_block', 'yellow_block', 'black_block' or some object with color tag.")
                    return f"Invalid actor: {actor}. Must be 'red_block', 'green_block', 'blue_block', 'yellow_block', 'black_block' or some object with color tag."
            
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
                    if(len(target_pose) != 7 and len(target_pose) != 3):
                        print(f"Invalid target_pose length: {len(target_pose)}. Must be a list or tuple of 7 or 3 numbers.")
                        return f"Invalid target_pose length: {len(target_pose)}. Must be a list or tuple of 7 or 3 numbers."
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