from ._base_task import Base_Task
from .utils import *
import sapien
import glob
import ast
from .utils.quatSolve import quat_mul,compute_grasp_quat
class place_bread_skillet_vla(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags, table_static=False)
        self.vic_mode = False
        self.initial_armpose = {
            "left": self.get_arm_pose(ArmTag("left")),
            "right": self.get_arm_pose(ArmTag("right")),
        }

    def get_left_right_arm_pose(self):
        left_armpose = self.get_arm_pose(arm_tag="left")
        right_armpose = self.get_arm_pose(arm_tag="right")
        left_armpose[:2] = [-0.1, -0.05]
        left_armpose[2] += 0.05
        left_armpose[3:] = [-0.707, 0, -0.707, 0]
        right_armpose[:2] = [0.1, -0.05]
        right_armpose[2] += 0.05
        right_armpose[3:] = [0, 0.707, 0, -0.707]
        return(left_armpose,right_armpose)

    def load_actors(self):
        # id_list = [0, 1, 3, 5, 6]
        id_list = [0, 1, 3, 6]
        self.bread_id = np.random.choice(id_list)
        rand_pos = rand_pose(
            xlim=[-0.28, 0.28],
            ylim=[-0.2, 0.05],
            qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 4, 0],
        )
        while abs(rand_pos.p[0]) < 0.2:
            rand_pos = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=[-0.2, 0.05],
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 4, 0],
            )
        self.bread = create_actor(
            self,
            pose=rand_pos,
            modelname="075_bread",
            model_id=self.bread_id,
            convex=True,
        )

        xlim = [0.15, 0.25] if rand_pos.p[0] < 0 else [-0.25, -0.15]
        self.model_id_list = [0, 1, 2, 3]
        self.skillet_id = np.random.choice(self.model_id_list)
        rand_pos = rand_pose(
            xlim=xlim,
            ylim=[-0.2, 0.05],
            qpos=[0, 0, 0.707, 0.707],
            rotate_rand=False,
            # rotate_lim=[0, np.pi / 6, 0],
        )
        self.skillet = create_actor(
            self,
            pose=rand_pos,
            modelname="106_skillet",
            model_id=self.skillet_id,
            convex=True,
        )

        self.bread.set_mass(0.001)
        self.skillet.set_mass(0.01)
        self.add_prohibit_area(self.bread, padding=0.03)
        self.add_prohibit_area(self.skillet, padding=0.05)

    def play_test(self):
        # Determine which arm to use based on skillet position (right if on positive x, left otherwise)
        # print("ee pose of arm:")
        # print(np.around(self.get_arm_pose(ArmTag("left")),3))
        # print(np.around(self.get_arm_pose(ArmTag("right")),3))
        # arm_tag = ArmTag("right" if self.skillet.get_pose().p[0] > 0 else "left")
        # target_pose = self.get_arm_pose(arm_tag=arm_tag)
        # if arm_tag == "left":
        #     # Set specific position and orientation for left arm
        #     target_pose[:2] = [-0.1, -0.05]
        #     target_pose[2] -= 0.05
        #     target_pose[3:] = [-0.707, 0, -0.707, 0]
        # else:
        #     # Set specific position and orientation for right arm
        #     target_pose[:2] = [0.1, -0.05]
        #     target_pose[2] -= 0.05
        #     target_pose[3:] = [0, 0.707, 0, -0.707]
        # print(f"Using '{arm_tag}' arm to manipulate skillet, target_pose of skillet: ",target_pose)
        # print("bread pose: ",self.bread.get_pose().p,self.bread.get_pose().q)
        # skillet_pos = self.skillet.get_pose().p
        # skillet_pos[1] -= 0.07
        # print("skillet pose: ",skillet_pos,self.skillet.get_pose().q)
        # print(f"skillet target pose: left:{[-0.15,-0.15, 0.95, 0.707, 0, 0.707, 0]} right:{[0.15,-0.15, 0.95, 0, -0.707, 0, 0.707]}")
        # print("target functional point of skillet: ",self.skillet.get_functional_point(0))
        # # print("hand quat: ",quat_mul(self.bread.get_pose().q,[0.5,-0.5,0.5,0.5]))
        # print("hand quat: ",compute_grasp_quat(self.bread.get_pose().q).tolist())
        left_arm_pose = np.around(self.get_arm_pose(ArmTag("left")),3)
        right_arm_pose = np.around(self.get_arm_pose(ArmTag("right")),3)
        bread_grasp_pose = np.around(self.bread.get_pose().p,3).tolist()
        bread_grasp_quat = compute_grasp_quat(self.bread.get_pose().q)
        skillet_grasp_pos = self.skillet.get_pose().p
        skillet_grasp_pos[1] -= 0.07
        skillet_grasp_quat = [0.5,-0.5,0.5,0.5]
        skillet_place_pose = [-0.15,-0.15, 0.95, 0.707, 0, 0.707, 0] if self.skillet.get_pose().p[0]<0 else [0.15,-0.15, 0.95, 0, -0.707, 0, 0.707]
        bread_place_pos = [0, -0.15, 1, 0.707, 0, 0.707, 0]
        print("bread_grasp_pose: ",bread_grasp_pose)
        print("bread_grasp_quat: ",bread_grasp_quat)
        print("skillet_grasp_pos: ",np.around(skillet_grasp_pos,3).tolist())
        print("skillet_grasp_quat: ",skillet_grasp_quat)
        print("skillet_place_pose: ",skillet_place_pose)
        print("bread_place_pos: ",bread_place_pos)
        final_action = input("Final action(16dim): ")
        final_action = ast.literal_eval(final_action) if isinstance(final_action, str) else final_action
        # final_action = self.get_arm_pose(ArmTag("left")) +[1] + self.get_arm_pose(ArmTag("right"))+[1]
        # final_action[0:3] = action_left
        # final_action[8:11] = action_right
        if(len(final_action)!=16):
            print(len(final_action))
            return
        else:
            print("final_action: ",final_action)
            self.take_action(action=final_action,action_type='ee')
        return False
    def play_once(self):
        # Determine which arm to use based on skillet position (right if on positive x, left otherwise)
        arm_tag = ArmTag("right" if self.skillet.get_pose().p[0] > 0 else "left")

        # Grasp the skillet and bread simultaneously with dual arms
        self.move(
            self.grasp_actor(self.skillet, arm_tag=arm_tag, pre_grasp_dis=0.07, gripper_pos=0),
            self.grasp_actor(self.bread, arm_tag=arm_tag.opposite, pre_grasp_dis=0.07, gripper_pos=0),
        )

        # Lift both arms
        self.move(
            self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"),
            self.move_by_displacement(arm_tag=arm_tag.opposite, z=0.1),
        )

        # Define a custom target pose for the skillet based on its side (left or right)
        target_pose = self.get_arm_pose(arm_tag=arm_tag)
        if arm_tag == "left":
            # Set specific position and orientation for left arm
            target_pose[:2] = [-0.1, -0.05]
            target_pose[2] -= 0.05
            target_pose[3:] = [-0.707, 0, -0.707, 0]
        else:
            # Set specific position and orientation for right arm
            target_pose[:2] = [0.1, -0.05]
            target_pose[2] -= 0.05
            target_pose[3:] = [0, 0.707, 0, -0.707]

        # Place the skillet to the defined target pose
        self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=target_pose))

        # Get the functional point of the skillet as placement target for the bread
        target_pose = self.skillet.get_functional_point(0)

        # Place the bread onto the skillet
        self.move(
            self.place_actor(
                self.bread,
                target_pose=target_pose,
                arm_tag=arm_tag.opposite,
                constrain="free",
                pre_dis=0.05,
                dis=0.05,
            ))

        self.info["info"] = {
            "{A}": f"106_skillet/base{self.skillet_id}",
            "{B}": f"075_bread/base{self.bread_id}",
            "{a}": str(arm_tag),
        }
        return self.info
    
    def get_assistantInfo(self):
        leftpose,rightpose = self.get_left_right_arm_pose()
        # left_quat = GRASP_DIRECTION_DIC["front_right"]
        # right_quat = GRASP_DIRECTION_DIC["front_left"] #相反，按照原本play_once()里面的
        return (
            "This is a two arm task. You must use both arms(left and right) to finish the task. You should grasp object with the nearest arm, else the grasp action will fail.\n"
            "The main steps of this task: 1. grasp the bread and skillet with the nearest arms. 2. raise arms in case of collision with other objects 3. move the arm grasping skillet to target pose. 4. place the bread to the target functional position on skillet\n"
            f"The target pose of skillet is {leftpose} when you are using left arm. Similarly, the target pose of skillet is {rightpose} when you are using right arm.\n"
            "In parameter (actor), you must output bread or skillet to represent the object. \n"
            f"!!!! Now {self.skillet.get_functional_point(0)} is the target functional point of bread on skillet. When you plan to place bread on it, you can use them *directly* in the target_pose of place_actor() function. It's 7-dim. \n"
            "You must raise the robot arm a certain distance to move the object or execute place command, otherwise it may hit something if it moves directly close to the table surface. But you should not raise the arm too high, otherwise the object may fall off or the raise action will fail. \n"
            "You can use the action 'back_to_origin' to return the arm to the origin position. \n"
            "!!! Besides, if you don't need arm right now, *please make arm back to origin* in dual arm tasks. Else it may block actions of another arm. \n"
            f" NOTE: In parameters of PLACE_ACTOR() function, you must pay attention to parameter setting. When you plan to place bread on skillet, you should set constrain='free', pre_dis=0.05 and dis=0.05 to make it easier to succeed. When you plan to raise the arm grasping the skillet, please set move_axis='arm' to make it easier to succeed. !!"
        )
    def get_assistantInfo_vla(self):
        left_arm_pose = np.around(self.get_arm_pose(ArmTag("left")),5)
        right_arm_pose = np.around(self.get_arm_pose(ArmTag("right")),5)
        bread_grasp_pos = np.around(self.bread.get_pose().p,3)
        bread_grasp_quat = compute_grasp_quat(self.bread.get_pose().q)
        skillet_grasp_pos = self.skillet.get_pose().p
        skillet_grasp_pos[1] -= 0.07
        skillet_grasp_quat = [0.5,-0.5,0.5,0.5]
        skillet_place_pose = [-0.15,-0.15, 0.95, 0.707, 0, 0.707, 0] if self.skillet.get_pose().p[0]<0 else [0.15,-0.15, 0.95, 0, -0.707, 0, 0.707]
        bread_place_pos = [0, -0.15, 1, 0.707, 0, 0.707, 0]
        # print("bread pose: ",self.bread.get_pose().p,self.bread.get_pose().q)
        # print("hand quat: ",compute_grasp_quat(self.bread.get_pose().q).tolist())

        return (
            f"This is a two arm task. You must use both arms(left and right) to finish the task. \n"
            "The main steps of this task: 1.move grippers above the object(use nearest arm to grasp object on the same side) 2.lower the gripper to prepare to grasp the objects 3.close the gripper to grasp the object 4.raise the arm and keep gripper close 5.move skillet to ideal place in ideal quat 6.move bread above the skillet and then release the gripper\n"
            f"Now the left arm pose is {left_arm_pose} and the right arm pose is {right_arm_pose}.\n"
            f"Now the positions of skillet(handle for grasp) is {np.round(skillet_grasp_pos,5)} and of bread is {bread_grasp_pos}. You should judge the position of two objects and use proper arm to grasp. But you can't directly use it because they're position of object, no the position of gripper. Since there is a certain distance between the end of the gripper and the center of the gripper (the gripper has a height), you need to add 0.144 to the object height (z), which should be the actual position of the gripper. When you try to grasp bread, you should set q or quat={np.round(bread_grasp_quat,3)}. When you try to grasp skillet, you should set quat=[0.5,-0.5,0.5,0.5]. However, when you place skillet to center(ideal position), you should use another quat. When you use left arm to place skillet to center, you should set 7-dim pose = [-0.15,-0.15, 0.95, 0.707, 0, 0.707, 0]. When you use right arm to place skillet to center, you should set 7-dim pose = [0.15,-0.15, 0.95, 0, -0.707, 0, 0.707]. This pose will help to place the skillet to center. You can directly use it (Don't forget to add 1-dim gripper state in action). When you have placed skillet to center, you should move bread above the skillet and release the gripper(you should lift bread first after grasping bread). The ideal pose for this is [0, -0.15, 1, 0.707, 0, 0.707, 0]. This is the final 7-dim pose for gripper with bread. You can directly use it (Don't forget to add 1-dim gripper state in action).\n"
            "!!!!!NOTE: Please keep gripper close after you grasp the skillet and place it to center. If you release the gripper, the skillet will fall and task will fail. You should keep gripper still closed until the bread is on skillet. And you can't directly move the bread after you grasp it. You must raise it high enough(z of gripper pos about 1m) and then move it, otherwise it may collide with skillet and task will fail.\n"
            # f"The target position(x,y) of object is {self.tray.get_functional_point(0)[:2]} and of right object is {self.tray.get_functional_point(1)[:2]}.\n"
            # f"NOTE: The initial arm pose of left arm is {np.round(self.initial_armpose['left'],5)} and of right arm is {np.round(self.initial_armpose['right'],5)}. You can use these poses to make arm back to origin if you don't need this arm. The original gripper state is open or 1.\n"
            "!!!!!!NOTE: There's a distance difference between the gripper's center position and its lowest point, meaning the gripper's height is approximately 0.144m. You can't lower the gripper too low. For example, if the object's height (z) is 0.756m, and you should output a pose with a lowest point of 0.9m, you need to add 0.144 and set z=0.9 to prevent the gripper from hitting the table and causing damage. In other words, when you output a pose with a z value of 1m, the lowest point is 0.838m,. This involves some calculations. In the action, you must calculate and output the final result in advance, rather than outputting the expression for me to calculate. Note that your output is directly submitted to the environment for interaction, so make sure your output conforms to the format requirements! If you output like 0.756+0.144, this will be illegal.\n"
            "Since there is a certain error in the End-Effector Pose Control mode (maybe several millimeters), please do not trust the information provided in the assistant info too much. If the observation shows obvious deviation, please move it in a more appropriate direction based on the position provided in the assistant info (for example, add 0.1 (m) to the positive direction of the y-axis).\n"
            "Don't output too many actions in an output. The position of object will change when your former actions are executed. Make plan based on the latest observation.\n"
            "NOTE: To avoid a fight between two robotic arms, please consider whether the two will collide when operating the robotic arms. If there is a possibility of collision, operate one robotic arm first, move it away after the operation, and then operate the other robotic arm."
        )
    def check_success(self):
        target_pose = self.skillet.get_functional_point(0)
        bread_pose = self.bread.get_pose().p
        # print(abs(target_pose[:2] - bread_pose[:2]), target_pose[2] - (0.76 + self.table_z_bias), bread_pose[2] - (0.76 + self.table_z_bias))
        return (np.all(abs(target_pose[:2] - bread_pose[:2]) < [0.035, 0.035])
                and target_pose[2] > 0.76 + self.table_z_bias and bread_pose[2] > 0.76 + self.table_z_bias)
    
    def take_action_by_dict_vla(self,action_object):
        def safe_eval(expr):
            """安全计算简单算术表达式（仅数字和+-*/.）"""
            try:
                return eval(expr, {"__builtins__": None}, {})
            except Exception:
                return float("nan")  # 出错时返回 NaN，也可以 raise

        def parse_action_sequence(obj):
            """
            统一解析 VLM 输出的动作序列:
            - 输入可能是字符串: "[0.1+0.2, 1+0.1+0.02, 0.5]"
            - 或者是列表: ["0.1+0.2", "1+0.1+0.02", 0.5]
            - 或者是混合: [0.1+0.2, "0.3+0.4"]
            """
            # 如果是字符串，尝试转成 Python 对象
            if isinstance(obj, str):
                try:
                    obj = ast.literal_eval(obj)
                except Exception:
                    # fallback：去掉方括号再 split
                    obj = [x.strip() for x in obj.strip("[]").split(",")]

            # 如果还是不是 list，强制转成 list
            if not isinstance(obj, (list, tuple)):
                obj = [obj]

            result = []
            for x in obj:
                if isinstance(x, (int, float)):
                    result.append(float(x))
                elif isinstance(x, str):
                    result.append(float(safe_eval(x)))
                else:
                    # 万一模型直接给了表达式对象（极少见情况）
                    result.append(float(safe_eval(str(x))))
            return result
        # print("action: ",action_object)
        # return "OK"
        try:
            # final_action = ast.literal_eval(action_object) if isinstance(action_object, str) else action_object
            final_action = parse_action_sequence(action_object)
            final_action = [float(i) for i in final_action]
            print("parsed action: ",final_action)
            if(len(final_action)!=16):
                print(f"Invalid action length: {len(final_action)}. Must be a list or tuple of 16 numbers.")
                return f"Invalid action length: {len(final_action)}. Must be a list or tuple of 16 numbers."
            # print("execute: ",final_action)
            self.take_action(action=final_action,action_type='ee')
            return True
        except Exception as e:
            print(f"Error when parsing action: {e}")
            return f"Error when parsing action: {e}"
            # print("Unknown error when parsing action!")
            # return "Unknown error when parsing action!"

    def take_action_by_dict(self,action_object):
        try:
            parameters = action_object.get("parameters", {})
            arm_tag = parameters.get("arm_tag","")
            arm_tag = arm_tag.lower()
            if arm_tag not in ["left", "right"]:
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
                if "bread" in actor:
                    target_object = self.bread
                elif "skillet" in actor:
                    target_object = self.skillet
                else:
                    print(f"Invalid actor: {actor}. Must be 'skillet' or 'bread'.")
                    return f"Invalid actor: {actor}. Must be 'skillet' or 'bread'."
            
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
                except Exception as e:
                    print(f"Grasping failed for {arm_tag} arm: {e}")
                    return f"Grasping failed for {arm_tag} arm: {e}"

            elif action_object["action_name"] == "place_actor":
                if target_object is None:
                    print(f"Action Failed: No target object specified for placing:{arm_tag}. Please specify the target object in the parameters.")
                    return f"No target object specified for placing:{arm_tag}. Please specify the target object in the parameters."
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
                functional_point_id = parameters.get("functional_point_id",None) #functional_point_id 默认调为9
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
                # print(target_pose,pre_dis,dis,constrain)
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
                if(isinstance(x, str)):
                    x = float(x)
                if(isinstance(y, str)):
                    y = float(y)
                if(isinstance(z, str)):
                    z = float(z)
                quat = parameters.get("quat", None)
                move_axis = parameters.get("move_axis", "world")
                try:
                    self.move(self.move_by_displacement(arm_tag=arm_tag, x=x, y=y, z=z, quat=quat,move_axis=move_axis))
                    return True
                except Exception as e:
                    print(f"Moving by displacement failed for {arm_tag} arm: {e}")
                    return f"Moving by displacement failed for {arm_tag} arm: {e}"

                
            elif action_object["action_name"] == "move_to_pose":
                # print(action)
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
