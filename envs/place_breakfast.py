from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
import random
import ast
import glob

class place_breakfast(Base_Task):
    def setup_demo(self, **kwags):
        self.test_count = 0
        super()._init_task_env_(**kwags)

    def load_actors(self):

        def get_available_model_ids(modelname):
            asset_path = os.path.join("assets/objects", modelname)
            json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))
            available_ids = []
            for file in json_files:
                base = os.path.basename(file)
                try:
                    idx = int(base.replace("model_data", "").replace(".json", ""))
                    available_ids.append(idx)
                except ValueError:
                    continue
            return available_ids

        # 1. 物体类别池
        utensil_candidates = ["002_bowl", "003_plate", "021_cup", "039_mug"]
        food_candidates = ["005_french-fries", "006_hamburg", "035_apple", "075_bread"]
        seasoning_candidates = ["065_soy-sauce", "066_vinegar", "105_sauce-can"]

        # 2. 随机选出一个物体
        self.utensil_name = random.choice(utensil_candidates)
        self.food_name = random.choice(food_candidates)
        self.seasoning_name = random.choice(seasoning_candidates)
        print("objects: ",self.utensil_name,self.food_name,self.seasoning_name)
        # 3. 工具函数
        def random_pose(xlim, ylim, z=0.74, qpos=[1, 0, 0, 0]):
            return rand_pose(
                xlim=xlim,
                ylim=ylim,
                qpos=qpos,
                rotate_rand=True,
                rotate_lim=[0, 0, np.pi / 6],
            )

        # 4. 创建 utensil
        available_model_ids_utensil = get_available_model_ids(self.utensil_name)
        self.selected_model_id_utensil = np.random.choice(available_model_ids_utensil)
        self.utensil = create_actor(
            scene=self,
            pose=random_pose([-0.3, -0.2], [-0.15, -0.05]),
            modelname=self.utensil_name,
            convex=True,
            model_id=self.selected_model_id_utensil,
        )
        self.utensil.set_mass(0.02)

        # 5. 创建 food（薯条需要特殊处理）
        available_model_ids_food = get_available_model_ids(self.food_name)
        self.selected_model_id_food = np.random.choice(available_model_ids_food)

        if self.food_name == "005_french-fries":
            # 主要让薯条平躺，但加一点点随机扰动，避免刚体异常
            rand_pos_food = rand_pose(
                xlim=[0.2, 0.3],
                ylim=[-0.15, -0.07],
                qpos=[1.0, 0.0, 0.0, 0.0],   # 基本朝向是平躺
                rotate_rand=True,             # 开启随机
                rotate_lim=[0, 0, np.pi/18],  # 只在 z 轴 ±10° 内微调，仍然看起来是平躺
            )
            self.food = create_actor(
                scene=self,
                pose=rand_pos_food,
                modelname=self.food_name,
                convex=True,
                model_id=self.selected_model_id_food,
            )
        else:
            self.food = create_actor(
                scene=self,
                pose=random_pose([0.25, 0.35], [-0.15, -0.05]),  # 食物往右边分开一点
                modelname=self.food_name,
                convex=True,
                model_id=self.selected_model_id_food,
            )
        self.food.set_mass(0.02)

        # 6. 创建 seasoning
        available_model_ids_seasoning = get_available_model_ids(self.seasoning_name)
        self.selected_model_id_seasoning = np.random.choice(available_model_ids_seasoning)
        self.seasoning = create_actor(
            scene=self,
            pose=random_pose([-0.05, 0.05], [0.15, 0.25]),
            modelname=self.seasoning_name,
            convex=True,
            model_id=self.selected_model_id_seasoning,
        )
        self.seasoning.set_mass(0.02)

        # 7. 添加禁止区，防止重叠
        self.add_prohibit_area(self.utensil, padding=0.1)
        self.add_prohibit_area(self.food, padding=0.1)
        self.add_prohibit_area(self.seasoning, padding=0.1)

        # 8. 设置目标位置（左 → 中 → 右）
        y_pose = np.random.uniform(-0.2, -0.1)
        self.utensil_target_pose = [-0.1, y_pose, 0.74 + self.table_z_bias, 0, 1, 0, 0]
        self.food_target_pose = [0.0, y_pose, 0.74 + self.table_z_bias, 0, 1, 0, 0]
        self.seasoning_target_pose = [0.1, y_pose, 0.74 + self.table_z_bias, 0, 1, 0, 0]




    def play_once(self):
        self.last_gripper = None

        # 餐具 → 食物 → 调料
        arm_tag1 = self.pick_and_place_object(self.utensil, self.utensil_target_pose)
        arm_tag2 = self.pick_and_place_object(self.food, self.food_target_pose)
        arm_tag3 = self.pick_and_place_object(self.seasoning, self.seasoning_target_pose)

        self.info["info"] = {
            "{A}": "utensil",
            "{B}": "food",
            "{C}": "seasoning",
            "{a}": arm_tag1,
            "{b}": arm_tag2,
            "{c}": arm_tag3,
        }
        return self.info
    def play_test(self):
        input("Press Enter to continue.")
        self.take_action(action=[0,0,0],action_type="ee")
        # arm_tag = input("New action, Arm Tag(left or right): ")
        # action_name = input("Action Name(grasp, move, place or back): ")
        # if "grasp" in action_name:
        #     actor_name = input("target(utensil, food or seasoning): ")
        #     if "utensil" in actor_name:
        #         self.move(self.grasp_actor(actor=self.utensil,arm_tag=arm_tag,pre_grasp_dis=0.05))
        #     elif "food" in actor_name:
        #         self.move(self.grasp_actor(actor=self.food,arm_tag=arm_tag,pre_grasp_dis=0.05))
        #     elif "seasoning" in actor_name:
        #         self.move(self.grasp_actor(actor=self.seasoning,arm_tag=arm_tag,pre_grasp_dis=0.05))
        #     else:
        #         print(f"Error in actor_name({actor_name}).")
        #         return
        # elif "place" in action_name:
        #     actor_name = input("target(utensil, food or seasoning): ")
        #     if "utensil" in actor_name:
        #         self.move(self.place_actor(actor=self.utensil,arm_tag=arm_tag,target_pose=self.utensil_target_pose,constrain="free",
        #                      pre_dis=0.1,
        #                      pre_dis_axis='fp'))
        #     elif "food" in actor_name:
        #         self.move(self.place_actor(actor=self.food,arm_tag=arm_tag,target_pose=self.food_target_pose,constrain="free",
        #                      pre_dis=0.1,
        #                      pre_dis_axis='fp'))
        #     elif "seasoning" in actor_name:
        #         self.move(self.place_actor(actor=self.seasoning,arm_tag=arm_tag,target_pose=self.seasoning_target_pose,constrain="free",
        #                      pre_dis=0.1,
        #                      pre_dis_axis='fp'))
        #     else:
        #         print(f"Error in actor_name({actor_name}).")
        #         return
        # elif "move" in action_name:
        #     value = float(input("value in z: "))
        #     self.move(self.move_by_displacement(arm_tag=arm_tag,z=value))
        # elif "back" in action_name:
        #     self.move(self.back_to_origin(arm_tag=arm_tag))
        # else:
        #     print("Error in action_name.")
        #     return

        # if self.test_count == 0:
        #     self.last_gripper = None
        #     self.pick_and_place_object(self.utensil, self.utensil_target_pose)
        #     self.test_count += 1
        # elif self.test_count == 1:
        #     self.pick_and_place_object(self.food, self.food_target_pose)
        # elif self.test_count == 2:
        #     self.pick_and_place_object(self.seasoning, self.seasoning_target_pose)
        # else:
        #     input("Test Finished!")
        
    def pick_and_place_object(self, actor, target_pose=None):
        # 初始位置决定用左臂还是右臂
        obj_pose = actor.get_pose().p
        arm_tag = ArmTag("left" if obj_pose[0] < 0 else "right")

        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(
                self.grasp_actor(actor, arm_tag=arm_tag, pre_grasp_dis=0.09),
                self.back_to_origin(arm_tag=self.last_gripper),
            )
        else:
            self.move(self.grasp_actor(actor, arm_tag=arm_tag, pre_grasp_dis=0.09))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.move(
            self.place_actor(
                actor,
                target_pose=target_pose,
                arm_tag=arm_tag,
                pre_dis=0.09,
                dis=0.02,
                constrain="align",
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis="arm"))

        self.last_gripper = arm_tag
        return str(arm_tag)

    def check_success(self):
        utensil_pose = self.utensil.get_pose().p
        food_pose = self.food.get_pose().p
        seasoning_pose = self.seasoning.get_pose().p

        # 要求在一条直线上
        eps = [0.2, 0.08]
        aligned = (
            np.all(abs(utensil_pose[:2] - food_pose[:2]) < eps)
            and np.all(abs(food_pose[:2] - seasoning_pose[:2]) < eps)
        )

        # 顺序：utensil (最左) < food (中) < seasoning (最右)
        ordered = utensil_pose[0] < food_pose[0] < seasoning_pose[0]
        return aligned and ordered