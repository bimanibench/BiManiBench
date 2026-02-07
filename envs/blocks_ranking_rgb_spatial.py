from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
import ast

class blocks_ranking_rgb_spatial(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)
        self.playtestcount = 0
        self.vic_mode = False

    def load_actors(self):
        while True:
            block_pose_lst = []
            for i in range(3):
                if np.random.rand() < 0.5:
                    x_range = [-0.28, -0.02]  # 左侧
                else:
                    x_range = [0.02, 0.28]   # 右侧
                block_pose = rand_pose(
                    xlim=x_range,
                    ylim=[-0.08, 0.05],
                    zlim=[0.765],
                    qpos=[1, 0, 0, 0],
                    ylim_prop=True,
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )

                def check_block_pose(block_pose):
                    for j in range(len(block_pose_lst)):
                        if (np.sum(pow(block_pose.p[:2] - block_pose_lst[j].p[:2], 2)) < 0.01):
                            return False
                    return True

                while (abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2] - np.array([0, -0.1]), 2)) < 0.01
                       or not check_block_pose(block_pose)):
                    if np.random.rand() < 0.5:
                        x_range = [-0.28, -0.02]  # 左侧
                    else:
                        x_range = [0.02, 0.28]   # 右侧
                    block_pose = rand_pose(
                        xlim=x_range,
                        ylim=[-0.08, 0.05],
                        zlim=[0.765],
                        qpos=[1, 0, 0, 0],
                        ylim_prop=True,
                        rotate_rand=True,
                        rotate_lim=[0, 0, 0.75],
                    )
                block_pose_lst.append(deepcopy(block_pose))
            eps = [0.12, 0.03]
            block1_pose = block_pose_lst[0].p
            block2_pose = block_pose_lst[1].p
            block3_pose = block_pose_lst[2].p
            if (np.all(abs(block1_pose[:2] - block2_pose[:2]) < eps)
                    and np.all(abs(block2_pose[:2] - block3_pose[:2]) < eps) and block1_pose[0] < block2_pose[0]
                    and block2_pose[0] < block3_pose[0]):
                continue
            else:
                break

        size = np.random.uniform(0.015, 0.025)
        half_size = (size, size, size)
        self.block1 = create_box(
            scene=self,
            pose=block_pose_lst[0],
            half_size=half_size,
            color=(1, 0, 0),
            name="box",
        )
        self.block2 = create_box(
            scene=self,
            pose=block_pose_lst[1],
            half_size=half_size,
            color=(0, 1, 0),
            name="box",
        )
        self.block3 = create_box(
            scene=self,
            pose=block_pose_lst[2],
            half_size=half_size,
            color=(0, 0, 1),
            name="box",
        )

        self.add_prohibit_area(self.block1, padding=0.05)
        self.add_prohibit_area(self.block2, padding=0.05)
        self.add_prohibit_area(self.block3, padding=0.05)

        self.prohibited_area.append([-0.17, -0.22, 0.17, -0.12])

        # Generate random y position for all blocks
        y_pose = np.random.uniform(-0.2, -0.1)

        # Define target poses for each block with random x positions
        self.block1_target_pose = [
            np.random.uniform(-0.09, -0.08),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]
        self.block2_target_pose = [
            np.random.uniform(-0.01, 0.01),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]
        self.block3_target_pose = [
            np.random.uniform(0.08, 0.09),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]

    def play_once(self):
        # Initialize last gripper state
        self.last_gripper = None

        # Pick and place each block to their target positions
        arm_tag1 = self.pick_and_place_block(self.block1, self.block1_target_pose)
        arm_tag2 = self.pick_and_place_block(self.block2, self.block2_target_pose)
        arm_tag3 = self.pick_and_place_block(self.block3, self.block3_target_pose)

        # Store information about the blocks and which arms were used
        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": arm_tag1,
            "{b}": arm_tag2,
            "{c}": arm_tag3,
        }
        return self.info

    def play_test(self):
        arm_tag = ArmTag("right")
        if(self.playtestcount == 0):
            print("playtest start")
            self.move(
                self.grasp_actor(self.block1, arm_tag=arm_tag),
            )
            self.playtestcount += 1
        elif(self.playtestcount == 1):
            self.move(
                self.move_by_displacement(arm_tag=arm_tag, z=0.1),
            )
            self.playtestcount += 1
        elif(self.playtestcount == 2):
            self.move(
                self.place_actor(
                    #self.current_actor,
                    self.block1,
                    arm_tag=arm_tag,
                    target_pose=self.block1_target_pose,
                    constrain="align",
                    functional_point_id=0
                ))
            self.playtestcount += 1
        elif(self.playtestcount == 3):
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08))
            self.playtestcount += 1
        elif(self.playtestcount == 4):
            self.move(self.back_to_origin(arm_tag=arm_tag))
            self.playtestcount += 1
        else:
            print("playtest finished")
        return
    
    def pick_and_place_block(self, block, target_pose=None):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09, grasp_dis=0.01),  # arm_tag
                self.back_to_origin(arm_tag=arm_tag.opposite),  # arm_tag.opposite
            )
        else:
            self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))  # arm_tag

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))  # arm_tag

        self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align",
            ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis="arm"))  # arm_tag

        self.last_gripper = arm_tag
        return str(arm_tag)
    
    def get_assistantInfo(self):
            return("")
    
    def check_success(self):
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        block3_pose = self.block3.get_pose().p

        eps = [0.13, 0.03]

        return (np.all(abs(block1_pose[:2] - block2_pose[:2]) < eps)
                and np.all(abs(block2_pose[:2] - block3_pose[:2]) < eps) and block1_pose[0] < block2_pose[0]
                and block2_pose[0] < block3_pose[0] and self.is_left_gripper_open() and self.is_right_gripper_open())

    def evaluate_spatial(self,results_object):
        try:
            rgb_result = ['', '', '']

            def assign_arm(item):
                obj = item["object"].lower()
                arm = item["use_arm"].lower()
                if "red" in obj:
                    rgb_result[0] = arm
                elif "green" in obj:
                    rgb_result[1] = arm
                elif "blue" in obj:
                    rgb_result[2] = arm
                else:
                    print("unknown color:", obj)

            if isinstance(results_object, list):
                for item in results_object:
                    assign_arm(item)
            elif isinstance(results_object, dict):
                assign_arm(results_object)
            else:
                print("results_object 类型错误")

            # 评分部分
            blocks = [self.block1, self.block2, self.block3]
            scoreList = [0,0,0]
            total_score = 0
            sigma = 0.05 / 2  # 控制高斯平滑范围（越小过渡越快）

            for i in range(3):
                pose_x = blocks[i].get_pose().p[0]
                arm = rgb_result[i]

                true_arm = "left" if pose_x < 0 else "right"

                if arm == true_arm:
                    scoreList[i] = 100.0
                else:
                    dist = abs(pose_x)
                    smooth_weight = math.exp(- (dist ** 2) / (2 * sigma ** 2))
                    scoreList[i] = 100.0* smooth_weight

            total_score = sum(scoreList)/3
            return total_score
        
        except Exception as e:
            print("Error, please check the content: ",e)
            return("Error, please check the content: ",e)

