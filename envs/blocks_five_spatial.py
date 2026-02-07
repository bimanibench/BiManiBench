from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
import ast

class blocks_five_spatial(Base_Task):
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
            # x_range = [-0.3, 0.3]
            # if name == "red":
            #     x_range = [-0.3, -0.15]   # 左半边
            # elif name == "blue":
            #     x_range = [0.15, 0.3]     # 右半边
            # else:
            #     # xlim = [-0.28, 0.28]    # 整个桌面
            #     # 左右两侧分布：x < -0.15 或 x > 0.15
            if np.random.rand() < 0.5:
                    x_range = [-0.28, -0.02]  # 左侧
            else:
                    x_range = [0.02, 0.28]   # 右侧

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
            return("")
        
    def evaluate_spatial(self,results_object):
        try:
            rgb_result = ['', '', '','', '']  # red, green, blue, yellow, black

            def assign_arm(item):
                obj = item["object"].lower()
                arm = item["use_arm"].lower()
                if "red" in obj:
                    rgb_result[0] = arm
                elif "green" in obj:
                    rgb_result[1] = arm
                elif "blue" in obj:
                    rgb_result[2] = arm
                elif "yellow" in obj:
                    rgb_result[3] = arm
                elif "black" in obj:
                    rgb_result[4] = arm
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
            blocks = [self.block_red, self.block_green, self.block_blue, self.block_yellow, self.block_black]
            scoreList = [0,0,0,0,0]
            total_score = 0
            sigma = 0.05 / 2  # 控制高斯平滑范围（越小过渡越快）

            for i in range(5):
                pose_x = blocks[i].get_pose().p[0]
                arm = rgb_result[i]

                true_arm = "left" if pose_x < 0 else "right"

                if arm == true_arm:
                    scoreList[i] = 100.0
                else:
                    dist = abs(pose_x)
                    smooth_weight = math.exp(- (dist ** 2) / (2 * sigma ** 2))
                    scoreList[i] = 100.0* smooth_weight

            total_score = sum(scoreList)/5
            return total_score
        
        except Exception as e:
            print("Error, please check the content: ",e)
            return("Error, please check the content: ",e)