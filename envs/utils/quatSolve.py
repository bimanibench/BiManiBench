import numpy as np

def quat_mul(q1, q2):
    """四元数乘法 (w,x,y,z)"""
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conj(q):
    """四元数共轭"""
    w,x,y,z = q
    return np.array([w, -x, -y, -z])

def quat_rotate(q, v):
    """用四元数 q 旋转向量 v"""
    qv = np.array([0, *v])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]

def quat_from_zrot(deg):
    """绕 z 轴旋转 deg° 的四元数"""
    rad = np.deg2rad(deg)
    return np.array([np.cos(rad/2), 0, 0, np.sin(rad/2)])

# ===== 主流程 =====
def get_final_quat(q):
    # 1. 初始四元数
    # q = np.array([0.530, 0.530, 0.468, 0.468])
    q = np.array(q)
    # 2. 求旋转后方向
    v = quat_rotate(q, np.array([1,0,0]))
    v[2] = 0
    v /= np.linalg.norm(v)

    # 3. 求左侧垂直向量
    u = np.array([-v[1], v[0], 0])
    u /= np.linalg.norm(u)

    # 4. 计算夹角 theta（单位：度）
    theta = np.rad2deg(np.arctan2(u[1], u[0]))
    print("q: ",q,". theta before adjust:", theta)
    if theta < 0:
        theta += 360
    # print(f"θ = {theta:.3f}°")
    # 5. 固定旋转 [0.5, -0.5, 0.5, 0.5]
    q_fixed = np.array([0.5, -0.5, 0.5, 0.5])
    q_fixed = q_fixed / np.linalg.norm(q_fixed)

    # 6. 绕 z 轴旋转 -(270° - θ)
    phi = theta - 270
    # phi = -96.93
    q_z = quat_from_zrot(phi)

    # 7. 最终结果
    q_final = quat_mul(q_z, q_fixed)
    q_final /= np.linalg.norm(q_final)
    return q_final
    # print("最终四元数:", list(np.round(q_final, 6)))


def quat_to_rot(q):
    qw, qx, qy, qz = q
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)]
    ])
    return R

def rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        S = np.sqrt(t+1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S
    return np.array([qw, qx, qy, qz])

def normalize(v):
    return v / np.linalg.norm(v)

def compute_grasp_quat(q_obj):
    """输入红方块四元数(qw,qx,qy,qz)，输出机械臂最终夹持四元数"""
    # 固定旋转：从朝右 -> 朝下
    q_down = np.array([0.5, -0.5, 0.5, 0.5])

    # 1. 把物体quat转为旋转矩阵
    R = quat_to_rot(q_obj)

    # 2. 提取物体x轴在水平面的投影（用于对齐方块边）
    vx = R[:, 0]
    vx[2] = 0.0  # 去掉z分量，只保留平面信息
    if np.linalg.norm(vx) < 1e-6:
        vx = np.array([1, 0, 0])
    vx = normalize(vx)
    vy = np.cross([0, 0, 1], vx)  # 保证右手系
    vz = np.array([0, 0, 1])      # 朝上（重置垂直方向）

    # 3. 重建仅平面旋转的R_flat
    R_flat = np.column_stack((vx, vy, vz))
    q_flat = rot_to_quat(R_flat)

    # 4. 右乘固定变换[0.5,-0.5,0.5,0.5]
    q_final = quat_mul(q_flat, q_down)
    q_final = normalize(q_final)
    return q_final