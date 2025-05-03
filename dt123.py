import cv2
import mediapipe as mp
import json
import numpy as np
import math
import json
import os

# 初始化MediaPipe Pose模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 关键点名称列表，按MediaPipe Pose顺序
KEYPOINT_NAMES = [
    "Nose",
    "Left Eye Inner", "Left Eye", "Left Eye Outer",
    "Right Eye Inner", "Right Eye", "Right Eye Outer",
    "Left Ear", "Right Ear",
    "Mouth Left", "Mouth Right",
    "Left Shoulder", "Right Shoulder",
    "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist",
    "Left Pinky", "Right Pinky",
    "Left Index", "Right Index",
    "Left Thumb", "Right Thumb",
    "Left Hip", "Right Hip",
    "Left Knee", "Right Knee",
    "Left Ankle", "Right Ankle",
    "Left Heel", "Right Heel",
    "Left Foot Index", "Right Foot Index"
]

# def process_image(image_path): 过滤一些模糊点
def process_image(image_path, visibility_threshold=0.5):
    # 读取图像
    image = cv2.imread(image_path)
    
    # 转换为RGB，因为MediaPipe要求输入RGB图像
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 处理图像并获取姿态数据
    result = pose.process(image_rgb)
    
    keypoints = []
    
    if result.pose_landmarks:
        for i, landmark in enumerate(result.pose_landmarks.landmark):
            # 获取关键点的名称和坐标
            if landmark.visibility > visibility_threshold:
                keypoints.append({
                    'name': KEYPOINT_NAMES[i],
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            # 在图像上画出关节点
            h, w, _ = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)  # 在关节点位置画绿色圆圈

    # 返回带有关节点标记的图像
    return image, keypoints

def get_keypoints_from_images(front_image_path):
    # 获取带有关节点的图像和关键点数据
    annotated_image, front_pose_keypoints = process_image(front_image_path)
    
    # 输出为 JSON 格式
    result = {
        "front_pose": front_pose_keypoints
    }
    
    # 显示带有关节点的图像
    # cv2.imshow("Annotated Image", annotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return json.dumps(result, indent=4)
    # 111111111111111111111

# 提取关键点方便后续查找
# def get_point(name):
def get_point(name,keypoints_data):
    for p in keypoints_data["front_pose"]:
        if p["name"] == name:
            return p
    return None

# 计算水平（x轴）距离
def horizontal_distance(p1, p2):
    return p1["x"] - p2["x"]

# 计算垂直（y轴）距离
def vertical_distance(p1, p2):
    return p1["y"] - p2["y"]

# 计算两点间距离
def distance(p1, p2):
    return math.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)

def horizontal_distance(p1, p2):
    return abs(p1.x - p2.x)

def vertical_distance(p1, p2):
    return abs(p1.y - p2.y)

def angle_between(p1, p2, p3):
    a = np.array([p1.x - p2.x, p1.y - p2.y])
    b = np.array([p3.x - p2.x, p3.y - p2.y])
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def check_forward_head(ear, shoulder, threshold):
    if ear and shoulder:
        return (ear.x - shoulder.x) > threshold
    return False

def check_rounded_shoulders(shoulder, ear, threshold):
    if shoulder and ear:
        return (shoulder.x - ear.x) > threshold
    return False

def check_pelvic_tilt(hip, knee, ankle, threshold=0.03):
    if hip and knee and ankle:
        hip_knee_dx = horizontal_distance(hip, knee)
        knee_ankle_dx = horizontal_distance(knee, ankle)
        return hip_knee_dx > threshold or knee_ankle_dx > threshold
    return False

def check_high_shoulders(left_shoulder, right_shoulder, dynamic_threshold):
    if left_shoulder and right_shoulder:
        return vertical_distance(left_shoulder, right_shoulder) > dynamic_threshold
    return False

def check_head_tilt(left_ear, right_ear, dynamic_threshold):
    if left_ear and right_ear:
        return vertical_distance(left_ear, right_ear) > dynamic_threshold
    return False

def analyze_posture(keypoints, enabled_checks=None):
    """
    keypoints: dict, 可包含left_ear, right_ear, left_shoulder, right_shoulder, left_hip, right_hip等点
    enabled_checks: list, 指定启用哪些检测项目
    """
    issues = set()
    details = {}

     # 关键点

    left_ear = get_point("Left Ear",keypoints)
    right_ear = get_point("Right Ear",keypoints)
    left_shoulder = get_point("Left Shoulder",keypoints)
    right_shoulder = get_point("Right Shoulder",keypoints)
    left_hip = get_point("Left Hip",keypoints)
    right_hip = get_point("Right Hip",keypoints)
    left_knee = get_point("Left Knee",keypoints)
    right_knee = get_point("Right Knee",keypoints)
    left_ankle = get_point("Left Ankle",keypoints)
    right_ankle = get_point("Right Ankle",keypoints)


    # 动态阈值设定，基于肩宽
    if keypoints.get('left_shoulder') and keypoints.get('right_shoulder'):
        shoulder_width = horizontal_distance(keypoints['left_shoulder'], keypoints['right_shoulder'])
        dynamic_threshold = 0.05 * shoulder_width  # 5%的肩宽作为动态偏差容忍度
    else:
        dynamic_threshold = 0.02  # fallback

    enabled_checks = enabled_checks or [
        "forward_head", "rounded_shoulders", "pelvic_tilt", "high_shoulders", "head_tilt"
    ]

    # 头前引
    if "forward_head" in enabled_checks:
        left = check_forward_head(keypoints.get('left_ear'), keypoints.get('left_shoulder'), dynamic_threshold)
        right = check_forward_head(keypoints.get('right_ear'), keypoints.get('right_shoulder'), dynamic_threshold)
        if left or right:
            issues.add("头前引")
        details['forward_head'] = {"left": left, "right": right}

    # 含胸
    if "rounded_shoulders" in enabled_checks:
        left = check_rounded_shoulders(keypoints.get('left_shoulder'), keypoints.get('left_ear'), dynamic_threshold)
        right = check_rounded_shoulders(keypoints.get('right_shoulder'), keypoints.get('right_ear'), dynamic_threshold)
        if left or right:
            issues.add("含胸")
        details['rounded_shoulders'] = {"left": left, "right": right}

    # 骨盆前倾
    if "pelvic_tilt" in enabled_checks:
        left = check_pelvic_tilt(keypoints.get('left_hip'), keypoints.get('left_knee'), keypoints.get('left_ankle'))
        right = check_pelvic_tilt(keypoints.get('right_hip'), keypoints.get('right_knee'), keypoints.get('right_ankle'))
        if left or right:
            issues.add("骨盆前倾")
        details['pelvic_tilt'] = {"left": left, "right": right}

    # 高低肩
    if "high_shoulders" in enabled_checks:
        diff = check_high_shoulders(keypoints.get('left_shoulder'), keypoints.get('right_shoulder'), dynamic_threshold)
        if diff:
            issues.add("高低肩")
        details['high_shoulders'] = {"asymmetry": diff}

    # 头歪斜
    if "head_tilt" in enabled_checks:
        diff = check_head_tilt(keypoints.get('left_ear'), keypoints.get('right_ear'), dynamic_threshold)
        if diff:
            issues.add("头部歪斜")
        details['head_tilt'] = {"asymmetry": diff}

    return {
        "issues": list(issues),
        "details": details
    }

def generate_simple_report(keypoints, enabled_checks=None):
    result = analyze_posture(keypoints, enabled_checks)
    issues = result['issues']
    
    suggestions_map = {
        "头前引": "注意保持头部自然回正，减少低头时长。",
        "含胸": "可以加强肩胛稳定肌群训练，改善胸廓打开程度。",
        "骨盆前倾": "建议加强核心肌群，特别是腹部与臀部的稳定性训练。",
        "高低肩": "可以进行肩部对称性训练，注意日常姿势习惯。",
        "驼背": "建议注重背部伸展和强化训练，改善体态线条。",
        "寒背": "注意日常挺胸收腹，增加胸椎灵活度训练。",
    }
    
    # 生成个性化的小建议列表
    personal_suggestions = [suggestions_map.get(issue, "") for issue in issues if issue in suggestions_map]
    personal_suggestions = [s for s in personal_suggestions if s]  # 过滤空字符串
    
    # 根据问题数量定制整体建议
    if not issues:
        overall_suggestion = "体态良好，继续保持，日常也可以适当进行拉伸与力量训练巩固效果哦！"
    elif len(issues) <= 2:
        overall_suggestion = "发现了一些小小的体态问题，及时调整可以避免进一步发展哦～加油！"
    else:
        overall_suggestion = "体态存在一定程度的问题，建议系统性改善，循序渐进，身体会越来越好的！"

    # 合成最终建议
    full_suggestion = overall_suggestion
    if personal_suggestions:
        full_suggestion += " " + " ".join(personal_suggestions)
    
    return {
        "issues": issues,
        "suggestion": full_suggestion
    }


# front_image_path = "002.jpg"
# keypoints_json = get_keypoints_from_images(front_image_path)
# keypoints_data = json.loads(keypoints_json)
# diagnosis = generate_simple_report(keypoints_data)
# print(json.dumps(diagnosis, ensure_ascii=False, indent=2))

# 下面的是画图 可以先不写哈 先存着
import matplotlib.pyplot as plt

# 画骨架图并高亮问题
# def plot_skeleton_with_issues(diagnosis,show,keypoints_data):
def plot_skeleton_with_issues(diagnosis, keypoints_data, output_path=None, show=False):
    plt.figure(figsize=(5, 8))
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect('equal')

    # 定义常规连接线
    skeleton = [
        ("Left Shoulder", "Right Shoulder"),
        ("Left Shoulder", "Left Elbow"),
        ("Left Elbow", "Left Wrist"),
        ("Right Shoulder", "Right Elbow"),
        ("Right Elbow", "Right Wrist"),
        ("Left Hip", "Right Hip"),
        ("Left Shoulder", "Left Hip"),
        ("Right Shoulder", "Right Hip"),
        ("Left Hip", "Left Knee"),
        ("Left Knee", "Left Ankle"),
        ("Right Hip", "Right Knee"),
        ("Right Knee", "Right Ankle"),
        ("Left Ankle", "Left Heel"),
        ("Left Heel", "Left Foot Index"),
        ("Right Ankle", "Right Heel"),
        ("Right Heel", "Right Foot Index"),
        ("Nose", "Left Eye"),
        ("Nose", "Right Eye"),
        ("Left Eye", "Left Ear"),
        ("Right Eye", "Right Ear"),
    ]

    # 标记哪些关节连线要高亮
    problem_joints = set()
    if "头前引" in diagnosis["issues"]:
        problem_joints.update(["Left Ear", "Left Shoulder", "Right Ear", "Right Shoulder"])
    if "含胸" in diagnosis["issues"]:
        problem_joints.update(["Left Shoulder", "Left Ear", "Right Shoulder", "Right Ear"])
    if "骨盆前倾" in diagnosis["issues"]:
        problem_joints.update(["Left Hip", "Left Knee", "Left Ankle", "Right Hip", "Right Knee", "Right Ankle"])

    # 开始画
    for joint1, joint2 in skeleton:
        p1 = get_point(joint1,keypoints_data)
        p2 = get_point(joint2,keypoints_data)
        if p1 and p2:
            color = 'red' if joint1 in problem_joints or joint2 in problem_joints else 'blue'
            plt.plot([p1["x"], p2["x"]], [p1["y"], p2["y"]], color=color, marker='o', markersize=5)

    plt.title("Skeleton Visualization with Issues Highlighted")
    plt.savefig('py4001.png', dpi=300)
    if show:
        plt.show()
    plt.close()

# 调用画图
# plot_skeleton_with_issues(diagnosis)
# # 封装整个文件

# def analyze_image(image_path, show_plot=False):
#     # 1. 获取关键点数据
#     keypoints_json = get_keypoints_from_images(image_path)
#     keypoints_data = json.loads(keypoints_json)
    
#     # 2. 生成诊断报告
#     diagnosis = generate_simple_report(keypoints_data)
    
#     # 3. 保存可视化结果（不显示）
#     output_path = f"output_{os.path.basename(image_path)}.png"
#     plot_skeleton_with_issues(diagnosis, keypoints_data,output_path=output_path, show=show_plot)
    
#     return {
#         "diagnosis": diagnosis,
#         "plot_url": f"/results/{os.path.basename(output_path)}"
#     }



def analyze_image(image_path, show_plot=False):
    # 1. 获取关键点数据
    keypoints_json = get_keypoints_from_images(image_path)
    keypoints_data = json.loads(keypoints_json)
    
    # 2. 生成诊断报告
    diagnosis = generate_simple_report(keypoints_data)
    
    # 3. 保存可视化结果（不显示）
    output_path = f"output_{os.path.basename(image_path)}.png"
    plot_skeleton_with_issues(diagnosis, keypoints_data, output_path=output_path, show=show_plot)
    
    return {
        "diagnosis": diagnosis["suggestion"],  # 提供建议内容
        "issues": diagnosis["issues"],  # 如果有问题，返回问题列表
        "plot_url": f"/results/{os.path.basename(output_path)}"  # 返回图像URL
    }