import os
import cv2
import numpy as np
import torch
import time
from mss import mss
from ultralytics import YOLO
from sort import Sort
from model import EnhancedTrajectoryPredictor
from config import YOLO_MODEL_PATH, MODEL_DIR, INPUT_SIZE_LONG, HIDDEN_SIZE_LONG, NUM_HEADS_LONG, NUM_LAYERS_LONG, NUM_MODES_LONG, OUTPUT_STEPS_LONG
from PIL import Image, ImageDraw, ImageFont  # 导入PIL库用于绘制中文文本

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 YOLOv8 模型并迁移到 GPU
yolo_model = YOLO(YOLO_MODEL_PATH).to(device)

# 初始化跟踪器
tracker = Sort()

# 加载训练好的轨迹预测模型
model_path = os.path.join(MODEL_DIR, 'best_transformer_trajectory_model.pth')
model = EnhancedTrajectoryPredictor(INPUT_SIZE_LONG, HIDDEN_SIZE_LONG, NUM_HEADS_LONG, NUM_LAYERS_LONG, NUM_MODES_LONG, OUTPUT_STEPS_LONG).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# 屏幕捕获区域（可以根据需要调整）
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # 假设屏幕分辨率为 1920x1080

# 车辆历史数据存储
vehicle_history = {}

# 初始化 mss
sct = mss()

# 停车判断阈值
STOP_SPEED_THRESHOLD = 5  # 速度阈值（像素/秒）
STOP_ANGLE_CHANGE_THRESHOLD = 0.1  # 航向角变化阈值（弧度）

def calculate_speed_and_acceleration(prev_center, current_center, fps):
    """
    计算车辆的速度和加速度。
    """
    dx = current_center[0] - prev_center[0]
    dy = current_center[1] - prev_center[1]
    speed = np.sqrt(dx ** 2 + dy ** 2) * fps  # 速度（像素/秒）
    acceleration = speed - (prev_center[2] if len(prev_center) > 2 else 0)  # 加速度（像素/秒²）
    return speed, acceleration, (dx, dy)

def calculate_heading_angle(dx, dy):
    """
    计算车辆的航向角。
    """
    return np.arctan2(dy, dx)  # 弧度制

def build_features(centers, speeds, accelerations, heading_angles):
    """
    构建输入特征。
    :param centers: 历史路径 (N, 2)
    :param speeds: 速度 (N,)
    :param accelerations: 加速度 (N,)
    :param heading_angles: 航向角 (N,)
    :return: 输入特征 (N, 5)
    """
    features = np.hstack([
        centers,  # 历史路径 (N, 2)
        speeds.reshape(-1, 1),  # 速度 (N, 1)
        accelerations.reshape(-1, 1),  # 加速度 (N, 1)
        heading_angles.reshape(-1, 1),  # 航向角 (N, 1)
    ])
    return features

def predict_trajectory(features):
    """
    调用模型预测车辆轨迹。
    :param features: 输入特征 (N, 5)
    :return: 预测的轨迹 (num_modes, output_steps, 2)
    """
    with torch.no_grad():
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # 转换为张量
        trajectories, _ = model(features)  # 预测轨迹
        trajectories = trajectories.cpu().numpy()[0]  # 转换为 numpy 数组
    return trajectories

def calculate_direction(predicted_points):
    """
    计算车辆的未来行驶方向。
    :param predicted_points: 预测路径点 (N, 2)
    :return: 方向信息（"直行", "左转", "右转"）
    """
    if len(predicted_points) < 2:
        return "路径点数据不足"  # 如果路径点不足，默认直行

    # 计算起点和终点的向量
    start_point = predicted_points[0]
    end_point = predicted_points[-1]
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]

    # 计算角度（弧度制）
    angle = np.arctan2(dy, dx)

    # 判断方向
    if abs(angle) < np.pi / 6:  # 角度小于 30°，认为是直行
        return "直行"
    elif angle > 0:  # 角度为正，右转
        return "右转"
    else:  # 角度为负，左转
        return "左转"

def put_chinese_text(img, text, position, font_path, font_size, color):
    """
    在图像上绘制中文文本。
    :param img: 输入图像 (numpy array)
    :param text: 要绘制的中文文本
    :param position: 文本位置 (x, y)
    :param font_path: 字体文件路径
    :param font_size: 字体大小
    :param color: 文本颜色 (B, G, R)
    :return: 绘制文本后的图像 (numpy array)
    """
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

def is_stopped(speeds, heading_angles):
    """
    判断车辆是否处于停车状态。
    :param speeds: 速度历史列表
    :param heading_angles: 航向角历史列表
    :return: 是否停车 (True/False)
    """
    if len(speeds) < 2 or len(heading_angles) < 2:
        return False  # 数据不足，无法判断

    # 判断速度和航向角是否变化较小
    avg_speed = np.mean(np.abs(speeds[-5:]))  # 取最近5帧的平均速度
    angle_change = np.abs(heading_angles[-1] - heading_angles[-2])  # 航向角变化

    return avg_speed < STOP_SPEED_THRESHOLD and angle_change < STOP_ANGLE_CHANGE_THRESHOLD

def main():
    try:
        # 加载中文字体
        font_path = "simsun.ttc"  # 你需要提供一个支持中文的字体文件路径，例如宋体
        font_size = 20

        while True:
            # 捕获屏幕
            frame = np.array(sct.grab(monitor))  # 捕获屏幕
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # 转换颜色空间

            # 调整帧的分辨率
            frame = cv2.resize(frame, (960, 540))

            # 检测车辆
            results = yolo_model(frame, device='cuda')  # 使用 GPU 进行推理
            detections = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标
                conf = result.boxes.conf.cpu().numpy()  # 置信度
                cls = result.boxes.cls.cpu().numpy()  # 类别
                for i in range(len(boxes)):
                    if cls[i] == 2:  # 只保留汽车类别
                        x1, y1, x2, y2 = boxes[i]
                        detections.append([x1, y1, x2, y2, conf[i]])

            # 更新跟踪器
            if len(detections) > 0:
                detections = np.array(detections)
                tracks = tracker.update(detections[:, :4])  # 只传递边界框信息
            else:
                tracks = np.empty((0, 5))

            # 处理每个跟踪目标
            for track in tracks:
                track_id = int(track[4])  # 获取 track_id
                x1, y1, x2, y2 = track[:4]
                center = ((x1 + x2) / 2, (y1 + y2) / 2)  # 计算车辆中心点

                # 如果 track_id 不在 vehicle_history 中，初始化该车辆的历史数据
                if track_id not in vehicle_history:
                    vehicle_history[track_id] = {
                        'centers': [],
                        'speeds': [],
                        'accelerations': [],
                        'heading_angles': [],
                        'timestamps': [],
                        'predicted_trajectories': []  # 存储预测路径
                    }

                # 计算速度和加速度
                if len(vehicle_history[track_id]['centers']) > 0:
                    prev_center = vehicle_history[track_id]['centers'][-1]
                    speed, acceleration, (dx, dy) = calculate_speed_and_acceleration(prev_center, center, 30)  # 假设帧率为 30 FPS
                    heading_angle = calculate_heading_angle(dx, dy)
                    vehicle_history[track_id]['speeds'].append(speed)
                    vehicle_history[track_id]['accelerations'].append(acceleration)
                    vehicle_history[track_id]['heading_angles'].append(heading_angle)
                else:
                    vehicle_history[track_id]['speeds'].append(0)
                    vehicle_history[track_id]['accelerations'].append(0)
                    vehicle_history[track_id]['heading_angles'].append(0)

                # 保存当前帧的中心点
                vehicle_history[track_id]['centers'].append(center)

                # 判断车辆是否停车
                if is_stopped(vehicle_history[track_id]['speeds'], vehicle_history[track_id]['heading_angles']):
                    # 如果是停车状态，用红色边框标记
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # 红色边框
                    frame = put_chinese_text(frame, "停车", (int(x1), int(y1) - 30), font_path, font_size, (0, 0, 255))  # 红色文本
                else:
                    # 构建输入特征并预测轨迹
                    features = build_features(
                        np.array(vehicle_history[track_id]['centers']),
                        np.array(vehicle_history[track_id]['speeds']),
                        np.array(vehicle_history[track_id]['accelerations']),
                        np.array(vehicle_history[track_id]['heading_angles'])
                    )
                    predicted_trajectories = predict_trajectory(features)

                    # 更新预测路径
                    vehicle_history[track_id]['predicted_trajectories'].append(predicted_trajectories[0])  # 只取第一个模态的预测路径

                    # 如果预测路径过长，删除旧的路径点
                    if len(vehicle_history[track_id]['predicted_trajectories']) > 10:
                        vehicle_history[track_id]['predicted_trajectories'].pop(0)

                    # 绘制绿色边框标记目标
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 绿色边框

                    # 绘制黄色路径标记预测路线
                    predicted_points = np.array(vehicle_history[track_id]['predicted_trajectories']).reshape(-1, 2)
                    predicted_points = predicted_points.astype(np.int32)
                    cv2.polylines(frame, [predicted_points], isClosed=False, color=(0, 255, 255), thickness=2)  # 黄色路径

                    # 计算并显示方向信息（中文）
                    direction = calculate_direction(predicted_points)
                    frame = put_chinese_text(frame, f"方向: {direction}", (int(x1), int(y1) - 30), font_path, font_size, (0, 255, 0))

            # 显示结果
            cv2.imshow("Screen Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 退出
                break

    except KeyboardInterrupt:
        print("程序被用户中断。")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    