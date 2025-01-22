import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from config import *
from sort import Sort

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 YOLOv8 模型并迁移到 GPU
yolo_model = YOLO(YOLO_MODEL_PATH).to(device)

# 初始化跟踪器
tracker = Sort()

# 常量定义
STOP_SPEED_THRESHOLD = 0.01  # 停车速度阈值，单位：移动像素/s
MAX_DISAPPEAR_FRAMES = 10  # 最大允许消失帧数
RESOLUTION_WIDTH = 960  # 统一分辨率宽度
RESOLUTION_HEIGHT = 540  # 统一分辨率高度
OUTPUT_FRAME_FOLDER = os.path.join(OUTPUT_FOLDER, "video_frame_data")  # 保存帧图片的文件夹
LONG_PARKING_THRESHOLD = 60  # 长期停车阈值，单位：秒

# 车辆类别 ID（假设类别 ID 2 是汽车）
CAR_CLASS_ID = 2

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

def smooth_data(data, window_size=5):
    """
    使用滑动平均平滑数据。
    """
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def interpolate_data(data, target_length):
    """
    对数据进行插值，使其长度达到目标长度。
    :param data: 输入数据 (N, 2)
    :param target_length: 目标长度
    :return: 插值后的数据 (target_length, 2)
    """
    if len(data) < target_length:
        x = np.arange(len(data))
        x_new = np.linspace(0, len(data) - 1, target_length)
        interpolated_data = np.array([
            np.interp(x_new, x, data[:, 0]),  # 插值 x 坐标
            np.interp(x_new, x, data[:, 1])   # 插值 y 坐标
        ]).T
        return interpolated_data
    return data[:target_length]

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

def filter_data(vehicle_history, min_duration_threshold):
    """
    过滤无效数据。
    :param vehicle_history: 车辆历史数据
    :param min_duration_threshold: 最小存在时间阈值
    :return: 过滤后的车辆历史数据
    """
    filtered_history = {}
    for track_id, history in vehicle_history.items():
        duration = history['timestamps'][-1] - history['timestamps'][0]
        if duration >= min_duration_threshold:
            filtered_history[track_id] = history
    return filtered_history

def draw_frame(frame, vehicle_history, current_tracks, last_seen, frame_id):
    """
    在原始帧上绘制车辆行驶路径历史和当前目标。
    :param frame: 原始帧
    :param vehicle_history: 车辆历史数据
    :param current_tracks: 当前跟踪的目标
    :param last_seen: 车辆的最后出现时间
    :param frame_id: 当前帧 ID
    :return: 绘制后的帧
    """
    # 绘制车辆行驶路径历史（黄色）
    for track_id, history in vehicle_history.items():
        # 只绘制当前仍然存在的车辆路径
        if track_id in last_seen and (frame_id - last_seen[track_id]) <= MAX_DISAPPEAR_FRAMES:
            if len(history['centers']) > 1:
                # 绘制路径
                for i in range(1, len(history['centers'])):
                    prev_center = tuple(map(int, history['centers'][i - 1]))
                    curr_center = tuple(map(int, history['centers'][i]))
                    cv2.line(frame, prev_center, curr_center, (0, 255, 255), 2)  # 黄色高亮

    # 绘制当前目标
    for track in current_tracks:
        x1, y1, x2, y2, track_id = track
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        # 判断是否静止
        if track_id in vehicle_history:
            speeds = vehicle_history[track_id]['speeds']
            if len(speeds) > 0 and np.mean(speeds[-10:]) < STOP_SPEED_THRESHOLD:
                # 静止目标用红色标记
                cv2.circle(frame, tuple(map(int, center)), 5, (0, 0, 255), -1)
            else:
                # 运动目标用绿色标记
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    return frame

def process_video_long(video_path, output_path):
    """
    处理整个长视频，提取车辆运动数据，并保存帧图片。
    """
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数

    # 创建保存帧图片的文件夹
    if not os.path.exists(OUTPUT_FRAME_FOLDER):
        os.makedirs(OUTPUT_FRAME_FOLDER)

    # 存储数据
    data = {
        'vehicle_history': {},  # 每个车辆的完整历史路径
        'timestamps': [],  # 时间戳
        'frame_ids': []  # 帧 ID
    }

    # 记录被删除的车辆
    deleted_tracks = set()

    # 记录车辆的最后出现时间
    last_seen = {}

    # 使用 tqdm 创建进度条
    with tqdm(total=frame_count, desc="Processing Video", unit="frame") as pbar:
        for frame_id in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # 调整帧的分辨率
            vehicle_frame = cv2.resize(frame, (960, 544))  # 调整帧分辨率

            # 将帧数据转换为 PyTorch 张量并移动到 GPU
            frame_tensor = torch.from_numpy(vehicle_frame).float().to(device) / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # 调整维度顺序

            # 检测车辆（使用 GPU 加速）
            with torch.no_grad():
                results = yolo_model(frame_tensor, device='cuda')  # 使用 GPU 进行推理

            # 批量处理检测结果（只保留汽车类别）
            detections = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标
                conf = result.boxes.conf.cpu().numpy()  # 置信度
                cls = result.boxes.cls.cpu().numpy()  # 类别

                for i in range(len(boxes)):
                    if cls[i] == CAR_CLASS_ID:  # 只保留汽车类别
                        x1, y1, x2, y2 = boxes[i]
                        detections.append([x1, y1, x2, y2, conf[i], cls[i]])

            # 更新跟踪器
            if len(detections) > 0:
                detections = np.array(detections)
                tracks = tracker.update(detections[:, :4])  # 只传递边界框信息
            else:
                tracks = np.empty((0, 5))

            # 更新车辆的最后出现时间
            current_track_ids = set()
            for track in tracks:
                track_id = int(track[4])
                current_track_ids.add(track_id)
                last_seen[track_id] = frame_id  # 更新最后出现时间

            # 处理每个跟踪目标
            for track in tracks:
                track_id = int(track[4])  # 获取 track_id

                # 如果该车辆已被删除，则跳过
                if track_id in deleted_tracks:
                    continue

                x1, y1, x2, y2 = track[:4]
                center = ((x1 + x2) / 2, (y1 + y2) / 2)  # 计算车辆中心点

                # 如果 track_id 不在 vehicle_history 中，初始化该车辆的历史数据
                if track_id not in data['vehicle_history']:
                    data['vehicle_history'][track_id] = {
                        'centers': [],
                        'speeds': [],
                        'accelerations': [],
                        'heading_angles': [],
                        'timestamps': []
                    }

                # 计算速度和加速度
                if len(data['vehicle_history'][track_id]['centers']) > 0:
                    prev_center = data['vehicle_history'][track_id]['centers'][-1]
                    speed, acceleration, (dx, dy) = calculate_speed_and_acceleration(prev_center, center, fps)
                    heading_angle = calculate_heading_angle(dx, dy)
                    data['vehicle_history'][track_id]['speeds'].append(speed)
                    data['vehicle_history'][track_id]['accelerations'].append(acceleration)
                    data['vehicle_history'][track_id]['heading_angles'].append(heading_angle)
                else:
                    data['vehicle_history'][track_id]['speeds'].append(0)
                    data['vehicle_history'][track_id]['accelerations'].append(0)
                    data['vehicle_history'][track_id]['heading_angles'].append(0)

                # 保存当前帧的中心点
                data['vehicle_history'][track_id]['centers'].append(center)
                data['vehicle_history'][track_id]['timestamps'].append(frame_id / fps)

                # 检查是否长期停车
                if len(data['vehicle_history'][track_id]['timestamps']) > 0:
                    duration = data['vehicle_history'][track_id]['timestamps'][-1] - data['vehicle_history'][track_id]['timestamps'][0]
                    if duration > LONG_PARKING_THRESHOLD and np.mean(data['vehicle_history'][track_id]['speeds']) < STOP_SPEED_THRESHOLD:
                        # 删除长期停车的数据
                        del data['vehicle_history'][track_id]
                        deleted_tracks.add(track_id)  # 记录被删除的车辆

            # 记录时间戳和帧 ID
            data['timestamps'].append(frame_id / fps)
            data['frame_ids'].append(frame_id)

            # 每秒保存一张帧图片
            if frame_id % int(fps) == 0:
                frame_image = draw_frame(vehicle_frame, data['vehicle_history'], tracks, last_seen, frame_id)
                frame_image_path = os.path.join(OUTPUT_FRAME_FOLDER, f"frame_{frame_id}.jpg")
                cv2.imwrite(frame_image_path, frame_image)

            # 更新进度条
            pbar.update(1)

    # 对速度和加速度进行平滑处理
    for track_id, history in data['vehicle_history'].items():
        if len(history['speeds']) > 0:
            history['speeds'] = smooth_data(history['speeds'])
            history['accelerations'] = smooth_data(history['accelerations'])

    # 保存数据
    np.savez(output_path, **data)
    print('Final data saved to {}'.format(output_path))

    cap.release()

if __name__ == "__main__":
    # 确保输出文件夹存在
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 处理整个长视频
    output_path = os.path.join(OUTPUT_FOLDER, 'trajectory_long.npz')
    process_video_long(VIDEO_PATH, output_path)