import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录
VIDEO_PATH = os.path.join(BASE_DIR, "..", "video", "1080p30fps.mp4")  # 视频路径
OUTPUT_FOLDER = os.path.join(BASE_DIR, "..", "processed_data")  # 输出文件夹
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")  # 模型保存路径
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.pt")  # YOLO 模型路径

# 输入特征维度
INPUT_SIZE_LONG = 5  # 输入特征维度（速度、加速度、偏转角等）

# 模型超参数
HIDDEN_SIZE_LONG = 128  # 长视频模型的隐藏层大小
NUM_HEADS_LONG = 8  # 长视频模型的注意力头数
NUM_LAYERS_LONG = 4  # 长视频模型的 Transformer 层数

# 输出参数
NUM_MODES_LONG = 4  # 未来轨迹的模态数（输出4条路径）
OUTPUT_STEPS_LONG = 40  # 预测的时间步数

# 训练参数
BATCH_SIZE_LONG = 1  # 长视频的批量大小
NUM_EPOCHS_LONG = 20  # 长视频的训练轮数
LEARNING_RATE_LONG = 0.0005  # 长视频的学习率
PATIENCE = 5  # 早停法的 patience
MIN_DELTA = 0.01  # 早停法的 min_delta
GRADIENT_CLIP = 1.0  # 梯度裁剪的阈值

# 其他常量
STOP_SPEED_THRESHOLD = 0.01  # 停车速度阈值，单位：移动像素/s
MAX_DISAPPEAR_FRAMES = 30  # 最大允许消失帧数
RESOLUTION_WIDTH = 960  # 统一分辨率宽度
RESOLUTION_HEIGHT = 540  # 统一分辨率高度
LONG_PARKING_THRESHOLD = 60  # 长期停车阈值，单位：秒
MIN_DURATION_THRESHOLD = 1.0  # 最小存在时间阈值，单位：秒