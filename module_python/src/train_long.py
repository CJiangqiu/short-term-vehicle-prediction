import os
import time

import numpy as np
import torch
import torch.optim as optim
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import (
    INPUT_SIZE_LONG, HIDDEN_SIZE_LONG, NUM_HEADS_LONG, NUM_LAYERS_LONG,
    NUM_MODES_LONG, OUTPUT_STEPS_LONG, BATCH_SIZE_LONG, NUM_EPOCHS_LONG,
    LEARNING_RATE_LONG, PATIENCE, MIN_DELTA, GRADIENT_CLIP, MODEL_DIR,
    OUTPUT_FOLDER
)
from model import EnhancedTrajectoryPredictor

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义缺失的函数
def build_features(centers, speeds, accelerations, heading_angles):
    """
    构建输入特征。
    :param centers: 历史路径 (N, 2)
    :param speeds: 速度 (N,)
    :param accelerations: 加速度 (N,)
    :param heading_angles: 航向角 (N,)
    :return: 输入特征 (N, 5)
    """
    # 找到最小长度
    min_length = min(len(centers), len(speeds), len(accelerations), len(heading_angles))

    # 截取数据以确保长度一致
    centers = centers[:min_length]
    speeds = speeds[:min_length]
    accelerations = accelerations[:min_length]
    heading_angles = heading_angles[:min_length]

    # 构建输入特征
    features = np.hstack([
        centers,  # 历史路径 (N, 2)
        speeds.reshape(-1, 1),  # 速度 (N, 1)
        accelerations.reshape(-1, 1),  # 加速度 (N, 1)
        heading_angles.reshape(-1, 1),  # 航向角 (N, 1)
    ])
    return features


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
            np.interp(x_new, x, data[:, 1])  # 插值 y 坐标
        ]).T
        return interpolated_data
    return data[:target_length]


# 自定义数据集类
class TrajectoryDataset(Dataset):
    def __init__(self, data_path, mode='train', test_size=0.2, random_state=42):
        self.mode = mode
        self.test_size = test_size
        self.random_state = random_state

        # 加载数据
        data = np.load(data_path, allow_pickle=True)
        vehicle_history = data['vehicle_history'].item()  # 加载车辆历史数据

        # 提取特征和标签
        X, y = [], []
        for track_id, history in vehicle_history.items():
            # 提取车辆的历史中心点、速度、加速度、航向角等信息
            centers = np.array(history['centers'], dtype=float)  # 历史路径
            speeds = np.array(history['speeds'], dtype=float)  # 速度
            accelerations = np.array(history['accelerations'], dtype=float)  # 加速度
            heading_angles = np.array(history['heading_angles'], dtype=float)  # 航向角

            # 处理无效数据（如缺失值）
            if np.isnan(speeds).any():
                valid_indices = np.where(~np.isnan(speeds))[0]
                if len(valid_indices) > 1:  # 至少有 2 个有效点才能插值
                    f = interp1d(valid_indices, speeds[valid_indices], kind='linear', fill_value="extrapolate")
                    speeds = f(np.arange(len(speeds)))

            # 找到最小长度
            min_length = min(len(centers), len(speeds), len(accelerations), len(heading_angles))

            # 截取数据以确保长度一致
            centers = centers[:min_length]
            speeds = speeds[:min_length]
            accelerations = accelerations[:min_length]
            heading_angles = heading_angles[:min_length]

            # 构建输入特征
            features = build_features(centers, speeds, accelerations, heading_angles)

            # 未来路径
            future_centers = interpolate_data(centers, OUTPUT_STEPS_LONG)

            # 添加到数据集
            X.append(features)  # 输入特征（历史路径 + 状态）
            y.append(future_centers)  # 标签（未来路径）

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # 根据模式选择数据
        if mode == 'train':
            self.X = X_train
            self.y = y_train
        else:
            self.X = X_val
            self.y = y_val

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        # 将数据转换为 PyTorch 张量
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return X, y


# 早停法类
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: 允许验证损失不再下降的 epoch 数
        :param min_delta: 验证损失的最小变化量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model, save_path):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), save_path)
            print(f"Validation loss improved. Model saved to {save_path}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# 检查输入数据
def check_data(X, y):
    if torch.isnan(X).any() or torch.isinf(X).any():
        raise ValueError("Input data contains NaN or inf values.")
    if torch.isnan(y).any() or torch.isinf(y).any():
        raise ValueError("Target data contains NaN or inf values.")


# 多模态损失函数
def multimodal_loss(pred_trajectories, pred_probabilities, true_trajectories, num_modes):
    # 调整 true_trajectories 的维度
    true_trajectories = true_trajectories.unsqueeze(1)  # (batch_size, 1, output_steps, 2)

    # 确保 true_trajectories 的 output_steps 与 pred_trajectories 的 output_steps 一致
    if true_trajectories.size(2) != pred_trajectories.size(2):
        true_trajectories = true_trajectories[:, :, :pred_trajectories.size(2), :]

    # 计算轨迹损失
    errors = torch.norm(pred_trajectories - true_trajectories, dim=-1).mean(dim=-1)  # (batch_size, num_modes)
    min_error, _ = errors.min(dim=1)
    trajectory_loss = -torch.log(pred_probabilities + 1e-8) * min_error.unsqueeze(1)  # 避免 log(0)

    # 总损失
    total_loss = trajectory_loss.mean()
    return total_loss


# 训练模型
def train_model(train_loader, val_loader):
    # 构建模型
    model = EnhancedTrajectoryPredictor(INPUT_SIZE_LONG, HIDDEN_SIZE_LONG, NUM_HEADS_LONG, NUM_LAYERS_LONG,
                                        NUM_MODES_LONG, OUTPUT_STEPS_LONG).to(device)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_LONG)
    scaler = GradScaler()  # 用于混合精度训练
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)  # 早停法

    # 模型保存路径
    best_model_path = os.path.join(MODEL_DIR, 'best_transformer_trajectory_model.pth')

    # 训练循环
    for epoch in range(NUM_EPOCHS_LONG):
        model.train()
        train_loss = 0
        start_time = time.time()

        # 训练集进度条
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS_LONG}', unit='batch') as pbar:
            for X, y in pbar:
                X = X.float().to(device)
                y = y.float().to(device)

                # 检查输入数据
                check_data(X, y)

                # 混合精度训练
                with autocast():
                    pred_trajectories, pred_probabilities = model(X)
                    loss = multimodal_loss(pred_trajectories, pred_probabilities, y, NUM_MODES_LONG)

                # 反向传播和优化
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)  # 梯度裁剪
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                pbar.set_postfix({'Train Loss': loss.item()})

        # 验证集评估
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.float().to(device)
                y = y.float().to(device)
                pred_trajectories, pred_probabilities = model(X)
                loss = multimodal_loss(pred_trajectories, pred_probabilities, y, NUM_MODES_LONG)
                val_loss += loss.item()

        # 计算平均损失
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        epoch_time = time.time() - start_time
        print(
            f'Epoch [{epoch + 1}/{NUM_EPOCHS_LONG}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s')

        # 早停法检查并保存最佳模型
        early_stopping(val_loss, model, best_model_path)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    return model


# 主函数
if __name__ == "__main__":
    # 数据文件路径
    data_path = os.path.join(OUTPUT_FOLDER, 'trajectory_long.npz')

    # 构建数据集和数据加载器
    train_dataset = TrajectoryDataset(data_path, mode='train')
    val_dataset = TrajectoryDataset(data_path, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_LONG, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_LONG, shuffle=False)

    # 训练模型
    model = train_model(train_loader, val_loader)

    # 保存最终模型
    final_model_path = os.path.join(MODEL_DIR, 'final_transformer_trajectory_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"模型训练完成并保存为 '{final_model_path}'")