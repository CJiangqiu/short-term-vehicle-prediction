import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConvNet(nn.Module):
    """
    时间卷积网络（TCN），用于捕捉局部时间依赖关系。
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_size if i == 0 else hidden_size, hidden_size, kernel_size=3, padding=1)
            for i in range(num_layers)
        ])

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, seq_len)
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_size)
        return x

class EnhancedTrajectoryPredictor(nn.Module):
    """
    增强的轨迹预测模型，仅考虑单车的历史轨迹数据。
    """
    def __init__(self, input_size, hidden_size, num_heads, num_layers, num_modes, output_steps):
        super().__init__()
        self.temporal_module = TemporalConvNet(input_size, hidden_size, num_layers)
        self.trajectory_head = nn.Linear(hidden_size, output_steps * 2 * num_modes)
        self.probability_head = nn.Linear(hidden_size, num_modes)
        self.num_modes = num_modes
        self.output_steps = output_steps

    def forward(self, x):
        """
        :param x: 输入特征 (batch_size, seq_len, input_size)
        :return:
            - trajectories: 预测轨迹 (batch_size, num_modes, output_steps, 2)
            - probabilities: 模态概率 (batch_size, num_modes)
        """
        # 时间建模
        x = self.temporal_module(x)  # (batch_size, seq_len, hidden_size)

        # 输出头
        trajectories = self.trajectory_head(x[:, -1, :]).view(-1, self.num_modes, self.output_steps, 2)
        probabilities = torch.softmax(self.probability_head(x[:, -1, :]), dim=-1)
        return trajectories, probabilities