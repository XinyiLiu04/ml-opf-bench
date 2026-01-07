import os
import sys
import torch
import torch.nn as nn

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 🆕 导入 Slack 版本的 PinnLayer
try:
    from PinnLayer_slack import PinnLayer
except ImportError:
    from PINN_DC_KKT.PinnLayer import PinnLayer


class PinnModel(nn.Module):
    """
    PyTorch版本的PINN模型（集成 Slack Bus）

    版本: v2.0 - Slack Bus Integration

    说明:
    ----
    PinnModel 本身不需要大改动，主要改动在 PinnLayer 中。
    这里只需要更新导入，使用支持 Slack Bus 的 PinnLayer。
    """

    def __init__(self, weight1, weight2, simulation_parameters, learning_rate=0.001, device='cuda'):
        super(PinnModel, self).__init__()

        self.device = device

        # 🆕 使用支持 Slack Bus 的 PinnLayer
        self.pinn_layer = PinnLayer(simulation_parameters=simulation_parameters, device=device)

        # 损失权重
        self.loss_weights = [1.0, weight1, weight1, weight1, weight1, weight1, weight2]

        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # 损失函数 (MAE)
        self.criterion = nn.L1Loss()

    def forward(self, inputs):
        """
        前向传播

        返回:
        ----
        - pg_non_slack: [batch, n_g_non_slack] 非 Slack Pg（🆕 改变）
        - lambda, mu_g_up, mu_g_down: 对偶变量（维度保持 n_g）
        - mu_line_up, mu_line_down: 线路对偶变量
        - kkt_error: KKT 误差
        """
        return self.pinn_layer(inputs)

    def compute_loss(self, outputs, targets):
        """
        计算加权损失

        Args:
            outputs: 模型输出 (pg, lambda, mu_g_up, mu_g_down, mu_line_up, mu_line_down, kkt_error)
            targets: 目标值 (pg, lambda, mu_g_up, mu_g_down, mu_line_up, mu_line_down, physics)

        注意:
        ----
        - outputs[0]: pg_non_slack [batch, n_g_non_slack]（🆕 只有非 Slack）
        - targets[0]: pg_non_slack [batch, n_g_non_slack]（🆕 相应调整）
        - 对偶变量维度保持 n_g（所有发电机）
        """
        total_loss = 0.0
        losses = []

        for i, (output, target, weight) in enumerate(zip(outputs, targets, self.loss_weights)):
            loss = self.criterion(output, target)
            weighted_loss = weight * loss
            total_loss += weighted_loss
            losses.append(loss.item())

        return total_loss, losses

    def predict(self, x):
        """预测（推理模式）"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
        return outputs