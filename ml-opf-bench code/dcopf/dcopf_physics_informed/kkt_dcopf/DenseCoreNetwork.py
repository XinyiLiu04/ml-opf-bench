import torch
import torch.nn as nn


class DenseCoreNetwork(nn.Module):
    """
    PINN模型的核心神经网络 - PyTorch版本（集成 Slack Bus）

    版本: v2.0 - Slack Bus Integration

    核心改动:
    --------
    1. Pg 输出维度: n_gbus_non_slack（只预测非 Slack）
    2. 对偶变量维度: n_gbus_all（所有发电机，包括 Slack）

    设计原理:
    --------
    虽然 Pg 只预测非 Slack 发电机，但对偶变量（μ_g）必须包含所有发电机：
    - Slack 发电机虽然由功率平衡确定，但仍然有 Pg_min/max 约束
    - 如果 Slack Pg 触碰约束，相应的对偶变量 μ_g 应该非零
    - KKT 条件要求所有约束都有对应的对偶变量

    支持动态层数
    """

    def __init__(self, n_gbus_non_slack, n_gbus_all, n_line,
                 neurons_in_hidden_layers_Pg, neurons_in_hidden_layers_Lm):
        """
        初始化核心网络

        参数:
        ----
        n_gbus_non_slack : int
            非 Slack 发电机数量（Pg 输出维度）
        n_gbus_all : int
            所有发电机数量（对偶变量维度）
        n_line : int
            支路数量
        neurons_in_hidden_layers_Pg : list
            Pg 网络的隐藏层神经元数
        neurons_in_hidden_layers_Lm : list
            Lm 网络的隐藏层神经元数
        """
        super(DenseCoreNetwork, self).__init__()

        self.n_gbus_non_slack = n_gbus_non_slack
        self.n_gbus_all = n_gbus_all
        self.n_line = n_line

        # ========== Pg网络的隐藏层 (动态创建) ==========
        pg_layers = []
        prev_size = None  # 将在forward中动态确定
        for i, n_units in enumerate(neurons_in_hidden_layers_Pg):
            if i == 0:
                # 第一层的输入大小将在forward中确定
                self.pg_input_size = None
            else:
                pg_layers.append(nn.Linear(prev_size, n_units))
                pg_layers.append(nn.ReLU())
            prev_size = n_units

        self.pg_hidden = nn.ModuleList()
        self.pg_hidden_sizes = neurons_in_hidden_layers_Pg

        # 🆕 Pg输出层（只输出非 Slack）
        self.pg_output = nn.Linear(prev_size, n_gbus_non_slack)

        # ========== Lm网络的隐藏层 (动态创建) ==========
        lm_layers = []
        prev_size = None
        for i, n_units in enumerate(neurons_in_hidden_layers_Lm):
            if i == 0:
                self.lm_input_size = None
            else:
                lm_layers.append(nn.Linear(prev_size, n_units))
                lm_layers.append(nn.ReLU())
            prev_size = n_units

        self.lm_hidden = nn.ModuleList()
        self.lm_hidden_sizes = neurons_in_hidden_layers_Lm

        # Lm输出层 (拉格朗日乘子) - 维度保持所有发电机
        self.lm_output = nn.Linear(prev_size, 1)  # λ (系统级)
        self.mu_g_up_output = nn.Linear(prev_size, n_gbus_all)  # 🆕 所有发电机
        self.mu_g_down_output = nn.Linear(prev_size, n_gbus_all)  # 🆕 所有发电机
        self.mu_line_up_output = nn.Linear(prev_size, n_line)
        self.mu_line_down_output = nn.Linear(prev_size, n_line)

        self._initialized = False

    def _initialize_layers(self, input_size):
        """首次forward时初始化网络层"""
        if self._initialized:
            return

        # 初始化Pg网络
        pg_layers = []
        prev_size = input_size
        for n_units in self.pg_hidden_sizes:
            pg_layers.append(nn.Linear(prev_size, n_units))
            pg_layers.append(nn.ReLU())
            prev_size = n_units
        self.pg_hidden = nn.Sequential(*pg_layers)

        # 初始化Lm网络
        lm_layers = []
        prev_size = input_size
        for n_units in self.lm_hidden_sizes:
            lm_layers.append(nn.Linear(prev_size, n_units))
            lm_layers.append(nn.ReLU())
            prev_size = n_units
        self.lm_hidden = nn.Sequential(*lm_layers)

        self._initialized = True

        # 移动到正确的设备
        device = next(self.parameters()).device
        self.pg_hidden = self.pg_hidden.to(device)
        self.lm_hidden = self.lm_hidden.to(device)

    def forward(self, inputs):
        """
        前向传播 - 支持任意层数

        返回:
        ----
        pg_output : Tensor [batch, n_gbus_non_slack]
            非 Slack 发电机出力预测
        lm_output : Tensor [batch, 1]
            λ - 功率平衡对偶变量
        mu_g_up : Tensor [batch, n_gbus_all]
            发电机上限对偶变量（所有发电机）
        mu_g_down : Tensor [batch, n_gbus_all]
            发电机下限对偶变量（所有发电机）
        mu_line_up : Tensor [batch, n_line]
            线路上限对偶变量
        mu_line_down : Tensor [batch, n_line]
            线路下限对偶变量
        """
        # 首次调用时初始化
        if not self._initialized:
            self._initialize_layers(inputs.shape[1])

        # Pg网络前向传播（🆕 输出非 Slack）
        x_pg = self.pg_hidden(inputs)
        pg_output = self.pg_output(x_pg)  # [batch, n_gbus_non_slack]

        # Lm网络前向传播（维度保持所有发电机）
        x_lm = self.lm_hidden(inputs)
        lm_output = self.lm_output(x_lm)  # [batch, 1]
        mu_g_up = self.mu_g_up_output(x_lm)  # [batch, n_gbus_all]
        mu_g_down = self.mu_g_down_output(x_lm)  # [batch, n_gbus_all]
        mu_line_up = self.mu_line_up_output(x_lm)  # [batch, n_line]
        mu_line_down = self.mu_line_down_output(x_lm)  # [batch, n_line]

        return pg_output, lm_output, mu_g_up, mu_g_down, mu_line_up, mu_line_down