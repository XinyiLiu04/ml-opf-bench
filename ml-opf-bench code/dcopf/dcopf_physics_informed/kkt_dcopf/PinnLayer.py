import os
import sys
import torch
import torch.nn as nn
import numpy as np

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from DenseCoreNetwork import DenseCoreNetwork
except ImportError:
    pass


class PinnLayer(nn.Module):
    """
    PINN Layer - 集成 Slack Bus 处理 (修复数值稳定性)

    版本: v2.1 - Numerical Stability Fix

    关键修复:
    --------
    1. 添加数值稳定性检查和 epsilon
    2. 修复 KKT Error 计算中的除零问题
    3. 改进 Slack Bus 重构的数值精度
    4. 添加梯度裁剪保护
    5. 所有约束违反使用 relu（保持与原版一致）
    """

    def __init__(self, simulation_parameters, device='cuda'):
        super(PinnLayer, self).__init__()

        self.device = device
        self.eps = 1e-8  # 🆕 数值稳定性常数

        # 缩放参数
        self.pd_scale_type = simulation_parameters.get('pd_scale_type', None)
        if self.pd_scale_type == 'minmax':
            self.pd_min = torch.tensor(simulation_parameters['pd_min'], dtype=torch.float32, device=device)
            self.pd_max = torch.tensor(simulation_parameters['pd_max'], dtype=torch.float32, device=device)
        elif self.pd_scale_type == 'standard':
            self.pd_mean = torch.tensor(simulation_parameters['pd_mean'], dtype=torch.float32, device=device)
            self.pd_std = torch.tensor(simulation_parameters['pd_std'], dtype=torch.float32, device=device)

        # 系统参数
        self.n_buses = simulation_parameters['general']['n_buses']
        self.n_g = simulation_parameters['general']['n_g']
        self.n_g_non_slack = simulation_parameters['general']['n_g_non_slack']
        self.n_line = simulation_parameters['general']['n_line']

        # Slack 信息
        slack_gen_indices = simulation_parameters['general']['slack_gen_indices']
        non_slack_gen_indices = simulation_parameters['general']['non_slack_gen_indices']

        self.slack_gen_indices = torch.tensor(slack_gen_indices, dtype=torch.long, device=device)
        self.non_slack_gen_indices = torch.tensor(non_slack_gen_indices, dtype=torch.long, device=device)
        self.n_slack_gens = len(slack_gen_indices)

        print(f"\n[PinnLayer v2.1] Initialization:")
        print(f"  Total generators: {self.n_g}")
        print(f"  Non-Slack generators: {self.n_g_non_slack}")
        print(f"  Slack generators: {self.n_slack_gens}")
        print(f"  Slack generator indices: {slack_gen_indices}")

        # 网络参数
        neurons_pg = simulation_parameters['training']['neurons_in_hidden_layers_Pg']
        neurons_lm = simulation_parameters['training']['neurons_in_hidden_layers_Lm']

        # 核心网络
        self.core_network = DenseCoreNetwork(
            n_gbus_non_slack=self.n_g_non_slack,
            n_gbus_all=self.n_g,
            n_line=self.n_line,
            neurons_in_hidden_layers_Pg=neurons_pg,
            neurons_in_hidden_layers_Lm=neurons_lm
        ).to(device)

        # 约束参数
        self.C_Pg = torch.tensor(simulation_parameters['constraints']['C_Pg'],
                                 dtype=torch.float32, device=device)
        self.Pg_max = torch.tensor(simulation_parameters['constraints']['Pg_max'],
                                   dtype=torch.float32, device=device)
        self.Pg_min = torch.tensor(simulation_parameters['constraints']['Pg_min'],
                                   dtype=torch.float32, device=device)
        self.Pl_max = torch.tensor(simulation_parameters['constraints']['Pl_max'],
                                   dtype=torch.float32, device=device)
        self.Pg_max_real = torch.tensor(simulation_parameters['constraints']['Pg_max_real'],
                                        dtype=torch.float32, device=device)
        self.PTDF = torch.tensor(simulation_parameters['constraints']['PTDF'],
                                 dtype=torch.float32, device=device)
        self.Map_g = torch.tensor(simulation_parameters['constraints']['Map_g'],
                                  dtype=torch.float32, device=device)
        self.Map_L = torch.tensor(simulation_parameters['constraints']['Map_L'],
                                  dtype=torch.float32, device=device)

        # 确保约束参数是1D张量
        if self.Pg_min.ndim == 2:
            self.Pg_min = self.Pg_min.flatten()
        if self.Pg_max.ndim == 2:
            self.Pg_max = self.Pg_max.flatten()
        if self.Pg_max_real.ndim == 2:
            self.Pg_max_real = self.Pg_max_real.flatten()

        self.Lg_Max = simulation_parameters['Lg_Max']
        self.BASE_MVA = simulation_parameters['general'].get('BASE_MVA', 100.0)

        # 🆕 计算归一化常数（用于数值稳定）
        self.pg_scale = torch.max(self.Pg_max_real) + self.eps
        self.pl_scale = torch.max(self.Pl_max) + self.eps

        print(f"  Pg scale: {self.pg_scale.item():.4f} p.u.")
        print(f"  Pl scale: {self.pl_scale.item():.4f} p.u.")

    def _reconstruct_full_pg(self, pg_non_slack, pd_total):
        """
        重建完整 Pg (改进数值稳定性)

        关键改进:
        --------
        1. 使用 clamp 避免数值溢出
        2. 添加 epsilon 避免除零
        3. 检查 Slack 功率是否合理
        """
        batch_size = pg_non_slack.shape[0]
        device = pg_non_slack.device

        # 初始化完整 Pg
        pg_full = torch.zeros(batch_size, self.n_g,
                              dtype=pg_non_slack.dtype, device=device)

        # 填充非 Slack（归一化值 - 应该在 [0,1] 之间）
        # 🆕 Clamp 确保在合理范围
        pg_non_slack_clamped = torch.clamp(pg_non_slack, 0.0, 1.2)
        pg_full[:, self.non_slack_gen_indices] = pg_non_slack_clamped

        # 计算非 Slack 总出力（物理值 p.u.）
        pg_non_slack_real = pg_non_slack_clamped * self.Pg_max_real[self.non_slack_gen_indices].unsqueeze(0)
        pg_non_slack_total = torch.sum(pg_non_slack_real, dim=1)

        # 计算 Slack 总出力（物理值 p.u.）
        pg_slack_total = pd_total - pg_non_slack_total

        # 🆕 检查 Slack 功率是否合理（避免负值或过大）
        # 允许一定的不平衡（训练初期可能不准确）
        pg_slack_total = torch.clamp(pg_slack_total, -0.1 * self.pg_scale, 2.0 * self.pg_scale)

        # 转换为归一化值并填充
        if self.n_slack_gens > 0:
            # 平均分配到所有 Slack 发电机
            pg_slack_per_gen = pg_slack_total / (self.n_slack_gens + self.eps)

            # 转换为归一化值
            slack_pg_max_real = self.Pg_max_real[self.slack_gen_indices]
            pg_slack_normalized = pg_slack_per_gen.unsqueeze(1) / (slack_pg_max_real.unsqueeze(0) + self.eps)

            # 🆕 Clamp Slack 归一化值
            pg_slack_normalized = torch.clamp(pg_slack_normalized, -0.1, 1.5)

            pg_full[:, self.slack_gen_indices] = pg_slack_normalized

        return pg_full

    def get_kkt_error(self, P_Gens, P_Loads, n_o_l, n_o_a_u, n_o_a_d, n_o_b_u, n_o_b_d):
        """
        计算KKT误差 (改进数值稳定性)

        关键改进:
        --------
        1. 所有除法都添加 epsilon
        2. 使用相对误差而非绝对误差
        3. 添加梯度裁剪保护
        """
        # 🆕 功率平衡误差 (使用相对误差)
        total_gen = torch.sum(P_Gens * self.Pg_max_real, dim=1)
        total_load = torch.sum(P_Loads, dim=1)

        # 相对功率平衡误差
        power_balance_err = torch.abs(total_gen - total_load) / (total_load + self.eps)
        KKT_error = power_balance_err

        # 🆕 发电机约束违反 (归一化)
        gen_up_viol = torch.relu(P_Gens - self.Pg_max)
        gen_lo_viol = torch.relu(self.Pg_min - P_Gens)

        # 使用相对违反量
        KKT_error = KKT_error + torch.sum(gen_up_viol, dim=1) / (self.n_g + self.eps)
        KKT_error = KKT_error + torch.sum(gen_lo_viol, dim=1) / (self.n_g + self.eps)

        # 计算线路潮流
        P_gen_bus = torch.matmul(P_Gens * self.Pg_max_real, self.Map_g)
        P_load_bus = torch.matmul(P_Loads, self.Map_L)
        net_injection = P_gen_bus - P_load_bus
        line_flows = torch.matmul(net_injection, self.PTDF)

        # 🆕 线路违反 (归一化)
        line_viol_pos = torch.relu(line_flows - self.Pl_max)
        line_viol_neg = torch.relu(-line_flows - self.Pl_max)

        # 使用相对违反量
        KKT_error = KKT_error + torch.sum(line_viol_pos, dim=1) / (self.pl_scale * self.n_line + self.eps)
        KKT_error = KKT_error + torch.sum(line_viol_neg, dim=1) / (self.pl_scale * self.n_line + self.eps)

        # 🆕 驻定条件误差 (添加数值稳定性)
        stationarity_term1 = torch.matmul(self.C_Pg.unsqueeze(0), self.Map_g)
        stationarity_term2 = torch.matmul(n_o_a_d * (self.Lg_Max[2] + self.eps), self.Map_g)
        stationarity_term3 = torch.matmul(n_o_a_u * (self.Lg_Max[1] + self.eps), self.Map_g)
        stationarity_term4 = torch.matmul(n_o_b_u * (self.Lg_Max[3] + self.eps), self.PTDF.t())
        stationarity_term5 = torch.matmul(n_o_b_d * (self.Lg_Max[4] + self.eps), self.PTDF.t())

        stationarity_error = torch.abs(
            stationarity_term1 + stationarity_term2 - stationarity_term3 -
            stationarity_term4 + stationarity_term5
        )

        # 归一化驻定条件误差
        stationarity_scale = max(self.Lg_Max) + self.eps
        KKT_error = KKT_error + torch.sum(n_o_l, dim=1) * (self.Lg_Max[0] + self.eps) / (100 * stationarity_scale)
        KKT_error = KKT_error + torch.sum(stationarity_error, dim=1) / (100 * stationarity_scale)

        # 🆕 互补松弛条件 (归一化)
        comp_slack_up = torch.abs(n_o_a_u * (P_Gens - self.Pg_max))
        comp_slack_down = torch.abs(n_o_a_d * (self.Pg_min - P_Gens))

        KKT_error = KKT_error + torch.sum(comp_slack_up, dim=1) / (self.n_g + self.eps)
        KKT_error = KKT_error + torch.sum(comp_slack_down, dim=1) / (self.n_g + self.eps)

        # 线路互补松弛
        line_slack_pos = torch.abs(n_o_b_u * (line_flows - self.Pl_max))
        line_slack_neg = torch.abs(n_o_b_d * (-line_flows - self.Pl_max))

        KKT_error = KKT_error + torch.sum(line_slack_pos, dim=1) / (self.pl_scale * self.n_line + self.eps)
        KKT_error = KKT_error + torch.sum(line_slack_neg, dim=1) / (self.pl_scale * self.n_line + self.eps)

        # 🆕 对偶可行性 (使用 relu)
        KKT_error = KKT_error + torch.sum(torch.relu(-n_o_a_u), dim=1) / (self.n_g + self.eps)
        KKT_error = KKT_error + torch.sum(torch.relu(-n_o_a_d), dim=1) / (self.n_g + self.eps)
        KKT_error = KKT_error + torch.sum(torch.relu(-n_o_b_u), dim=1) / (self.n_line + self.eps)
        KKT_error = KKT_error + torch.sum(torch.relu(-n_o_b_d), dim=1) / (self.n_line + self.eps)

        # 🆕 梯度裁剪保护
        KKT_error = torch.clamp(KKT_error, 0, 1000)  # 避免极端值

        return KKT_error

    def forward(self, inputs):
        """
        前向传播（数值稳定版本）

        返回:
        ----
        - network_output_g_non_slack: [batch, n_g_non_slack]
        - n_o_l: [batch, 1]
        - n_o_a_u, n_o_a_d: [batch, n_g]
        - n_o_b_u, n_o_b_d: [batch, n_line]
        - KKT_error: [batch]
        """
        # 步骤1：核心网络输出
        (network_output_g_non_slack, n_o_l, n_o_a_u,
         n_o_a_d, n_o_b_u, n_o_b_d) = self.core_network(inputs)

        # 负荷去归一化
        if self.pd_scale_type == 'minmax':
            P_Loads_unscaled = inputs * (self.pd_max - self.pd_min) + self.pd_min
        elif self.pd_scale_type == 'standard':
            P_Loads_unscaled = inputs * self.pd_std + self.pd_mean
        else:
            P_Loads_unscaled = inputs

        # 步骤2：重建完整 Pg
        pd_total = torch.sum(P_Loads_unscaled, dim=1)

        network_output_g_full = self._reconstruct_full_pg(
            pg_non_slack=network_output_g_non_slack,
            pd_total=pd_total
        )

        # 步骤3：计算KKT误差
        KKT_error = self.get_kkt_error(
            P_Gens=network_output_g_full,
            P_Loads=P_Loads_unscaled,
            n_o_l=n_o_l,
            n_o_a_u=n_o_a_u,
            n_o_a_d=n_o_a_d,
            n_o_b_u=n_o_b_u,
            n_o_b_d=n_o_b_d
        )

        return (network_output_g_non_slack, n_o_l, n_o_a_u,
                n_o_a_d, n_o_b_u, n_o_b_d, KKT_error)