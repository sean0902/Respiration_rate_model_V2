import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import glob
import logging
from datetime import datetime
import re
import torch.nn.functional as F
from DCLS.construct.modules import  Dcls1d 

# Configure logging
def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, seq_len, d_model)

# With square kernels, equal stride and dilation
class MultiScaleDilationBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation_rates=(1, 2, 4, 8)):
        super().__init__()
        self.dilation_rates = list(dilation_rates)
        self.num_branches = len(self.dilation_rates)

        # ✅ out_ch 是 block 最終輸出 channel，所以每個 branch 要分配 channel
        assert out_ch % self.num_branches == 0, \
            f"out_ch({out_ch}) must be divisible by num_branches({self.num_branches})"

        branch_ch = out_ch // self.num_branches
        self.convs = nn.ModuleList()

        for dilation in self.dilation_rates:
            # ⚠️ Dcls1d 的 padding 行為不一定等同 Conv1d，
            # 因此我們用 forward crop 對齊長度，而不是強依賴 padding
            conv = nn.Sequential(
                Dcls1d(
                    in_channels=in_ch,
                    out_channels=branch_ch,
                    kernel_count=kernel_size,
                    dilated_kernel_size=dilation
                ),
                nn.BatchNorm1d(branch_ch),
                nn.ReLU()
            )
            self.convs.append(conv)

        # ✅ shortcut 要對齊到 out_ch
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        # branches: list of (B, branch_ch, T_i)
        out_branches = [conv(x) for conv in self.convs]

        # ✅ 對齊時間長度：裁切到最短（避免 Dcls1d 長度不一致）
        min_len = min(o.size(-1) for o in out_branches)
        out_branches = [o[..., :min_len] for o in out_branches]

        out = torch.cat(out_branches, dim=1)  # (B, out_ch, min_len)

        residual = self.shortcut(x)
        if residual.size(-1) != min_len:
            residual = residual[..., :min_len]

        out = out + residual
        return self.relu(out)

# CNN + Transformer
class RadarBreathingModel(nn.Module):
    def __init__(self, in_channels=11, hidden_size=128, num_transformer_layers=1, num_classes=4):
        super().__init__()

        self.conv_block = nn.Sequential(
            MultiScaleDilationBlock(in_channels, 16, kernel_size=7),  # -> (B,16,T')
            nn.MaxPool1d(kernel_size=2),
            MultiScaleDilationBlock(16, 32, kernel_size=5),           # -> (B,32,T'')
            nn.MaxPool1d(kernel_size=2),
            MultiScaleDilationBlock(32, 64, kernel_size=3),           # -> (B,64,T)
            MultiScaleDilationBlock(64, 128, kernel_size=3),          # -> (B,128,T)
            MultiScaleDilationBlock(128, 128, kernel_size=3)          # -> (B,128,T)
        )

        # ✅ conv_block 輸出 channel=128，所以 projection input=128
        self.feature_projection = nn.Linear(128, hidden_size)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=512,
            dropout=0.4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # backbone feature
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.reg_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.activity_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.conv_block(x)        # (B, 128, T')
        x = x.permute(0, 2, 1)        # (B, T', 128)
        x = self.feature_projection(x) # (B, T', hidden)

        x = self.transformer(x)       # (B, T', hidden)
        x = self.layer_norm(x[:, -1, :])  # (B, hidden)

        feat = self.dense(x)          # (B, 64)
        rr_output = self.reg_head(feat)      # (B, 1)
        act_logits = self.activity_head(feat) # (B, num_classes)
        return rr_output, act_logits

# init weights
def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

class BreathingDataset(Dataset):
    def __init__(self, radar_file, ppg_file, acc_file, resp_file):
        self.radar_file = radar_file
        self.ppg_file = ppg_file
        self.acc_file = acc_file
        self.resp_file = resp_file

        df_resp = pd.read_csv(self.resp_file, encoding='utf-8-sig')
        self.valid = not pd.isna(df_resp['respiration_rate'].iloc[0])

    def __len__(self):
        return 1 if self.valid else 0

    def __getitem__(self, idx):
        if not self.valid:
            raise IndexError("Invalid respiration_rate")

        # -------- Radar --------
        radar_df = pd.read_csv(self.radar_file, encoding='utf-8-sig')
        radar_data = radar_df[['phase_rx0', 'phase_rx1', 'phase_rx2']].values.T

        # -------- PPG --------
        ppg_df = pd.read_csv(self.ppg_file, encoding='utf-8-sig')
        ppg_data = ppg_df[['MEAS1_PPG1','MEAS1_PPG2','MEAS1_PPG3','MEAS1_PPG4']].values.T

        # -------- ACC --------
        acc_df = pd.read_csv(self.acc_file, encoding='utf-8-sig')
        acc_data = acc_df[['ACCX','ACCY','ACCZ','ACC_mag']].values.T

        # -------- Combine into 11-channel input --------
        input_data = np.concatenate([radar_data, ppg_data, acc_data], axis=0)
        input_data = np.nan_to_num(input_data, nan=0.0)

        # -------- Respiration rate --------
        df_r = pd.read_csv(self.resp_file, encoding='utf-8-sig')
        resp_rate = df_r['respiration_rate'].iloc[0]

        return torch.FloatTensor(input_data), torch.FloatTensor([resp_rate])


def compute_stats(true_values, pred_values):
    t = np.asarray(true_values, dtype=np.float32)
    p = np.asarray(pred_values, dtype=np.float32)
    err = p - t
    abs_err = np.abs(err)

    stats = {
        "N": int(len(t)),
        "MAE": float(np.mean(abs_err)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "mean_err": float(np.mean(err)) if len(err) else np.nan,
        "std_err": float(np.std(err)) if len(err) else np.nan,
        "p68_abs_err": float(np.percentile(abs_err, 68)) if len(abs_err) else np.nan,
        "p95_abs_err": float(np.percentile(abs_err, 95)) if len(abs_err) else np.nan,
        "corr": float(np.corrcoef(t, p)[0, 1]) if len(t) > 1 else np.nan,
    }

    return stats

def plot_xy_stats(true_vals, pred_vals, save_path, title='Pred vs true', color='black'):
    if len(true_vals) == 0:
        return

    t = np.asarray(true_vals, dtype=float)
    p = np.asarray(pred_vals, dtype=float)
    stats = compute_stats(t, p)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(t, p, color=color, alpha=0.5, s=20, label="Data")

    # y = x 參考線
    lo = float(min(np.min(t), np.min(p)))
    hi = float(max(np.max(t), np.max(p)))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.5, label="y = x")

    # 最小平方法回歸線
    if len(t) >= 2:
        m, b = np.polyfit(t, p, 1)
        ax.plot([lo - pad, hi + pad],
                [m * (lo - pad) + b, m * (hi + pad) + b],
                lw=1.5, color="blue", label="Fit line")

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("True")
    ax.set_ylabel("Pred")
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper left")

    # 統計框
    txt = (
        f"N = {stats['N']}\n"
        f"MAE = {stats['MAE']:.2f}\n"
        f"mean(err) = {stats['mean_err']:.2f}\n"
        f"std(err) = {stats['std_err']:.2f}\n"
        f"|err| p68 = {stats['p68_abs_err']:.2f}\n"
        f"|err| p95 = {stats['p95_abs_err']:.2f}\n"
        f"corr = {stats['corr']:.3f}"
    )
    ax.text(0.98, 0.98, txt, transform=ax.transAxes, va="top", ha="right", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, lw=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


    # # 也把統計數值另存 CSV
    # stats_df = pd.DataFrame([stats])
    # stats_csv = os.path.splitext(save_path)[0] + "_stats.csv"
    # stats_df.to_csv(stats_csv, index=False, encoding="utf-8-sig")

# 測試函數
def test_model(model, test_loader, device):
    model.eval()
    
    all_true = []
    all_pred = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            rr_output, act_logits = model(inputs)   # <-- 修正！

            pred = rr_output.cpu().numpy().flatten()
            true = targets.cpu().numpy().flatten()

            all_pred.extend(pred)
            all_true.extend(true)

    mae = mean_absolute_error(all_true, all_pred)
    return mae, all_true, all_pred


# 主程式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
TEST_BASE_DIR = r"D:\code\python\Respiration_rate_model_V2\dataset\test_split"
RESULT_DIR = r"D:\code\python\Respiration_rate_model_V2\train_result\results"
MODEL_SAVE_PATH = r"D:\code\python\Respiration_rate_model_V2\result\radar_breathing_model_best.pth"

# Ensure result directory exists
os.makedirs(RESULT_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULT_DIR, "test_log.txt")

# Setup logging
setup_logging(LOG_PATH)
logging.info("Testing started")

# Get range from user
start_number = int(input("Enter the starting tester number (e.g., 11): "))
end_number = int(input("Enter the ending tester number (e.g., 20): "))

# Validate input
if start_number <= 0 or end_number <= 0 or start_number > end_number:
    print("Error: Invalid range. Start number must be positive and less than or equal to end number.")
    logging.error("Invalid range specified for testers.")
else:
    current_number = start_number

    model = RadarBreathingModel(in_channels=11).to(device)
    # model.apply(init_weights)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    model.eval()

    all_true_values = []
    all_pred_values = []
    all_colors = []
    red_true_values = []
    red_pred_values = []
    blue_true_values = []
    blue_pred_values = []

    while current_number <= end_number:
        TEST_FOLDER = os.path.join(TEST_BASE_DIR, f"tester_{current_number}")
        if not os.path.exists(TEST_FOLDER):
            print(f"Folder tester_{current_number} does not exist, skipping")
            logging.info(f"Folder tester_{current_number} does not exist, skipping")
        else:
            # ===== 新增 overlap 路徑 =====
            overlap_dir = os.path.join(TEST_FOLDER, "overlap")

            if not os.path.exists(overlap_dir):
                print(f"No overlap folder found in {TEST_FOLDER}, skipping")
                logging.info(f"No overlap folder found in {TEST_FOLDER}, skipping")
                current_number += 1
                continue

            # 要處理 state_0 / state_1 / state_2 / state_3（或部分存在）
            state_folders = sorted(
                [d for d in os.listdir(overlap_dir) if d.startswith("state_")]
            )

            for state_name in state_folders:
                state_path = os.path.join(overlap_dir, state_name)

                radar_dir = os.path.join(state_path, "radar")
                ppg_dir   = os.path.join(state_path, "ppg")
                acc_dir   = os.path.join(state_path, "acc")
                resp_dir  = os.path.join(state_path, "resp")

                # 檢查子資料夾是否存在
                if not all(os.path.exists(d) for d in [radar_dir, ppg_dir, acc_dir, resp_dir]):
                    print(f"Missing folders in {state_path}, skipping")
                    logging.info(f"Missing folders in {state_path}, skipping")
                    continue

                # 讀取所有 overlap 檔案
                radar_files = sorted(glob.glob(os.path.join(radar_dir, 'radar_process_overlap_*.csv')))
                ppg_files   = sorted(glob.glob(os.path.join(ppg_dir, 'ppg_process_overlap_*.csv')))
                acc_files   = sorted(glob.glob(os.path.join(acc_dir, 'acc_process_overlap_*.csv')))
                resp_files  = sorted(glob.glob(os.path.join(resp_dir, 'resp_process_overlap_*.csv')))

                if not (len(radar_files) == len(ppg_files) == len(acc_files) == len(resp_files)):
                    print(f"File count mismatch in {state_path}, skipping")
                    logging.error(f"File count mismatch in {state_path}, skipping")
                    continue

                # ===== 進行測試 =====
                tester_true_values = []
                tester_pred_values = []
                overall_mae = []
                for idx in range(len(radar_files)):
                    radar_file = radar_files[idx]
                    ppg_file   = ppg_files[idx]
                    acc_file   = acc_files[idx]
                    resp_file  = resp_files[idx]

                    test_dataset = BreathingDataset(radar_file, ppg_file, acc_file, resp_file)
                    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

                    avg_mae, true_values, pred_values = test_model(model, test_loader, device)
                    overall_mae.append(avg_mae)

                    tester_true_values.extend(true_values)
                    tester_pred_values.extend(pred_values)

                    all_true_values.extend(true_values)
                    all_pred_values.extend(pred_values)
                    overall_avg_mae = np.mean(overall_mae)

                    color = 'blue' if current_number % 2 == 0 else 'red'
                    all_colors.extend([color] * len(true_values))
                    if color == 'red':
                        red_true_values.extend(true_values)
                        red_pred_values.extend(pred_values)
                    else:
                        blue_true_values.extend(true_values)
                        blue_pred_values.extend(pred_values)
                    print(f"[tester_{current_number}][{state_name}] file {idx} → MAE={avg_mae:.3f}")
                    logging.info(f"[tester_{current_number}][{state_name}] file {idx} → MAE={avg_mae:.3f}")

                # Draw Bland-Altman plot for the current tester
                if start_number <= current_number <= end_number:
                    tester_diffs = np.array(tester_pred_values) - np.array(tester_true_values)
                    tester_means = (np.array(tester_true_values) + np.array(tester_pred_values)) / 2
                    tester_std = np.std(tester_diffs)
                    color = 'blue' if current_number % 2 == 0 else 'red'
                    fig, ax1 = plt.subplots(figsize=(8, 6))
                    ax1.scatter(tester_means, tester_diffs, c=color, alpha=0.5, label=f'tester_{current_number} ({color})')
                    ax1.axhline(np.mean(tester_diffs), color='r', linestyle='--', label='Mean Difference')
                    ax1.axhline(np.mean(tester_diffs) + 1.96 * tester_std, color='g', linestyle='--', label='±1.96 SD')
                    ax1.axhline(np.mean(tester_diffs) - 1.96 * tester_std, color='g', linestyle='--')
                    ax1.set_xlabel('Respiration rate Values')
                    ax1.set_ylabel('Difference (Predicted - True)')
                    ax1.set_title(f'Bland-Altman Plot - tester_{current_number} (Avg MAE: {overall_avg_mae:.4f})')
                    ax1.legend()
                    ax1.grid(True)
                    # Add right y-axis for std values
                    ax2 = ax1.twinx()
                    ax2.set_ylim(ax1.get_ylim())  # Align with left y-axis
                    ax2.set_yticks([np.mean(tester_diffs) + 1.96 * tester_std, np.mean(tester_diffs), np.mean(tester_diffs) - 1.96 * tester_std])
                    ax2.set_yticklabels([f'{np.mean(tester_diffs) + 1.96 * tester_std:.4f}', f'{np.mean(tester_diffs):.4f}', f'{np.mean(tester_diffs) - 1.96 * tester_std:.4f}'])
                    plt.savefig(os.path.join(RESULT_DIR, f'bland_altman_tester_{current_number}.png'))
                    plt.close()

                    plot_xy_stats(tester_true_values, tester_pred_values,
                                    os.path.join(RESULT_DIR, f'pred_vs_true_tester_{current_number}.png'),
                                    title=f'Pred vs True - tester_{current_number}', color=color)

        current_number += 1

    # Draw overall Bland-Altman plot for all test data
    if all_true_values and all_pred_values:
        all_diffs = np.array(all_pred_values) - np.array(all_true_values)
        all_means = (np.array(all_true_values) + np.array(all_pred_values)) / 2
        overall_avg_mae = np.mean([mean_absolute_error(np.array(all_true_values), np.array(all_pred_values))])
        overall_std = np.std(all_diffs)
        fig, ax1 = plt.subplots(figsize=(10, 8))
        scatter = ax1.scatter(all_means, all_diffs, c=all_colors, alpha=0.5)
        ax1.axhline(np.mean(all_diffs), color='r', linestyle='--', label='Mean Difference')
        ax1.axhline(np.mean(all_diffs) + 1.96 * overall_std, color='g', linestyle='--', label='±1.96 SD')
        ax1.axhline(np.mean(all_diffs) - 1.96 * overall_std, color='g', linestyle='--')
        ax1.set_xlabel('Respiration rate Values')
        ax1.set_ylabel('Difference (Predicted - True)')
        ax1.set_title(f'Bland-Altman Plot - All Test Data (Avg MAE: {overall_avg_mae:.4f})')
        ax1.legend()
        ax1.grid(True)
        # Add right y-axis for std values
        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim())  # Align with left y-axis
        ax2.set_yticks([np.mean(all_diffs) + 1.96 * overall_std, np.mean(all_diffs), np.mean(all_diffs) - 1.96 * overall_std])
        ax2.set_yticklabels([f'{np.mean(all_diffs) + 1.96 * overall_std:.4f}', f'{np.mean(all_diffs):.4f}', f'{np.mean(all_diffs) - 1.96 * overall_std:.4f}'])
        plt.savefig(os.path.join(RESULT_DIR, 'bland_altman_all_test.png'))
        plt.close()

        plot_xy_stats(all_true_values, all_pred_values,
                      os.path.join(RESULT_DIR, 'pred_vs_true_all_test.png'),
                      title='Pred vs True - All Test Data', color=all_colors)

    # Draw Bland-Altman plot for all red points (dynamic testers)
    if red_true_values and red_pred_values:
        red_diffs = np.array(red_pred_values) - np.array(red_true_values)
        red_means = (np.array(red_true_values) + np.array(red_pred_values)) / 2
        red_avg_mae = np.mean([mean_absolute_error(np.array(red_true_values), np.array(red_pred_values))])
        red_std = np.std(red_diffs)
        fig, ax1 = plt.subplots(figsize=(10, 8))
        ax1.scatter(red_means, red_diffs, c='red', alpha=0.5, label='Dynamic Testers')
        ax1.axhline(np.mean(red_diffs), color='r', linestyle='--', label='Mean Difference')
        ax1.axhline(np.mean(red_diffs) + 1.96 * red_std, color='g', linestyle='--', label='±1.96 SD')
        ax1.axhline(np.mean(red_diffs) - 1.96 * red_std, color='g', linestyle='--')
        ax1.set_xlabel('Respiration rate Values')
        ax1.set_ylabel('Difference (Predicted - True)')
        ax1.set_title(f'Bland-Altman Plot - Dynamic Testers (Red Points, Avg MAE: {red_avg_mae:.4f})')
        ax1.legend()
        ax1.grid(True)
        # Add right y-axis for std values
        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim())  # Align with left y-axis
        ax2.set_yticks([np.mean(red_diffs) + 1.96 * red_std, np.mean(red_diffs), np.mean(red_diffs) - 1.96 * red_std])
        ax2.set_yticklabels([f'{np.mean(red_diffs) + 1.96 * red_std:.4f}', f'{np.mean(red_diffs):.4f}', f'{np.mean(red_diffs) - 1.96 * red_std:.4f}'])
        plt.savefig(os.path.join(RESULT_DIR, 'bland_altman_dynamic_red.png'))
        plt.close()

        plot_xy_stats(red_true_values, red_pred_values,
                      os.path.join(RESULT_DIR, 'pred_vs_true_dynamic_red.png'),
                      title='Pred vs True - Dynamic Testers (Red Points)', color='red')

    # Draw Bland-Altman plot for all blue points (static testers)
    if blue_true_values and blue_pred_values:
        blue_diffs = np.array(blue_pred_values) - np.array(blue_true_values)
        blue_means = (np.array(blue_true_values) + np.array(blue_pred_values)) / 2
        blue_avg_mae = np.mean([mean_absolute_error(np.array(blue_true_values), np.array(blue_pred_values))])
        blue_std = np.std(blue_diffs)
        fig, ax1 = plt.subplots(figsize=(10, 8))
        ax1.scatter(blue_means, blue_diffs, c='blue', alpha=0.5, label='Static Testers')
        ax1.axhline(np.mean(blue_diffs), color='r', linestyle='--', label='Mean Difference')
        ax1.axhline(np.mean(blue_diffs) + 1.96 * blue_std, color='g', linestyle='--', label='±1.96 SD')
        ax1.axhline(np.mean(blue_diffs) - 1.96 * blue_std, color='g', linestyle='--')
        ax1.set_xlabel('Respiration rate Values')
        ax1.set_ylabel('Difference (Predicted - True)')
        ax1.set_title(f'Bland-Altman Plot - Static Testers (Blue Points, Avg MAE: {blue_avg_mae:.4f})')
        ax1.legend()
        ax1.grid(True)
        # Add right y-axis for std values
        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim())  # Align with left y-axis
        ax2.set_yticks([np.mean(blue_diffs) + 1.96 * blue_std, np.mean(blue_diffs), np.mean(blue_diffs) - 1.96 * blue_std])
        ax2.set_yticklabels([f'{np.mean(blue_diffs) + 1.96 * blue_std:.4f}', f'{np.mean(blue_diffs):.4f}', f'{np.mean(blue_diffs) - 1.96 * blue_std:.4f}'])
        plt.savefig(os.path.join(RESULT_DIR, 'bland_altman_static_blue.png'))
        plt.close()

        plot_xy_stats(blue_true_values, blue_pred_values,
                      os.path.join(RESULT_DIR, 'pred_vs_true_static_blue.png'),
                      title='Pred vs True - Static Testers (Blue Points)', color='blue')

print(f"Processing complete from tester_{start_number} to tester_{end_number}")
logging.info(f"Processing complete from tester_{start_number} to tester_{end_number}")