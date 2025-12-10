import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import tkinter as tk
from tkinter import filedialog
import re
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

# RadarBreathingModel_v1 with Positional Encoding (Commented Out)
def get_positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, seq_len, d_model)

# With square kernels, equal stride and dilation
class MultiScaleDilationBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.dilation_rates = [1, 2, 4, 8]
        self.convs = nn.ModuleList()

        for dilation in self.dilation_rates:
            conv = Dcls1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_count=kernel_size,
                padding=(kernel_size - 1) * dilation // 2,
                dilated_kernel_size=dilation
            )
            self.convs.append(conv)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm1d(out_ch)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out_branches = [conv(x) for conv in self.convs]
        out = torch.cat(out_branches, dim=1)
        residual = self.shortcut(x)
        out += residual
        return self.relu(out)

# CNN + Transformer
class RadarBreathingModel(nn.Module):
    def __init__(self, in_channels=11, hidden_size=128, num_transformer_layers=1, num_classes=4):
        super().__init__()
        self.conv_block = nn.Sequential(
            MultiScaleDilationBlock(in_channels, 16, kernel_size=7),
            nn.MaxPool1d(kernel_size=2),
            MultiScaleDilationBlock(16, 32, kernel_size=5),
            nn.MaxPool1d(kernel_size=2),
            MultiScaleDilationBlock(32, 64, kernel_size=3),
            MultiScaleDilationBlock(64, 128, kernel_size=3),
            MultiScaleDilationBlock(128, 128, kernel_size=3)
        )
        self.feature_projection = nn.Linear(128, hidden_size)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=512, dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(0.4),
            # nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # nn.Linear(32, 1)
        )
        # self.moe_head = RangeMoEHead(128, num_experts=5)

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
        x = self.conv_block(x)
        x = x.permute(0, 2, 1)
        x = self.feature_projection(x)
        # pe = get_positional_encoding(x.size(1), x.size(2)).to(x.device)
        # x = x + pe
        x = self.transformer(x)
        x = self.layer_norm(x[:, -1, :])
        x = self.dense(x)
        
        # rr_output = self.moe_head(x)          # (B, 1)
        rr_output = self.reg_head(x)          # (B, 1)
        act_logits = self.activity_head(x)    # (B, num_classes)
        return rr_output, act_logits

# ---------- Multi-scale conv with GroupNorm ----------
class MultiScaleDilationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(MultiScaleDilationBlock, self).__init__()
        self.dilation_rates = [1, 2, 4, 8]
        self.convs = nn.ModuleList()
        for dilation in self.dilation_rates:
            padding = (kernel_size - 1) * dilation // 2
            conv_branch = nn.Sequential(
                nn.Conv1d(in_channels, out_channels // len(self.dilation_rates), kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_channels // len(self.dilation_rates)),
                nn.ReLU(),
                nn.Conv1d(out_channels // len(self.dilation_rates), out_channels // len(self.dilation_rates), kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_channels // len(self.dilation_rates))
            )
            self.convs.append(conv_branch)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out_branches = [conv(x) for conv in self.convs]
        out = torch.cat(out_branches, dim=1)
        residual = self.shortcut(x)
        out += residual
        return self.relu(out)

#     # ---------- Mixture-of-Experts Output Head ----------
# class RangeMoEHead(nn.Module):
#     """自動分區學習不同呼吸率範圍的輸出頭"""
#     def __init__(self, d_model, num_experts=5):
#         super().__init__()
#         self.experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(d_model, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 64),
#                 nn.ReLU(),
#                 nn.Linear(64, 1)
#             ) for _ in range(num_experts)
#         ])
#         self.gate = nn.Sequential(
#             nn.Linear(d_model, 32),
#             nn.ReLU(),
#             nn.Linear(32, num_experts),
#             nn.Softmax(dim=-1)
#         )

#     def forward(self, x):
#         gate_weights = self.gate(x)  # (B, K)
#         expert_outputs = torch.stack([e(x) for e in self.experts], dim=-1)  # (B, 1, K)
#         return torch.sum(expert_outputs * gate_weights.unsqueeze(1), dim=-1)  # (B, 1)

# # CNN + Transformer
# class RadarBreathingModel(nn.Module):
#     def __init__(self, in_channels=11, hidden_size=256, num_transformer_layers=4, num_classes=4):
#         super().__init__()
#         self.conv_block = nn.Sequential(
#             MultiScaleDilationBlock(in_channels, 16, kernel_size=7),
#             nn.MaxPool1d(kernel_size=2),
#             MultiScaleDilationBlock(16, 32, kernel_size=5),
#             nn.MaxPool1d(kernel_size=2),
#             MultiScaleDilationBlock(32, 64, kernel_size=3),
#             MultiScaleDilationBlock(64, 128, kernel_size=3),
#             MultiScaleDilationBlock(128, 128, kernel_size=3)
#         )
#         self.feature_projection = nn.Linear(128, hidden_size)
#         transformer_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_size, nhead=8, dim_feedforward=512, dropout=0.2, batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
#         self.layer_norm = nn.LayerNorm(hidden_size)
        
#         self.dense = nn.Sequential(
#             nn.Linear(hidden_size, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             # nn.Dropout(0.4),
#             # nn.Linear(64, 32),
#             # nn.BatchNorm1d(32),
#             # nn.ReLU(),
#             # nn.Linear(32, 1)
#         )
#         # self.moe_head = RangeMoEHead(128, num_experts=5)

#         self.reg_head = nn.Sequential(
#             nn.Dropout(0.4),
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         )

#         self.activity_head = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, num_classes)
#         )
#     def forward(self, x):
#         x = self.conv_block(x)
#         x = x.permute(0, 2, 1)
#         x = self.feature_projection(x)
#         # pe = get_positional_encoding(x.size(1), x.size(2)).to(x.device)
#         # x = x + pe
#         x = self.transformer(x)
#         x = self.layer_norm(x[:, -1, :])
#         x = self.dense(x)
        
#         # rr_output = self.moe_head(x)          # (B, 1)
#         rr_output = self.reg_head(x)          # (B, 1)
#         act_logits = self.activity_head(x)    # (B, num_classes)
#         return rr_output, act_logits

# init weights
def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

# Custom Dataset (Unchanged)
class BreathingDataset(Dataset):
    def __init__(self, radar_files, ppg_files, acc_files, resp_files, augment=True):
        self.radar_files = radar_files
        self.ppg_files = ppg_files
        self.acc_files = acc_files
        self.resp_files = resp_files
        self.augment = augment

        self.valid_indices = []
        self.state_labels_all = []   # 存每個樣本對應的 state_k

        for idx, resp_path in enumerate(self.resp_files):
            # 從路徑中抓出 state_k 當作動作標籤
            # ex: .../overlap/state_2/resp/resp_process_overlap_0_5.csv
            m = re.search(r"overlap\\state_(\d+)", resp_path)
            state_label = int(m.group(1)) if m else 0
            self.state_labels_all.append(state_label)

            resp_df = pd.read_csv(resp_path, encoding='utf-8-sig')
            if not pd.isna(resp_df['respiration_rate'].iloc[0]):
                self.valid_indices.append(idx)
            else:
                logging.warning(f"Invalid respiration_rate in {resp_path}")

        print(f"Total files: {len(self.resp_files)}, Valid files: {len(self.valid_indices)}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]

        # ---- Radar (3 ch) ----
        radar_df = pd.read_csv(self.radar_files[valid_idx], encoding='utf-8-sig')
        radar_data = radar_df[['phase_rx0', 'phase_rx1', 'phase_rx2']].values.T  # (3, T)

        # ---- PPG (4 ch) ----
        ppg_df = pd.read_csv(self.ppg_files[valid_idx], encoding='utf-8-sig')
        ppg_data = ppg_df[['MEAS1_PPG1', 'MEAS1_PPG2', 'MEAS1_PPG3', 'MEAS1_PPG4']].values.T  # (4, T)

        # ---- ACC (4 ch：XYZ + ACC_mag) ----
        acc_df = pd.read_csv(self.acc_files[valid_idx], encoding='utf-8-sig')
        acc_data = acc_df[['ACCX', 'ACCY', 'ACCZ', 'ACC_mag']].values.T  # (4, T)

        # ---- 合併成 11 通道 ----
        input_data = np.concatenate([radar_data, ppg_data, acc_data], axis=0)  # (11, T)
        input_data = np.nan_to_num(input_data, nan=0.0)

        # ---- 呼吸率 label ----
        resp_df = pd.read_csv(self.resp_files[valid_idx], encoding='utf-8-sig')
        resp_rate = float(resp_df['respiration_rate'].iloc[0])

        # ---- 狀態 label (0~3) ----
        state_label = self.state_labels_all[valid_idx]

        return (
            torch.FloatTensor(input_data),           # (C, T)
            torch.FloatTensor([resp_rate]),          # (1,)
            torch.tensor(state_label, dtype=torch.long)  # ()
        )

# Parameters
TRAIN_BASE_DIR = r"D:\code\python\Respiration_rate_model_V2\dataset\train_split"
VAL_BASE_DIR = r"D:\code\python\Respiration_rate_model_V2\dataset\val_split"
TEST_BASE_DIR = r"D:\code\python\Respiration_rate_model_V2\dataset\test_split"
MODEL_SAVE_PATH = r"D:\code\python\Respiration_rate_model_V2\result\radar_breathing_model.pth"
BEST_MODEL_SAVE_PATH = r"D:\code\python\Respiration_rate_model_V2\result\radar_breathing_model_best.pth"

BATCH_SIZE = 32
NUM_EPOCHS = 1000
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
ENCODING = 'utf-8-sig'
INPUT_SIZE = 2560
IN_CHANNELS = 11     # 3 radar + 4 ppg + 4 acc
NUM_CLASSES = 4      # state_0, state_1, state_2, state_3
EARLY_STOPPING_PATIENCE = 50

# Ensure output directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Setup logging
current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = f"D:\\code\\python\\Respiration_rate_model_V2\\result\\training_log_{current_time}.txt"
setup_logging(LOG_PATH)
logging.info("Training started")

# Initialize tkinter
root = tk.Tk()
root.withdraw()

folder_count = 0

# Select starting tester_{} folder
print("Please select a tester_{} folder (subsequent folders will be processed automatically, cancel to exit)")
BASE_FOLDER = filedialog.askdirectory(title="Select tester_{} folder", initialdir=TRAIN_BASE_DIR)
if not BASE_FOLDER:
    print("Folder selection canceled")
    logging.info("Folder selection canceled")
else:
    folder_name = os.path.basename(BASE_FOLDER)
    match = re.match(r'tester_(\d+)', folder_name)
    if not match:
        print(f"Error: Selected folder {BASE_FOLDER} does not match tester_{{number}} format")
        logging.error(f"Selected folder {BASE_FOLDER} does not match tester_{{number}} format")
    else:
        start_number = int(match.group(1))
        folder_count = 0

        # Collect data files
        train_radar_files, train_ppg_files, train_acc_files, train_resp_files = [], [], [], []
        val_radar_files,   val_ppg_files,   val_acc_files,   val_resp_files   = [], [], [], []
        test_radar_files,  test_ppg_files,  test_acc_files,  test_resp_files  = [], [], [], []

        folder_count = 0

        # 找出所有 tester_* 資料夾 (不管編號)
        train_folders = [f for f in os.listdir(TRAIN_BASE_DIR) if re.match(r"tester_\d+", f)]
        val_folders = [f for f in os.listdir(VAL_BASE_DIR) if re.match(r"tester_\d+", f)]
        test_folders = [f for f in os.listdir(TEST_BASE_DIR) if re.match(r"tester_\d+", f)]

        # 排序（確保編號順序）
        train_folders.sort(key=lambda x: int(re.search(r"\d+", x).group()))
        val_folders.sort(key=lambda x: int(re.search(r"\d+", x).group()))
        test_folders.sort(key=lambda x: int(re.search(r"\d+", x).group()))

        # ---------- Train ----------
        for folder in train_folders:
            TRAIN_FOLDER = os.path.join(TRAIN_BASE_DIR, folder)
            overlap_root = os.path.join(TRAIN_FOLDER, "overlap")
            if not os.path.exists(overlap_root):
                continue

            state_dirs = [d for d in glob.glob(os.path.join(overlap_root, "state_*")) if os.path.isdir(d)]
            for state_dir in state_dirs:
                radar_dir = os.path.join(state_dir, "radar")
                ppg_dir   = os.path.join(state_dir, "ppg")
                acc_dir   = os.path.join(state_dir, "acc")
                resp_dir  = os.path.join(state_dir, "resp")
                if all(os.path.exists(d) for d in [radar_dir, ppg_dir, acc_dir, resp_dir]):
                    radar_files = sorted(glob.glob(os.path.join(radar_dir, 'radar_process_overlap_*.csv')))
                    ppg_files   = sorted(glob.glob(os.path.join(ppg_dir,   'ppg_process_overlap_*.csv')))
                    acc_files   = sorted(glob.glob(os.path.join(acc_dir,   'acc_process_overlap_*.csv')))
                    resp_files  = sorted(glob.glob(os.path.join(resp_dir,  'resp_process_overlap_*.csv')))

                    if len(radar_files) == len(ppg_files) == len(acc_files) == len(resp_files):
                        train_radar_files.extend(radar_files)
                        train_ppg_files.extend(ppg_files)
                        train_acc_files.extend(acc_files)
                        train_resp_files.extend(resp_files)
                        folder_count += 1

        # ---------- Val ----------
        for folder in val_folders:
            VAL_FOLDER = os.path.join(VAL_BASE_DIR, folder)
            overlap_root = os.path.join(VAL_FOLDER, "overlap")
            if not os.path.exists(overlap_root):
                continue

            state_dirs = [d for d in glob.glob(os.path.join(overlap_root, "state_*")) if os.path.isdir(d)]
            for state_dir in state_dirs:
                radar_dir = os.path.join(state_dir, "radar")
                ppg_dir   = os.path.join(state_dir, "ppg")
                acc_dir   = os.path.join(state_dir, "acc")
                resp_dir  = os.path.join(state_dir, "resp")
                if all(os.path.exists(d) for d in [radar_dir, ppg_dir, acc_dir, resp_dir]):
                    radar_files = sorted(glob.glob(os.path.join(radar_dir, 'radar_process_overlap_*.csv')))
                    ppg_files   = sorted(glob.glob(os.path.join(ppg_dir,   'ppg_process_overlap_*.csv')))
                    acc_files   = sorted(glob.glob(os.path.join(acc_dir,   'acc_process_overlap_*.csv')))
                    resp_files  = sorted(glob.glob(os.path.join(resp_dir,  'resp_process_overlap_*.csv')))

                    if len(radar_files) == len(ppg_files) == len(acc_files) == len(resp_files):
                        val_radar_files.extend(radar_files)
                        val_ppg_files.extend(ppg_files)
                        val_acc_files.extend(acc_files)
                        val_resp_files.extend(resp_files)
                        folder_count += 1

        # ---------- Test ----------
        for folder in test_folders:
            TEST_FOLDER = os.path.join(TEST_BASE_DIR, folder)
            overlap_root = os.path.join(TEST_FOLDER, "overlap")
            if not os.path.exists(overlap_root):
                continue

            state_dirs = [d for d in glob.glob(os.path.join(overlap_root, "state_*")) if os.path.isdir(d)]
            for state_dir in state_dirs:
                radar_dir = os.path.join(state_dir, "radar")
                ppg_dir   = os.path.join(state_dir, "ppg")
                acc_dir   = os.path.join(state_dir, "acc")
                resp_dir  = os.path.join(state_dir, "resp")
                if all(os.path.exists(d) for d in [radar_dir, ppg_dir, acc_dir, resp_dir]):
                    radar_files = sorted(glob.glob(os.path.join(radar_dir, 'radar_process_overlap_*.csv')))
                    ppg_files   = sorted(glob.glob(os.path.join(ppg_dir,   'ppg_process_overlap_*.csv')))
                    acc_files   = sorted(glob.glob(os.path.join(acc_dir,   'acc_process_overlap_*.csv')))
                    resp_files  = sorted(glob.glob(os.path.join(resp_dir,  'resp_process_overlap_*.csv')))

                    if len(radar_files) == len(ppg_files) == len(acc_files) == len(resp_files):
                        test_radar_files.extend(radar_files)
                        test_ppg_files.extend(ppg_files)
                        test_acc_files.extend(acc_files)
                        test_resp_files.extend(resp_files)
                        folder_count += 1

        print(f"Total tester folders processed: {folder_count}")
        logging.info(f"Total tester folders processed: {folder_count}")


        if not (len(train_radar_files) == len(train_ppg_files) == len(train_acc_files) == len(train_resp_files)):
            print("Error: Inconsistent number of training files")
            logging.error("Inconsistent number of training files")
        elif not (len(val_radar_files) == len(val_ppg_files) == len(val_acc_files) == len(val_resp_files)):
            print("Error: Inconsistent number of validation files")
            logging.error("Inconsistent number of validation files")
        elif not (len(test_radar_files) == len(test_ppg_files) == len(test_acc_files) == len(test_resp_files)):
            print("Error: Inconsistent number of test files")
            logging.error("Inconsistent number of test files")
        elif not train_radar_files or not val_radar_files or not test_radar_files:
            print("Error: Training, validation, or test files are empty")
            logging.error("Training, validation, or test files are empty")
        else:
            # Create datasets and loaders
            train_dataset = BreathingDataset(train_radar_files, train_ppg_files, train_acc_files, train_resp_files, augment=True)
            val_dataset = BreathingDataset(val_radar_files, val_ppg_files, val_acc_files, val_resp_files, augment=False)
            test_dataset = BreathingDataset(test_radar_files, test_ppg_files, test_acc_files, test_resp_files, augment=False)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Initialize neural network model, loss, and optimizer
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = RadarBreathingModel(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
            model.apply(init_weights)

            activity_criterion = nn.CrossEntropyLoss()
            # ACT_LOSS_WEIGHT = 0.15   # activity loss weight
            def rr_loss(outputs, targets):
                error = outputs - targets

                # Base huber
                base = F.huber_loss(outputs, targets, delta=1.5, reduction='none')

                # ========== 1) Low respiratory rate (< 14) should not be pushed up ==========
                low_mask = (targets < 14).float()
                penalty_high = torch.clamp(error, min=0)          # pred > true
                loss_low = 0.20 * penalty_high * low_mask

                # ========== 2) Mid range (14–26) slightly increase sensitivity ==========
                mid_mask = ((targets >= 14) & (targets <= 26)).float()
                loss_mid = 0.15 * torch.abs(error) * mid_mask

                # ========== 3) High respiratory rate (> 26) must be strongly learned! ==========
                high_mask = (targets > 26).float()
                under_high = torch.clamp(targets - outputs, min=0)  # true > pred
                loss_high = 0.35 * under_high * high_mask

                return (base + loss_low + loss_mid + loss_high).mean()


            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

            # Neural network training loop
            print(f"Starting neural network training, Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
            logging.info(f"Starting neural network training, Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
            best_val_loss = float('inf')
            best_epoch = 0
            epochs_no_improve = 0
            train_losses, val_losses, epochs_list = [], [], []
            ema_rr = 0.0
            ema_act = 0.0
            alpha = 0.1   # smoothing factor
            for epoch in range(NUM_EPOCHS):
                model.train()
                train_loss = 0.0
                for inputs, rr_targets, state_targets in train_loader:
                    inputs, rr_targets, state_targets = inputs.to(device), rr_targets.to(device), state_targets.to(device)

                    if np.random.rand() < 0.2:
                        noise = torch.randn_like(inputs) * np.sqrt(0.01)
                        inputs = inputs + noise.to(device)

                    if np.random.rand() < 0.2:
                        offset = np.random.randint(0, 128)
                        inputs = inputs[:, :, offset:offset+2560-128]
                        if inputs.size(2) < 2560:
                            padding = inputs[:, :, :2560 - inputs.size(2)]
                            inputs = torch.cat((inputs, padding), dim=2)

                    if np.random.rand() < 0.2:
                        # 隨機選幾個 channel 乘上一個 [0.5, 1.0] 的縮放
                        scale = torch.rand(inputs.size(0), inputs.size(1), 1, device=inputs.device) * 0.5 + 0.5
                        inputs = inputs * scale


                    # if np.random.rand() < 0.2:
                    #     radar_fft = torch.fft.rfft(inputs, dim=-1)
                    #     magnitude = radar_fft.abs()
                    #     phase = radar_fft.angle()
                    #     magnitude *= torch.rand(1, device=device) * 0.2 + 0.9
                    #     radar_fft = magnitude * torch.exp(1j * phase)
                    #     inputs = torch.fft.irfft(radar_fft, n=2560, dim=-1)
                    
                    optimizer.zero_grad()
                    rr_preds, act_logits = model(inputs)

                    loss_rr  = rr_loss(rr_preds, rr_targets)
                    loss_act  = activity_criterion(act_logits, state_targets)

                    ema_rr  = alpha * loss_rr.detach().mean()  + (1 - alpha) * ema_rr
                    ema_act = alpha * loss_act.detach().mean() + (1 - alpha) * ema_act
                    ACT_LOSS_WEIGHT = ema_rr / (ema_act + 1e-8)
                    loss = loss_rr  + ACT_LOSS_WEIGHT * loss_act

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item() * inputs.size(0)

                train_loss /= len(train_dataset)

                model.eval()
                val_loss = 0.0
                val_mae = 0.0
                val_cls_correct = 0
                val_cls_total = 0
                with torch.no_grad():
                    for inputs, rr_targets, state_targets in val_loader:
                        inputs, rr_targets, state_targets = inputs.to(device), rr_targets.to(device), state_targets.to(device)

                        rr_preds, act_logits = model(inputs)

                        loss_rr  = rr_loss(rr_preds, rr_targets)
                        loss_act  = activity_criterion(act_logits, state_targets)
                        loss = loss_rr
                        
                        mae = torch.mean(torch.abs(rr_preds - rr_targets))

                        val_loss += loss.item() * inputs.size(0)
                        val_mae += mae.item() * inputs.size(0)

                        # 動作分類 accuracy（可選，之後可以印在 log）
                        _, act_pred = torch.max(act_logits, dim=1)
                        val_cls_correct += (act_pred == state_targets).sum().item()
                        val_cls_total += state_targets.size(0)
                    val_loss /= len(val_dataset)
                    val_mae  /= len(val_dataset)
                    val_cls_acc = val_cls_correct / val_cls_total if val_cls_total > 0 else 0.0

                scheduler.step()

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                epochs_list.append(epoch + 1)

                log_message = (f"Epoch {epoch+1}/{NUM_EPOCHS}, "
                            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                            f"Val MAE: {val_mae:.4f}, Val Act Acc: {val_cls_acc:.3f}, "
                            f"LR: {optimizer.param_groups[0]['lr']:.6f}")

                print(log_message)
                logging.info(log_message)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        logging.info(f"Early stopping triggered after {epoch+1} epochs")
                        break

            print(f"Best neural network model saved to: {BEST_MODEL_SAVE_PATH}, Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
            logging.info(f"Best neural network model saved to: {BEST_MODEL_SAVE_PATH}, Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Final neural network model saved to: {MODEL_SAVE_PATH}")
            logging.info(f"Final neural network model saved to: {MODEL_SAVE_PATH}")

            # Neural network test evaluation
            model.eval()
            test_cls_correct = 0
            test_cls_total = 0
            test_mae = 0.0
            test_mse = 0.0
            for inputs, rr_targets, state_targets in test_loader:
                inputs = inputs.to(device)
                rr_targets = rr_targets.to(device)
                state_targets = state_targets.to(device)

                rr_pred, act_logits = model(inputs)

                mse_loss = nn.MSELoss()(rr_pred, rr_targets)
                mae_loss = torch.mean(torch.abs(rr_pred - rr_targets))
                test_mse += mse_loss.item() * inputs.size(0)
                test_mae += mae_loss.item() * inputs.size(0)

                _, act_pred = torch.max(act_logits, dim=1)
                test_cls_correct += (act_pred == state_targets).sum().item()
                test_cls_total += state_targets.size(0)

            test_mse /= len(test_dataset)
            test_mae /= len(test_dataset)
            test_cls_acc = test_cls_correct / test_cls_total if test_cls_total > 0 else 0.0

            print(f"Test: MSE = {test_mse:.4f}, MAE = {test_mae:.4f}, Act Acc = {test_cls_acc:.3f}")
            logging.info(f"Neural network test set results: MSE = {test_mse:.4f}, MAE = {test_mae:.4f}, Act Acc = {test_cls_acc:.3f}")

            # Plot neural network losses
            plt.figure(figsize=(10, 6))
            plt.plot(epochs_list, train_losses, label='Train Loss')
            plt.plot(epochs_list, val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Neural Network Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'nn_loss_curve.png'))
            plt.close()
print(f"Processing complete, total folders processed: {folder_count}")
logging.info(f"Processing complete, total folders processed: {folder_count}")