import os 
import glob
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, detrend
import tkinter as tk
from tkinter import filedialog
import re
from scipy.interpolate import interp1d
# from sklearn.decomposition import FastICA
import logging

# Configuration Parameters
INPUT_BASE_DIR = r"dataset\processed_data"
OUTPUT_BASE_DIR = r"dataset\processed_data_label_overlap_interp"
LOG_DIR = r"dataset\dataset_label_overlap_logs"

# Signal Processing Parameters
RADAR_PPG_ACC_SPS   = 128
BREATH_FS           = 20
WINDOW_SEC          = 20  # Window size (seconds)
OVERLAP_PERCENT     = 0.95 # Overlap ratio
ENCODING            = 'utf-8-sig'

# Time Constraints
ALLOWED_TIME_S      = [180, 360] 

# Calculated Parameters
WINDOW_SAMPLES      = int(RADAR_PPG_ACC_SPS * WINDOW_SEC)   # radar/ppg/acc 一窗長度（20s）
STEP_SAMPLES        = int(WINDOW_SAMPLES * (1 - OVERLAP_PERCENT))
BREATH_WINDOW       = int(BREATH_FS * WINDOW_SEC)           # n_process 一窗長度（20s）
BREATH_STEP         = int(BREATH_WINDOW * (1 - OVERLAP_PERCENT))

# Peak Detection Parameters
MIN_PEAK_DIST_SEC     = 2.0
MIN_PEAK_PROMINENCE   = 1.0
PEAK_HEIGHT_THRESHOLD = 1.5

# Global Variables for Respiration Rate Calculation
resp_list = []        # 最近 N 個窗的呼吸率
prev_resp_rate = 0    # 上一窗呼吸率
AVG_WINDOW = 8        # 平滑視窗，建議 6～10 之間

# Setup logging
log_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FILE = os.path.join(LOG_DIR, f"{log_time}.log")
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')

logging.info("Starting data overlap processing")

if not os.path.exists(OUTPUT_BASE_DIR):
    os.makedirs(OUTPUT_BASE_DIR)
    logging.info(f"Created output directory: {OUTPUT_BASE_DIR}")

# Initialize Tkinter root
root = tk.Tk()
root.withdraw()

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    y = filtfilt(b, a, data)
    return y

def find_peaks_in_signal(signal, distance=BREATH_FS, prominence=MIN_PEAK_PROMINENCE, height=PEAK_HEIGHT_THRESHOLD):
    locs, _ = find_peaks(signal, distance=distance, prominence=prominence, height=height)

    if len(locs) > 1:
        intervals = np.diff(locs) / BREATH_FS
        resp_rate = np.mean(60.0 / intervals) if intervals.size > 0 else 0.0
    else:
        resp_rate = 0.0
    return int(resp_rate)

def avg_resp_rates(n_process_window, tester_id, state, window_idx=0):
    global resp_list, prev_resp_rate, AVG_WINDOW

    # ---------------------------------
    # 1. 原始 raw peak-based 呼吸率
    # ---------------------------------
    resp_rate_raw = find_peaks_in_signal(n_process_window)

    # 修正 0 或跳動
    if resp_rate_raw == 0 and prev_resp_rate > 0:
        resp_rate_raw = prev_resp_rate

    # ---------------------------------
    # 2. 平滑視窗平均（原本 avg 算法）
    # ---------------------------------
    if len(resp_list) < AVG_WINDOW:
        resp_list.append(resp_rate_raw)
    else:
        resp_list.pop(0)
        resp_list.append(resp_rate_raw)

    resp_rate_avg = float(np.mean(resp_list))
    prev_resp_rate = resp_rate_avg

    # ---------------------------------
    # 3. 插值呼吸率（比較用，不寫入 CSV）
    # ---------------------------------
    rr_interp = interp_rr_rate(resp_list, window_idx, window_sec=20)

    # ---------------------------------
    # 4. 在 log 內記錄三種版本
    # ---------------------------------
    logging.info(
        f"[Resp RR] tester={tester_id}, state={state}, win={window_idx}, "
        f"raw={resp_rate_raw:.2f}, avg={resp_rate_avg:.2f}, interp={rr_interp:.2f}"
    )

    return rr_interp 

def interp_rr_rate(rr_history, window_idx, window_sec=20):
    """
    rr_history: 過去每個窗口的 raw RR (list)
    window_idx: 目前窗口編號
    """
    if len(rr_history) < 2:
        return rr_history[-1]  # 無法插值 → 回傳自己

    # --- 建立時間軸 ---
    t = np.arange(len(rr_history)) * window_sec  # ex: [0,20,40,...]

    try:
        # cubic interpolation
        f_cubic = interp1d(t, rr_history, kind='cubic', fill_value="extrapolate")
        rr_interp = float(f_cubic(window_idx * window_sec))

        # 合理範圍檢查（避免 cubic overshoot）
        if rr_interp < 5 or rr_interp > 40:
            rr_interp = rr_history[-1]  # fallback

    except:
        # fallback（previous step）
        f_prev = interp1d(t, rr_history, kind='previous', fill_value="extrapolate")
        rr_interp = float(f_prev(window_idx * window_sec))

    return rr_interp

if __name__ == "__main__":
    # Select input directory
    BASE_FOLDER = filedialog.askdirectory(title="Select tester", initialdir=INPUT_BASE_DIR)
    if not BASE_FOLDER:
        logging.error(f"Invalid input directory selected. Exiting.")
        exit()
    else:
        logging.info(f"Selected input directory: {BASE_FOLDER}")

        #　Extract tester name
        tester_name = os.path.basename(BASE_FOLDER)
        matcher = re.match(r'tester_(\d+)', tester_name)
        if not matcher:
            logging.error(f"Invalid tester name: {tester_name}. Exiting.")
            exit()
        else:
            logging.info(f"Processing tester: {tester_name}")
            start_number = int(matcher.group(1))
            folder_counter = 0
            corrent_number = start_number

            while True:
                BASE_FOLDER_CURRENT = os.path.join(INPUT_BASE_DIR, f"tester_{corrent_number}")
                if not os.path.exists(BASE_FOLDER_CURRENT):
                    logging.info(f"No more tester folders found after tester_{corrent_number - 1}. Ending process.")
                    break
                logging.info(f"Processing folder: {BASE_FOLDER_CURRENT}")

                # Reset respiration averaging state for new tester
                resp_list = []
                prev_resp_rate = 0
                logging.info(f"Reset respiration rate averaging state for tester_{corrent_number}")

                OUTPUT_FOLDER_CURRENT = os.path.join(OUTPUT_BASE_DIR, f"tester_{corrent_number}")

                # Create subdirectoies
                if not os.path.exists(OUTPUT_FOLDER_CURRENT):
                    os.makedirs(OUTPUT_FOLDER_CURRENT)
                    logging.info(f"Created output subdirectory: {OUTPUT_FOLDER_CURRENT}")

                state_dir = ["state_0", "state_1", "state_2", "state_3"]
                signal_dir = ["radar", "ppg", "acc", "resp"]
                # Check existing states
                for state_file in state_dir if corrent_number % 2 == 1 else ["state_0"]:
                    state_path = os.path.join(OUTPUT_FOLDER_CURRENT, state_file)
                    if not os.path.exists(state_path):
                        os.makedirs(state_path)
                        logging.info(f"Created state subdirectory: {state_path}")
                    for signal_file in signal_dir:
                        signal_path = os.path.join(state_path, signal_file)
                        if not os.path.exists(signal_path):
                            os.makedirs(signal_path)
                            logging.info(f"Created signal subdirectory: {signal_path}")

                # Read input files
                input_fliles = sorted(glob.glob(os.path.join(BASE_FOLDER_CURRENT, 'process_all_data_*.csv')))
                if not input_fliles:
                    logging.warning(f"No input files found in {BASE_FOLDER_CURRENT}. Skipping to next tester.")
                    corrent_number += 1
                    continue
                corrent_number += 1

                # pass 
                # Process input files
                for input_file in input_fliles:
                    df = pd.read_csv(input_file, encoding=ENCODING, low_memory=False)

                    # Determine data length in seconds
                    data_length = len(df['phase_rx0']) / RADAR_PPG_ACC_SPS
                    TIME_S = min(ALLOWED_TIME_S, key=lambda x: abs(x - data_length))
                    logging.info(f"Processing file: {input_file} with data length: {data_length:.2f}s, using TIME_S: {TIME_S}s")

                    if data_length < TIME_S:
                        logging.warning(f"Data length {data_length:.2f}s is less than TIME_S {TIME_S}s. Skipping file.")
                        continue

                    # Select relevant columns
                    radar_cols = ['phase_rx0', 'phase_rx1', 'phase_rx2']
                    ppg_cols = ['MEAS1_PPG1', 'MEAS1_PPG2', 'MEAS1_PPG3', 'MEAS1_PPG4']
                    acc_cols = ['ACCX', 'ACCY', 'ACCZ']
                    n_process = df['n_process'].values

                    dynamic_ranges = [[0, 59], [60, 179], [180, 299], [300, TIME_S - 1]]
                    static_ranges  = [[0, TIME_S - 1]]

                    state_ranges = dynamic_ranges if TIME_S == 360 else static_ranges
                    for state_idx, time_range in enumerate(state_ranges):
                        resp_list = []
                        prev_resp_rate = 0
                        start_s, end_s = time_range
                        logging.info(f"Processing time range: {start_s}s to {end_s}s")

                        start_sample = start_s * RADAR_PPG_ACC_SPS
                        end_sample   = (end_s + 1) * RADAR_PPG_ACC_SPS

                        # Create output directories for each signal type (非 overlap 原本資料)
                        dir_name = f"state_{state_idx}"
                        radar_output_dir = os.path.join(OUTPUT_FOLDER_CURRENT, dir_name, "radar")
                        ppg_output_dir   = os.path.join(OUTPUT_FOLDER_CURRENT, dir_name, "ppg")
                        acc_output_dir   = os.path.join(OUTPUT_FOLDER_CURRENT, dir_name, "acc")
                        resp_output_dir  = os.path.join(OUTPUT_FOLDER_CURRENT, dir_name, "resp")

                        os.makedirs(radar_output_dir, exist_ok=True)
                        os.makedirs(ppg_output_dir, exist_ok=True)
                        os.makedirs(acc_output_dir, exist_ok=True)
                        os.makedirs(resp_output_dir, exist_ok=True)

                        # -------------------------
                        # 原本：整段 state radar / ppg / acc（非 overlap）
                        # -------------------------
                        # Process Radar Data (full state)
                        radar_dict = {}
                        for radar_idx in radar_cols:
                            radar_signal = df[radar_idx].values
                            radar_segment = radar_signal[start_sample:end_sample]
                            radar_filtered = bandpass_filter(
                                radar_segment,
                                lowcut=0.1,
                                highcut=1.0,
                                fs=RADAR_PPG_ACC_SPS,
                                order=4
                            )
                            radar_dict[radar_idx] = radar_filtered

                        base_name = os.path.basename(input_file).replace("process_all_data", "radar")
                        radar_file_path = os.path.join(radar_output_dir, base_name)
                        pd.DataFrame(radar_dict).to_csv(radar_file_path, index=False)
                        logging.info(f"Saved combined radar RX0/RX1/RX2 to {radar_file_path}")
                        print(f"Saved radar: {radar_file_path}")

                        # Process PPG Data (full state)
                        ppg_dict = {}
                        for ppg_idx in ppg_cols:
                            ppg_signal = df[ppg_idx].values
                            ppg_segment = ppg_signal[start_sample:end_sample]
                            ppg_filtered = bandpass_filter(
                                ppg_segment,
                                lowcut=0.1,
                                highcut=4.0,
                                fs=RADAR_PPG_ACC_SPS,
                                order=4
                            )
                            ppg_dict[ppg_idx] = ppg_filtered

                        ppg_base_name = os.path.basename(input_file).replace("process_all_data", "ppg")
                        ppg_file_path = os.path.join(ppg_output_dir, ppg_base_name)
                        pd.DataFrame(ppg_dict).to_csv(ppg_file_path, index=False)
                        logging.info(f"Saved PPG segment to {ppg_file_path}")
                        print(f"Saved ppg: {ppg_file_path}")

                        # Process ACC Data (full state)
                        acc_dict = {}
                        for acc_idx in acc_cols:
                            acc_signal = df[acc_idx].values
                            acc_segment = acc_signal[start_sample:end_sample]
                            acc_dict[acc_idx] = acc_segment

                        # 轉成 DataFrame 後新增 ACC_mag 欄位
                        acc_df = pd.DataFrame(acc_dict)
                        acc_df['ACC_mag'] = np.sqrt(
                            acc_df['ACCX']**2 + acc_df['ACCY']**2 + acc_df['ACCZ']**2
                        )

                        acc_base_name = os.path.basename(input_file).replace("process_all_data", "acc")
                        acc_file_path = os.path.join(acc_output_dir, acc_base_name)
                        acc_df.to_csv(acc_file_path, index=False)
                        logging.info(f"Saved ACC segment with ACC_mag to {acc_file_path}")
                        print(f"Saved acc: {acc_file_path}")

                        # Process overall respiration rate from n_process (full state)
                        overlap_state_dir = os.path.join(OUTPUT_FOLDER_CURRENT, "overlap", dir_name)
                        radar_overlap_dir = os.path.join(overlap_state_dir, "radar")
                        ppg_overlap_dir   = os.path.join(overlap_state_dir, "ppg")
                        acc_overlap_dir   = os.path.join(overlap_state_dir, "acc")
                        resp_overlap_dir  = os.path.join(overlap_state_dir, "resp")

                        os.makedirs(radar_overlap_dir, exist_ok=True)
                        os.makedirs(ppg_overlap_dir, exist_ok=True)
                        os.makedirs(acc_overlap_dir, exist_ok=True)
                        os.makedirs(resp_overlap_dir, exist_ok=True)

                        # 把原始檔案中的 index 抓出來（ex: process_all_data_3.csv → 3）
                        base_name_full = os.path.basename(input_file)           # process_all_data_3.csv
                        m = re.search(r'process_all_data_(\d+)', base_name_full)
                        file_idx_str = m.group(1) if m else "0"

                        # 以「秒」為單位計算 state 長度
                        state_len_sec = end_s - start_s + 1

                        # 視窗跟步長（秒）
                        STEP_SEC = max(1, int(WINDOW_SEC * (1 - OVERLAP_PERCENT)))  # 20s 視窗，95% overlap → 1s step

                        n_process = df['n_process'].values

                        window_counter = 0
                        for rel_start_sec in range(0, state_len_sec - WINDOW_SEC + 1, STEP_SEC):
                            win_start_sec = start_s + rel_start_sec
                            win_end_sec   = win_start_sec + WINDOW_SEC

                            # ---- Radar / PPG / ACC 的 sample index ----
                            radar_start = win_start_sec * RADAR_PPG_ACC_SPS
                            radar_end   = win_end_sec * RADAR_PPG_ACC_SPS

                            # ---- n_process 的 sample index ----
                            nproc_start = win_start_sec * BREATH_FS
                            nproc_end   = win_end_sec * BREATH_FS

                            # 安全檢查避免越界
                            if radar_end > len(df[radar_cols[0]]) or nproc_end > len(n_process):
                                break

                            # ========== Radar overlapped segment ==========
                            radar_ov_dict = {}
                            for radar_idx in radar_cols:
                                radar_signal = df[radar_idx].values
                                radar_segment = radar_signal[radar_start:radar_end]
                                radar_filtered = bandpass_filter(
                                    radar_segment,
                                    lowcut=0.1,
                                    highcut=1.0,
                                    fs=RADAR_PPG_ACC_SPS,
                                    order=4
                                )
                                radar_ov_dict[radar_idx] = radar_filtered

                            radar_ov_name = f"radar_process_overlap_{file_idx_str}_{window_counter}.csv"
                            radar_ov_path = os.path.join(radar_overlap_dir, radar_ov_name)
                            pd.DataFrame(radar_ov_dict).to_csv(radar_ov_path, index=False)

                            # ========== PPG overlapped segment ==========
                            ppg_ov_dict = {}
                            for ppg_idx in ppg_cols:
                                ppg_signal = df[ppg_idx].values
                                ppg_segment = ppg_signal[radar_start:radar_end]  # 同樣 128 Hz, 用 radar_start/end
                                ppg_filtered = bandpass_filter(
                                    ppg_segment,
                                    lowcut=0.1,
                                    highcut=4.0,
                                    fs=RADAR_PPG_ACC_SPS,
                                    order=4
                                )
                                ppg_ov_dict[ppg_idx] = ppg_filtered

                            ppg_ov_name = f"ppg_process_overlap_{file_idx_str}_{window_counter}.csv"
                            ppg_ov_path = os.path.join(ppg_overlap_dir, ppg_ov_name)
                            pd.DataFrame(ppg_ov_dict).to_csv(ppg_ov_path, index=False)

                            # ========== ACC overlapped segment ==========
                            acc_ov_dict = {}
                            for acc_idx in acc_cols:
                                acc_signal = df[acc_idx].values
                                acc_segment = acc_signal[radar_start:radar_end]
                                acc_ov_dict[acc_idx] = acc_segment

                            acc_ov_df = pd.DataFrame(acc_ov_dict)
                            acc_ov_df['ACC_mag'] = np.sqrt(
                                acc_ov_df['ACCX']**2 + acc_ov_df['ACCY']**2 + acc_ov_df['ACCZ']**2
                            )

                            acc_ov_name = f"acc_process_overlap_{file_idx_str}_{window_counter}.csv"
                            acc_ov_path = os.path.join(acc_overlap_dir, acc_ov_name)
                            acc_ov_df.to_csv(acc_ov_path, index=False)

                            # ========== Resp overlapped (from n_process) ==========
                            nproc_segment = n_process[nproc_start:nproc_end]
                            nproc_filtered = bandpass_filter(
                                nproc_segment,
                                lowcut=0.1,
                                highcut=1.0,
                                fs=BREATH_FS,
                                order=4
                            )

                            resp_rate = avg_resp_rates(nproc_filtered, tester_id=corrent_number, state=dir_name, window_idx=window_counter)

                            resp_ov_name = f"resp_process_overlap_{file_idx_str}_{window_counter}.csv"
                            resp_ov_path = os.path.join(resp_overlap_dir, resp_ov_name)
                            pd.DataFrame({"respiration_rate": [resp_rate]}).to_csv(resp_ov_path, index=False)

                            # logging.info(f"[overlap] state={dir_name}, window={window_counter}, files saved.")
                            print(f"[overlap] Saved window {window_counter} for {dir_name}")

                            window_counter += 1

                print(f"Processing complete.{OUTPUT_FOLDER_CURRENT}")
                