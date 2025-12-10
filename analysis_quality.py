import os
import re
import glob
import datetime
import pandas as pd
import numpy as np
from scipy.signal import correlate
from scipy.fft import rfft, rfftfreq

# === 路徑設定 ===
base_raw   = r"D:\code\python\Respiration_rate_model\dataset\raw_data\ppg_data_MAX86177"
_path= r"dataset"
base_total = r"dataset\processed_data_label_overlap"
base_train = r"dataset\train_split"
base_test  = r"dataset\test_split"
base_val   = r"dataset\val_split"

# 人名與模式
PEOPLE = ["BLACK", "HONG", "HSU", "LEN", "SONG", "BOSHAUN", "DE", "EARZ", "KUNBAO", "QUN"]
MODES  = ["MOTION", "STATIC"]

# === 日期區間 ===
start_date = datetime.date(2025, 6, 30)
end_date   = datetime.date(2025, 8, 21)

# === 正則：BLACK_MOTION_tester_1 ===
folder_pat = re.compile(
    rf'^({"|".join(PEOPLE)})_({"|".join(MODES)})_(?:test|tester)_(\d+)$'
)

# 用來記錄 tester → person 名稱
tester_to_person = {}  # 🔵 新增

folder_counts     = {p: {"MOTION": 0, "STATIC": 0} for p in PEOPLE}
person_testers    = {p: set() for p in PEOPLE}

# === 自動建立 tester_id → person 名字 mapping ===
d = start_date
while d <= end_date:
    day_dir = os.path.join(base_raw, d.strftime("%Y%m%d"))
    if os.path.isdir(day_dir):
        for name in os.listdir(day_dir):
            m = folder_pat.match(name)
            if not m:
                continue
            person, mode, tester_id = m.group(1), m.group(2), int(m.group(3))

            folder_counts[person][mode] += 1
            person_testers[person].add(tester_id)

            tester_to_person[tester_id] = person   # 🔵 新增 mapping

    d += datetime.timedelta(days=1)


# ==========================================================
# 正規化互相關（修正版）
# ==========================================================
def normalized_xcorr(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    N = len(a)

    c = correlate(a, b, mode="same")
    c = c / (np.std(a) * np.std(b) * N + 1e-12)
    return float(np.max(c))


# ==========================================================
# 計算單一 CSV 的所有 Phase 品質指標
# ==========================================================
def compute_metrics_single_csv(phase0, phase1, phase2, fs=128):

    phases = [phase0, phase1, phase2]
    API_list, SCS_list, ASI_list, PUS_list, BR_list = [], [], [], [], []

    for sig in phases:
        sig = sig - np.mean(sig)

        # API
        ac = correlate(sig, sig, mode="full")
        ac = ac[len(ac)//2:]
        R0 = ac[0]
        API = np.max(ac[1:]) / (R0 + 1e-12)
        API_list.append(API)

        # FFT
        N = len(sig)
        freqs = rfftfreq(N, 1/fs)
        spectrum = np.abs(rfft(sig))
        peak_idx = np.argmax(spectrum)
        BR = freqs[peak_idx]
        BR_list.append(BR)

        # SCS
        band = (np.abs(freqs - BR) < 0.05)
        SCS = np.sum(spectrum[band]) / (np.sum(spectrum) + 1e-12)
        SCS_list.append(SCS)

        # ASI
        pos = sig[sig > 0]
        neg = -sig[sig < 0]
        L = min(len(pos), len(neg))
        if L > 5:
            amps = pos[:L] + neg[:L]
            ASI = 1 - (np.std(amps) / (np.mean(amps) + 1e-6))
        else:
            ASI = 0
        ASI_list.append(ASI)

        # PUS
        diff = np.diff(sig)
        PUS = 1 - np.mean(np.abs(diff - np.median(diff)))
        PUS_list.append(PUS)

    # 多通道 Xcorr
    xc01 = normalized_xcorr(phase0, phase1)
    xc02 = normalized_xcorr(phase0, phase2)
    xc12 = normalized_xcorr(phase1, phase2)
    Xcorr = (xc01 + xc02 + xc12) / 3

    # Quality Score
    BR_std = np.std(BR_list)
    Q = (
        0.4 * Xcorr +
        0.3 * (1 - BR_std) +
        0.3 * np.mean(SCS_list)
    )

    return {
        "API": np.mean(API_list),
        "SCS": np.mean(SCS_list),
        "ASI": np.mean(ASI_list),
        "PUS": np.mean(PUS_list),
        "Xcorr": Xcorr,
        "Quality": Q
    }


# ==========================================================
# 主函式：依 state 輸出中位數（加入 person 名字）
# ==========================================================
def compute_phase_metrics_for_tester(tester_folder):

    tester_id = int(re.search(r"tester_(\d+)", tester_folder).group(1))
    person = tester_to_person.get(tester_id, "UNKNOWN")  # 🔵 新增

    overlap_root = os.path.join(tester_folder, "overlap")

    if not os.path.isdir(overlap_root):
        return {}

    results_by_state = {}

    state_dirs = glob.glob(os.path.join(overlap_root, "state_*"))

    for state_dir in state_dirs:

        state_name = os.path.basename(state_dir)

        radar_dir = os.path.join(state_dir, "radar")
        csv_files = glob.glob(os.path.join(radar_dir, "radar_process_overlap_*.csv"))

        if len(csv_files) == 0:
            continue

        API_list, SCS_list, ASI_list, PUS_list, Xcorr_list, Q_list = [], [], [], [], [], []

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)

            if not all(c in df.columns for c in ["phase_rx0", "phase_rx1", "phase_rx2"]):
                continue

            phase0 = df["phase_rx0"].to_numpy()
            phase1 = df["phase_rx1"].to_numpy()
            phase2 = df["phase_rx2"].to_numpy()

            m = compute_metrics_single_csv(phase0, phase1, phase2)

            API_list.append(m["API"])
            SCS_list.append(m["SCS"])
            ASI_list.append(m["ASI"])
            PUS_list.append(m["PUS"])
            Xcorr_list.append(m["Xcorr"])
            Q_list.append(m["Quality"])

        # 🔵 回傳 State 中位數 + 人名 + Tester ID
        results_by_state[state_name] = {
            "Tester_ID": tester_id,
            "Person": person,
            "API": np.median(API_list),
            "SCS": np.median(SCS_list),
            "ASI": np.median(ASI_list),
            "PUS": np.median(PUS_list),
            "Xcorr": np.median(Xcorr_list),
            "Quality": np.median(Q_list),
        }

    return results_by_state



# ======================================================
# 🔥 通用函式：統計每個 split tester_x 的 CSV 數量
# ======================================================
def count_split_csv(base_path):
    split_counts = {}

    if not os.path.isdir(base_path):
        return split_counts

    testers = [f for f in os.listdir(base_path) if re.match(r"tester_\d+", f)]
    testers.sort(key=lambda x: int(re.search(r"\d+", x).group()))

    for folder in testers:
        tester_id = int(re.search(r"\d+", folder).group())
        root_dir = os.path.join(base_path, folder)

        overlap_dir = os.path.join(root_dir, "overlap")
        if not os.path.isdir(overlap_dir):
            split_counts[tester_id] = 0
            continue

        # 取得 state_0, state_1...
        state_dirs = [
            os.path.join(overlap_dir, d)
            for d in os.listdir(overlap_dir)
            if d.startswith("state_") and os.path.isdir(os.path.join(overlap_dir, d))
        ]

        total_csv = 0

        for state_dir in state_dirs:
            radar_dir = os.path.join(state_dir, "radar")
            ppg_dir   = os.path.join(state_dir, "ppg")
            acc_dir   = os.path.join(state_dir, "acc")
            resp_dir  = os.path.join(state_dir, "resp")

            if all(os.path.isdir(d) for d in [radar_dir, ppg_dir, acc_dir, resp_dir]):
                radar_files = glob.glob(os.path.join(radar_dir, "radar_process_overlap_*.csv"))
                ppg_files   = glob.glob(os.path.join(ppg_dir,   "ppg_process_overlap_*.csv"))
                acc_files   = glob.glob(os.path.join(acc_dir,   "acc_process_overlap_*.csv"))
                resp_files  = glob.glob(os.path.join(resp_dir,  "resp_process_overlap_*.csv"))

                # 每組四種類型的檔案，一組等於一筆樣本
                total_csv += min(len(radar_files), len(ppg_files), len(acc_files), len(resp_files))

        split_counts[tester_id] = total_csv

    return split_counts



# === 統計各 split 的 CSV 數 ===
csv_total = count_split_csv(base_total)
csv_train = count_split_csv(base_train)
csv_test  = count_split_csv(base_test)
csv_val   = count_split_csv(base_val)

# ======================================================
# 🔥 By Person (raw_data + split 統計)
# ======================================================
rows_person = []
for p in PEOPLE:
    testers = sorted(person_testers[p])

    # raw total (total_split)
    csv_total_sum = sum(csv_total.get(tid, 0) for tid in testers)
    csv_train_sum = sum(csv_train.get(tid, 0) for tid in testers)
    csv_test_sum  = sum(csv_test.get(tid, 0)  for tid in testers)
    csv_val_sum   = sum(csv_val.get(tid, 0)   for tid in testers)


    rows_person.append({
        "Person": p,
        "MOTION_Folders": folder_counts[p]["MOTION"],
        "STATIC_Folders": folder_counts[p]["STATIC"],
        "Total_Folders": folder_counts[p]["MOTION"] + folder_counts[p]["STATIC"],
        "Tester_IDs": ",".join(map(str, testers)),

        # 直接加總，不要除以 4
        "CSV_Total(for Person)": csv_total_sum,
        "CSV_train(for Person)": csv_train_sum,
        "CSV_test(for Person)":  csv_test_sum,
        "CSV_val(for Person)":   csv_val_sum,
    })

# ======================================================
# 🔥 產生 Phase_Metrics_By_Tester DataFrame
# ======================================================
phase_rows = []

for tester_id in sorted(csv_total.keys()):

    tester_folder = os.path.join(base_total, f"tester_{tester_id}")
    metrics_by_state = compute_phase_metrics_for_tester(tester_folder)

    for state_name, m in metrics_by_state.items():
        phase_rows.append({
            "Tester_ID": m["Tester_ID"],
            "Person": m["Person"],         # 🔵 新增人名
            "State": state_name,
            "API": m["API"],
            "SCS": m["SCS"],
            "ASI": m["ASI"],
            "PUS": m["PUS"],
            "Xcorr": m["Xcorr"],
            "Quality": m["Quality"]
        })

df_phase = pd.DataFrame(phase_rows)

df_person = pd.DataFrame(
    rows_person,
    columns=[
        "Person", "MOTION_Folders", "STATIC_Folders", "Total_Folders",
        "Tester_IDs",
        "CSV_Total(for Person)",
        "CSV_train(for Person)", "CSV_test(for Person)", "CSV_val(for Person)"
    ]
)

# === 轉為 DataFrame ===
df_total = pd.DataFrame([{"Tester_ID": tid, "CSV_Count": cnt} for tid, cnt in sorted(csv_total.items())])
df_train = pd.DataFrame([{"Tester_ID": tid, "CSV_Count": cnt} for tid, cnt in sorted(csv_train.items())])
df_test  = pd.DataFrame([{"Tester_ID": tid, "CSV_Count": cnt} for tid, cnt in sorted(csv_test.items())])
df_val   = pd.DataFrame([{"Tester_ID": tid, "CSV_Count": cnt} for tid, cnt in sorted(csv_val.items())])


# ======================================================
# 🔥 輸出結果到 EXCEL（多 sheet）
# ======================================================
output_path = os.path.join(_path, "ppg_folder_csv_statistics_all_splits.xlsx")

with pd.ExcelWriter(output_path) as writer:
    df_phase.to_excel(writer,  sheet_name="Phase_Metrics_By_Tester", index=False)
    df_person.to_excel(writer, sheet_name="By_Person", index=False)
    df_total.to_excel(writer,  sheet_name="Total_Split", index=False)
    df_train.to_excel(writer,  sheet_name="Train_Split", index=False)
    df_test.to_excel(writer,   sheet_name="Test_Split", index=False)
    df_val.to_excel(writer,    sheet_name="Val_Split", index=False)

print("\n🔥 統計完成：")
print(output_path)
