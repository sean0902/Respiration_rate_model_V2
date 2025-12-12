import os
import sys
import pandas as pd
import numpy as np
import torch
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog,
    QLabel, QHBoxLayout, QSlider
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from torch.utils.data import Dataset, DataLoader
import logging
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.metrics import mean_absolute_error
# Import the model class
# Assuming the model is defined in a file named model.py
from model_architecture import RadarBreathingModel

MODEL_SAVE_PATH = r"D:\Research\AI_model\Respiration_Rate_Analysis_Research\train_result\train_cnn_transformer_0822_20_sec\radar_breathing_model_best.pth"
BATCH_SIZE = 32


# Configure logging
def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a bandpass filter to the data."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

class BreathingDataset(Dataset):
    """Dataset for breathing data."""
    def __init__(self, ppg_signal, radar_signal, acc_signal, respiration_rate):
        """Initialize the dataset with file paths."""
        # All must be already slider to the same length
        assert ppg_signal.ndim == 2 and radar_signal.ndim == 2 and acc_signal.ndim == 2, "Signals must be 2D arrays"
        T = min(ppg_signal.shape[1], radar_signal.shape[1], acc_signal.shape[1])
        # Trim to the shortest length to keep shape consistent
        ppg_signal = ppg_signal[:, :T]
        radar_signal = radar_signal[:, :T]
        acc_signal = acc_signal[:, :T]

        x = np.concatenate([ppg_signal, radar_signal, acc_signal], axis=0)
        x = np.nan_to_num(x, nan=0.0).astype(np.float32)  # Replace NaNs with 0.0

        self.x = torch.from_numpy(x)
        self.y = torch.tensor([respiration_rate], dtype=torch.float32)
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.x, self.y

class DataVisualizer(QMainWindow):
    def __init__(self, *args, **kwargs):
        """Initializer. Set up the main window and UI components."""
        super().__init__(*args, **kwargs)
        self.dark_mode = False
        self.init_ui()                  
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        self.main_Layout = QVBoxLayout(self.main_widget)

        # Add a button to open file dialog (top left corner)
        top_bar = QHBoxLayout()

        self.btn_open_file = QPushButton("Open File", self)
        self.btn_open_file.clicked.connect(self.open_file_dialog)

        self.btn_theme = QPushButton("Dark mode", self)
        self.btn_theme.clicked.connect(self.toggle_theme)

        self.btn_run = QPushButton("Run Model", self)
        self.btn_run.clicked.connect(lambda: self.run_model(MODEL_SAVE_PATH))

        right_group = QHBoxLayout()
        right_group.setSpacing(5)  # Spacing between buttons
        right_group.addWidget(self.btn_theme)
        right_group.addWidget(self.btn_run)

        top_bar.addWidget(self.btn_open_file, alignment=Qt.AlignmentFlag.AlignLeft)
        top_bar.addStretch(1)  # Stretch to fill space
        top_bar.addLayout(right_group)
        self.main_Layout.addLayout(top_bar)

        # Label to show the selected file
        self.lbl_file_info = QLabel("No file selected", self)
        self.main_Layout.addWidget(self.lbl_file_info)
        self.main_Layout.addStretch(1)

        # Matplot figure for plotting
        self.figure, self.axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True) # 4 subplot
        self.canvas = FigureCanvas(self.figure)
        self.main_Layout.addWidget(self.canvas)

        # Slider for navigation
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)  # Initially disabled
        self.slider.valueChanged.connect(self.update_plot)
        self.main_Layout.addWidget(self.slider)

        # Bottom bar for navigation buttons and result display
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(0, 0, 0, 0)

        # Navigation buttons
        nav_group = QHBoxLayout()
        nav_group.setSpacing(5)
        self.btn_left = QPushButton("<", self)
        self.btn_right = QPushButton(">", self)
        self.btn_left.clicked.connect(self.move_left)
        self.btn_right.clicked.connect(self.move_right)
        self.btn_left.clicked.connect(lambda: self.run_model(MODEL_SAVE_PATH))
        self.btn_right.clicked.connect(lambda: self.run_model(MODEL_SAVE_PATH))
        nav_group.addWidget(self.btn_left)
        nav_group.addWidget(self.btn_right)

        # Result display label
        self.result_label = QLabel("", self)
        self.result_label.setMinimumWidth(220)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Respiration avg 7 data
        self.seven_rr_label = QLabel("7-win RR: -", self)
        self.seven_rr_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Add to bottom bar
        bottom_bar.addLayout(nav_group, stretch=0)
        bottom_bar.addWidget(self.seven_rr_label, stretch=8)
        bottom_bar.addSpacing(8)
        bottom_bar.addStretch(1)
        bottom_bar.addWidget(self.result_label, stretch=0)

        # Add bottom bar to main layout
        self.main_Layout.addLayout(bottom_bar)

        
        # Data placeholder
        self.filtetered_signals = {}
        self.max_time = 0
        self.window_size = 1  # seconds
        self.respiration_rate = 0
        self.respiration_rate_avg7 = 0
        self.respiration_rate_find_peaks = 0
        self.window_sec = 20  # seconds for model input

        # Apply default theme
        self.apply_light_theme()

    def apply_light_theme(self):
        """Apply light theme."""
        self.setStyleSheet("""
            QWidget { background-color: #f7f7fb; font-family: "Segoe UI", Arial; font-size: 13px; color: #222; }
            QLabel { color: #333; }
            QPushButton {
                background-color: #6a85b6; color: white; border-radius: 10px;
                padding: 6px 12px; font-weight: 600; border: 0px;
            }
            QPushButton:hover { background-color: #5a75a6; }
            QSlider::groove:horizontal { border:1px solid #bbb; background:#e7e7ee; height:6px; border-radius:3px; }
            QSlider::handle:horizontal { background:#6a85b6; border:1px solid #444; width:14px; height:14px; margin:-5px 0; border-radius:7px; }
            QLabel {
                border: 1px solid rgba(0,0,0,0.25);
                border-radius: 8px;
                padding: 6px 10px;
                background: rgba(255,255,255,0.75);
                color: #222;
                font-weight: 600;
            }
        """)
        self.figure.patch.set_facecolor("#f7f7fb")
        for ax in self.axes: ax.set_facecolor("#ffffff")
        self._apply_mpl_look()
        self.canvas.draw_idle()

    def _apply_dark_theme(self):
        """Dark theme"""
        self.setStyleSheet("""
            QWidget { background-color: #1f2125; font-family: "Segoe UI", Arial; font-size: 13px; color: #eee; }
            QLabel { color: #eee; }
            QPushButton {
                background-color: #3b3f46; color: #f3f3f3; border-radius: 10px;
                padding: 6px 12px; font-weight: 600; border: 0px;
            }
            QPushButton:hover { background-color: #4a5059; }
            QSlider::groove:horizontal { border:1px solid #555; background:#2b2f36; height:6px; border-radius:3px; }
            QSlider::handle:horizontal { background:#aaa; border:1px solid #222; width:14px; height:14px; margin:-5px 0; border-radius:7px; }
            QLabel {
                border: 1px solid rgba(255,255,255,0.35);
                border-radius: 8px;
                padding: 6px 10px;
                background: rgba(0,0,0,0.35);
                color: #eee;
                font-weight: 600;
            }
        """)
        self.figure.patch.set_facecolor("#1f2125")
        for ax in self.axes: ax.set_facecolor("#262a30")
        self._apply_mpl_look()
        self.canvas.draw_idle()

    def toggle_theme(self):
        """swith Light / Dark"""
        if self.dark_mode:
            self.apply_light_theme()
            self.btn_theme.setText("Dark Mode")
        else:
            self._apply_dark_theme()
            self.btn_theme.setText("Light Mode")
        self.dark_mode = not self.dark_mode

    def init_ui(self):
        """Initialize window setting."""
        self.setWindowTitle("Radar Breathing Data Visualizer")
        self.setGeometry(10, 50, 1000, 600) # x, y, width, height

    def move_left(self):
        """Move the slider left."""
        current_value = self.slider.value()
        if current_value > 0:
            self.slider.setValue(current_value - 1)
        
    def move_right(self):
        """Move the slider right."""
        current_value = self.slider.value()
        # if current_value < self.slider.maximum() && current_value <  (self.max_time // self.window_size) - 1:
        print(f"Max slider: {self.slider.maximum()}, Current: {current_value}, Max time: {self.max_time}, Window size: {self.window_size}")
        self.slider.setValue(current_value + 1)

    def _apply_mpl_look(self):
        """依視窗大小與主題動態調整 Matplotlib 風格"""
        # 取得目前畫布大小（像素）
        w = max(self.canvas.width(), 1)
        h = max(self.canvas.height(), 1)

        # 動態字級
        base = max(8, min(14, int(h / 60)))       # tick
        title = max(10, min(16, int(h / 40)))     # title
        legend = max(8, min(13, int(h / 65)))     # legend

        # 顏色
        fg = "#111111" if not self.dark_mode else "#f0f0f0"
        grid_c = "#cccccc" if not self.dark_mode else "#666666"
        spine_c = fg

        for ax in self.axes:
            ax.tick_params(labelsize=base, colors=fg)
            for spine in ax.spines.values():
                spine.set_color(spine_c)
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)
            ax.title.set_fontsize(title)
            ax.grid(True, color=grid_c, linewidth=0.6, alpha=0.6)

            # 每次重畫時 legend 用統一字級與顏色
            leg = ax.get_legend()
            if leg:
                for text in leg.get_texts():
                    text.set_fontsize(legend)
                    text.set_color(fg)
                leg.get_frame().set_edgecolor(fg)
                leg.get_frame().set_alpha(0.1)

        # 讓子圖間距跟著尺寸調整
        self.figure.subplots_adjust(top=0.95, bottom=0.08, left=0.07, right=0.98, hspace=0.35)

    def update_plot(self):
        """Update the plot based on the slider value."""
        # if end_sec < self.max_time:
        start_sec = self.slider.value() * self.window_size    
        end_sec = start_sec + self.window_sec  # Show 20 seconds of data
        print(f"Update plot: {start_sec} to {end_sec} seconds")

        color = "#111111" if not self.dark_mode else "#f0f0f0"

        for ax in self.axes:
            ax.clear()

        self._apply_mpl_look()
        # Group 1   
        for col in ['MEAS1_PPG1', 'MEAS1_PPG2', 'MEAS1_PPG3', 'MEAS1_PPG4']:
            if col in self.filtetered_signals:
                t, y = self.filtetered_signals[col]
                mask = (t >= start_sec) & (t < end_sec)
                self.axes[0].plot(t[mask], y[mask], label=col)
        self.axes[0].set_title("PPG Signals", color=color)

        # Group 2 (ACC)
        for col in ['ACCX', 'ACCY', 'ACCZ', 'ACC_Magnitude']:  # ← 加入 ACC_Magnitude
            if col in self.filtetered_signals:
                t, y = self.filtetered_signals[col]
                mask = (t >= start_sec) & (t < end_sec)
                self.axes[1].plot(t[mask], y[mask], label=col)
        self.axes[1].set_title("Accelerometer Signals", color=color)
        self.axes[1].legend()

        print('Has ACC_Magnitude?', 'ACC_Magnitude' in self.filtetered_signals)

        # Group 3
        for col in ['phase_rx0', 'phase_rx1', 'phase_rx2']:
            if col in self.filtetered_signals:
                t, y = self.filtetered_signals[col]
                
                mask = (t >= start_sec) & (t < end_sec)
                self.axes[2].plot(t[mask], y[mask], label=col)
        self.axes[2].set_title("Phase Signals", color=color)

        # Group 4: n_process + respiration rate
        if 'n_process' in self.filtetered_signals:
            t, y = self.filtetered_signals['n_process']
            mask = (t >= start_sec) & (t < end_sec)
            t_win, y_win = t[mask], y[mask]

            if len(t_win) > 0:
                # Calculate respiration rate
                self.axes[3].plot(t_win, y_win, label='n_process')
                y_arr = np.array(y_win)
                t_arr = np.array(t_win)
                # Find peaks in the n_process signal
                peaks, _ = find_peaks(y_arr, distance=20, height=0.5, prominence=1.0)
                self.axes[3].plot(t_arr[peaks], y_arr[peaks], 'ro', label='Peaks')
                self.respiration_rate_find_peaks = len(peaks) * 3  # peaks per 20 seconds

                fs = len(t_arr) / (t_arr[-1] - t_arr[0]) if (t_arr[-1] - t_arr[0]) > 0 else 1
                N = len(y_arr)
                yf = np.fft.rfft(y_arr - np.mean(y_arr))
                xf = np.fft.rfftfreq(N, 1/fs)

                amplitude = np.abs(yf)
                # Limit to 0.1-0.6 Hz (6-42 bpm)
                mask_freq = (xf >= 0.1) & (xf <= 0.7)   
                if np.any(mask_freq):
                    xf_masked = xf[mask_freq]
                    amplitude_masked = amplitude[mask_freq]
                    peak_idx = np.argmax(amplitude_masked)
                    peak_freq = xf_masked[peak_idx]
                    freq_bpm = peak_freq * 60
                    self.respiration_rate = freq_bpm

                    # Calculate 7-window average respiration rate
                    self.respiration_rate_avg7, rr_list = self._avg_rr_over_7(start_sec)
                    print(f"respiration_rate_avg7:{self.respiration_rate_avg7}")

                    title_fft = f"{self.respiration_rate:.2f} bpm" if not np.isnan(self.respiration_rate) else "N/A"
                    title_avg = f"{self.respiration_rate_avg7:.2f} bpm" if not np.isnan(self.respiration_rate_avg7) else "N/A"

                self.axes[3].set_title(
                    f"n_process Signal (FFT RR: {title_fft}, 7-win avg RR: {title_avg})",
                    color=color
                )

                parts = []
                secs = []
                for rr_sec, rr_win_data in rr_list:  # rr_list 要是 [(秒數, RR), ...]
                    if rr_win_data is None or (isinstance(rr_win_data, float) and np.isnan(rr_win_data)):
                        s = "—"
                    else:
                        s = f"{rr_win_data:.1f}"
                    parts.append(s)
                    secs.append(f"{int(rr_sec)}s")  # 秒數顯示成整數

                print(f"parts:{parts}, secs:{secs}")

                rows = []
                rows.append("<tr>" + "".join(f"<td>{p}</td>" for p in parts) + "</tr>")
                rows.append("<tr>" + "".join(f"<td>{s}</td>" for s in secs) + "</tr>")

                self.seven_rr_label.setText(
                    "<table border='0' cellspacing='8' cellpadding='2'>"
                    + "".join(rows) +
                    "</table>"
                )


                # self.axes[3].set_title(f"n_process Signal (Respiration Rate: {self.respiration_rate:.2f} bpm, Peaks: {respiration_rate_peak_find} bpm)", color=color)


        # msg_html = (
        #     f"FFT RR: <b>{title_fft}</b><br>"
        #     f"7-win Avg RR: <b>{title_avg}</b>"
        # )
        # self.result_label.setText(msg_html)

                
        for ax in self.axes:
            ax.grid(True)
            ax.legend()
        self.figure.tight_layout()
        self.canvas.draw()

    def _rr_from_segment(self, y_arr: np.ndarray, t_arr: np.ndarray) -> float:
        """
        用 FFT 在 0.1~0.7 Hz（6~42 bpm）找主頻，回傳 bpm。
        若資料長度或時間軸不合法則回傳 np.nan。
        """
        if len(y_arr) < 4 or len(t_arr) < 2 or (t_arr[-1] - t_arr[0]) <= 0:
            return np.nan
        fs = len(t_arr) / (t_arr[-1] - t_arr[0])

        sig = y_arr - np.mean(y_arr)               # 去 DC
        yf = np.fft.rfft(sig)
        xf = np.fft.rfftfreq(len(sig), 1.0 / fs)
        amp = np.abs(yf)

        # 只看 0.1~0.7 Hz
        mask = (xf >= 0.1) & (xf <= 0.7)
        if not np.any(mask):
            return np.nan
        xf_m = xf[mask]
        amp_m = amp[mask]

        if len(amp_m) <= 1:
            return np.nan
        idx = np.argmax(amp_m)          # 最大振幅
        freq_hz = xf_m[idx]
        return float(freq_hz * 60.0)    # 轉 bpm
    
    def _rr_form_find_peaks(self, y_arr: np.ndarray) -> float:
        """
        用 find_peaks 找峰值數量，回傳 bpm。
        若資料長度或時間軸不合法則回傳 np.nan。
        """
        # if len(y_arr) < 4 or len(t_arr) < 2 or (t_arr[-1] - t_arr[0]) <= 0:
        #     return np.nan
        # duration_sec = t_arr[-1] - t_arr[0]
        duration_sec = 20.0
        if duration_sec <= 0:
            return np.nan

        peaks, _ = find_peaks(y_arr, distance=20, height=0.5, prominence=1.0)
        if len(peaks) == 0:
            return np.nan
        bpm = len(peaks) * (60 / duration_sec)
        print(f"peaks:{len(peaks)}")
        print(f"duration_sec:{duration_sec}")
        print(f"bpm:{bpm}")
        return int(bpm)

    def _avg_rr_over_7(self, center_start_sec: float) -> float:
        """
        以目前 window 的起點時間為中心，取前後各 3 個 window（共 7 個），stride = 1 秒，
        每個 window 長 self.window_sec 秒，回傳 7-window 平均呼吸率（bpm）。
        邊界自動縮減到可用的範圍。
        """
        if 'n_process' not in self.filtetered_signals:
            return np.nan

        t_full, y_full = self.filtetered_signals['n_process']
        rr_list = []
        rr_bpm_list = []

        for offset in (range(0, int(center_start_sec) + 1) if center_start_sec < 6 else range(0, 7)):
            # print(f"center_start_sec:{center_start_sec}")
            s = center_start_sec - offset * 1
            e = s + self.window_sec

            # 邊界：需落在 [0, self.max_time] 內
            if s < 0 or e > self.max_time:
                continue

            mask = (t_full >= s) & (t_full < e)
            if not np.any(mask):
                continue

            y_seg = np.asarray(y_full)[mask]
            t_seg = np.asarray(t_full)[mask]


            # rr_bpm = self._rr_from_segment(y_seg, t_seg)
            rr_bpm = self._rr_form_find_peaks(y_seg)
            # print(f"rr_bpm:{rr_bpm}")
            if not np.isnan(rr_bpm):
                rr_list.append((s, rr_bpm))
                rr_bpm_list.append(rr_bpm)

        if len(rr_list) == 0:
            return np.nan
        return (np.mean(rr_bpm_list)), rr_list


    def open_file_dialog(self):
        """Open a file dialog to select a file."""
        fileName, _ = QFileDialog.getOpenFileName(self, 
                                                  "choose csv file",
                                                  os.path.expanduser("D:\\Research\\AI_model\\Respiration_Rate_Analysis_Research\\dataset\\process_data"),
                                                  "CSV Files (*.csv)"
                                                  )
        if fileName:
            print(f"Selected file: {fileName}")
            self.lbl_file_info.setText(f"Selected file: {fileName}")

            try:
                # Load the data
                data = pd.read_csv(fileName, encoding='utf-8-sig', low_memory=False)
                logging.info(f"Data loaded from {fileName}")
                columes = data.columns.tolist()

                logging.info(f"Columns in the data: {columes}")
                # print(f"Columns in the data: {columes}")

                fs128_group = [
                    (['MEAS1_PPG1', 'MEAS1_PPG2', 'MEAS1_PPG3', 'MEAS1_PPG4'], 128),
                    (['ACCX', 'ACCY', 'ACCZ'], 128),
                    (['phase_rx0', 'phase_rx1', 'phase_rx2'], 128),
                ]

                # Apply bandpass filter to each group
                for group, fs in fs128_group:
                    for col in group:
                        if col in data.columns:
                            sig = bandpass_filter(data[col], lowcut=0.1, highcut=1.0, fs=fs, order=4)
                            t = np.arange(len(sig)) / fs  # Assuming fs is the sampling frequency
                            self.filtetered_signals[col] = (t, sig)

                # Calculate ACC magnitude if all three axes are present
                acc_axes = ['ACCX', 'ACCY', 'ACCZ']
                if all(ax in self.filtetered_signals for ax in acc_axes):
                    print("Hello, ACC axes:", acc_axes)
                    # Calculate the magnitude of the accelerometer signal
                    tx, x = self.filtetered_signals['ACCX']
                    ty, y = self.filtetered_signals['ACCY']
                    tz, z = self.filtetered_signals['ACCZ']
                    L = min(len(x), len(y), len(z))

                    x, y, z = np.asarray(x[:L]), np.asarray(y[:L]), np.asarray(z[:L])
                    t_mag = np.asarray(tx[:L])

                    mag = np.sqrt(x*x + y*y + z*z)
                    # print(f"ACC_Magnitude calculated with length {len(mag)}")
                    self.filtetered_signals['ACC_Magnitude'] = (t_mag, mag)

                if "n_process" in data.columns:
                    sig = data['n_process']
                    fs_n = 20.0  # n_process is sampled at 20 Hz
                    t = np.arange(len(sig), dtype=float) / fs_n
                    self.filtetered_signals['n_process'] = (t, sig)


                # Determine the maximum time for the slider
                # self.max_time = max(t[-1] for t, _ in self.filtetered_signals.values()) if self.filtetered_signals else 0
                self.max_time = len(data['phase_rx0']) / 128   # Assuming phase_rx0 is sampled at 128 Hz
                print(f"Max time for slider: {self.max_time} seconds")
                if self.max_time > self.window_size:
                    slider_max = int(np.floor((self.max_time - self.window_size) / self.window_size)) - self.window_sec
                else:
                    slider_max = 0

                self.slider.setMaximum(slider_max)
                self.slider.setValue(0)  # Reset slider to the start
                self.slider.setEnabled(bool(self.max_time > self.window_size))  
                
                # Initial plot
                self.update_plot()

                # show columns in the label
                # self.lbl_file_info.setText(f"Columns in the data: {', '.join(columes)}")
                self.lbl_file_info.setText(f"number of columns: {len(columes)}")
            except Exception as e:
                logging.error(f"Error loading data from {fileName}: {e}")
                self.lbl_file_info.setText(f"Error loading file: {e}")

    def _window_arrays_for_model(self, start_sec: float, end_sec: float):
        """
        Slice current window [start_sec, end_sec) from self.filtetered_signals and
        return (ppg(Cp,T), radar(Cr,T), acc(Ca,T)) arrays ready for model.
        """
        def slice_stack(cols):
            tracks = []
            for col in cols:
                if col not in self.filtetered_signals:
                    continue
                t, y = self.filtetered_signals[col]
                mask = (t >= start_sec) & (t < end_sec)
                y_win = np.asarray(y)[mask]
                tracks.append(y_win)
            if not tracks:
                return None
            # Ensure all tracks are the same length
            min_len = min(len(tr) for tr in tracks)
            tracks = [tr[:min_len] for tr in tracks]
            return np.stack(tracks, axis=0).astype(np.float32)  # (C, T)

        radar_cols = ['phase_rx0', 'phase_rx1', 'phase_rx2']
        ppg_cols   = ['MEAS1_PPG1', 'MEAS1_PPG2', 'MEAS1_PPG3', 'MEAS1_PPG4']
        acc_cols   = ['ACCX', 'ACCY', 'ACCZ', 'ACC_Magnitude']

        radar = slice_stack(radar_cols)
        ppg   = slice_stack(ppg_cols)
        acc   = slice_stack(acc_cols)

        # If all arrays are None, return None
        if radar is None and ppg is None and acc is None:
            return None, None, None

        # Ensure all arrays have the same time length T
        T = max([arr.shape[1] for arr in [radar, ppg, acc] if arr is not None])

        def ensure_T(arr, C_expected):
            if arr is None:
                return np.zeros((C_expected, T), dtype=np.float32)
            C, tlen = arr.shape
            if tlen == T:
                return arr
            if tlen > T:
                return arr[:, :T]
            pad = np.zeros((C, T - tlen), dtype=np.float32)
            return np.concatenate([arr, pad], axis=1)

        radar = ensure_T(radar, 3)
        ppg   = ensure_T(ppg,   4)
        acc   = ensure_T(acc,   4)
        return ppg, radar, acc

    
    def run_model(self, model_path):
        """Run the model on the CURRENT 20-second window and overlay results at bottom-right."""
        start_sec = self.slider.value() * self.window_size
        end_sec   = start_sec + self.window_sec  # 20 seconds window

        #  Get the data for the current window
        ppg, radar, acc = self._window_arrays_for_model(start_sec, end_sec)
        if ppg is None or radar is None or acc is None:
            logging.warning("No data available for the selected window.")
            return None

        # Convert to tensors
        y_true = float(getattr(self, "respiration_rate_find_peaks", 0.0))
        # y_true = float(getattr(self, "respiration_rate_avg7", y_true))

        # Dataset / Loader
        dataset = BreathingDataset(ppg_signal=ppg, radar_signal=radar, acc_signal=acc, respiration_rate=y_true)
        loader  = DataLoader(dataset, batch_size=1, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RadarBreathingModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)           # (B, C, T)： ppg(4, T), radar(3, T), acc(3, T)
                pred = model(x).detach().cpu().numpy().flatten()[0]
                y_true = y.detach().cpu().numpy().flatten()[0]

        mae = float(abs(pred - y_true))
        fft_mae = float(abs(pred - self.respiration_rate)) if not np.isnan(self.respiration_rate) else np.nan
        peak_mae = float(abs(pred - self.respiration_rate_avg7)) if not np.isnan(self.respiration_rate_avg7) else np.nan

        # Update the plot with the prediction

        # Remove previous annotation if exists
        if hasattr(self, "_result_annot") and self._result_annot is not None:
            try:
                self._result_annot.remove()
            except Exception:
                pass
            self._result_annot = None
 
        color = "#111111" if not self.dark_mode else "#f0f0f0"
        # Create a new annotation
        # msg_html = f"True: <b>{y_true:.2f}</b> bpm<br>Pred: <b>{pred:.2f}</b> bpm<br>MAE:&nbsp;<b>{mae:.2f}</b>"
        msg_html = (
            f"Pred: <b>{pred:.2f}</b> bpm<br>"
            f"Find_peak RR: <b>{y_true:.2f}</b> bpm | "
            f"Find_peak RR MAE: <b>{mae:.2f}</b> bpm<br>"
            f"FFT RR: <b>{self.respiration_rate:.2f}</b> bpm | "
            f"FFT MAE: <b>{fft_mae:.2f}</b> bpm<br>"
            f"7-win Avg RR: <b>{self.respiration_rate_avg7:.2f}</b> bpm | "
            f"7-win Avg MAE: <b>{peak_mae:.2f}</b> bpm"
        )
        self.result_label.setText(msg_html)
        self.canvas.draw_idle()

        self.canvas.draw_idle()
        print(f"[{start_sec:.1f}-{end_sec:.1f}s] True={y_true:.2f}, Pred={pred:.2f}, MAE={mae:.2f}")
        logging.info(f"[{start_sec:.1f}-{end_sec:.1f}s] True={y_true:.2f}, Pred={pred:.2f}, MAE={mae:.2f}")
        return pred, y_true, mae

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataVisualizer()
    window.show()
    sys.exit(app.exec())
