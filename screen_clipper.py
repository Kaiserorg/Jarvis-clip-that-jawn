

from collections import deque
import shutil
import os
import sys
import threading
import time
import tempfile
import importlib
import subprocess
from datetime import datetime
import wave
 

import tkinter as tk
from tkinter import ttk, messagebox
import json

try:
    import sv_ttk
    SV_TTK_AVAILABLE = True
except Exception:
    sv_ttk = None
    SV_TTK_AVAILABLE = False

try:
    from PIL import Image, ImageTk
    import mss
    import sounddevice as sd
    import numpy as np
    import cv2
    try:
        import openwakeword as ow
        OPENWAKE_AVAILABLE = True
    except Exception:
        ow = None
        OPENWAKE_AVAILABLE = False
    try:
        import soundcard as sc
        SOUND_CARD_AVAILABLE = True
    except Exception:
        sc = None
        SOUND_CARD_AVAILABLE = False
except ImportError as e:
                                                     
    msg_lines = [f"Missing dependency: {e}",
                 "Please install required packages: mss, sounddevice, numpy, opencv-python, pillow"]
    project_dir = os.path.dirname(__file__)
    venv_python = os.path.join(project_dir, '.venv', 'bin', 'python')
    if os.path.exists(venv_python):
        msg_lines.append(f"If you use the project's virtualenv, run:\n  {venv_python} -m pip install mss sounddevice numpy opencv-python pillow")
    else:
        if os.name == 'nt':
            msg_lines.append("Or install for the current user with:\n  python -m pip install --user mss sounddevice numpy opencv-python pillow")
        else:
            msg_lines.append("Or install for the current user with:\n  python3 -m pip install --user mss sounddevice numpy opencv-python pillow")

    print('\n'.join(msg_lines))
    sys.exit(1)

                 
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    PSUTIL_AVAILABLE = False

APP_TITLE = "Screen Clipper"
PREVIEW_FPS = 12                                           
CAPTURE_FPS = 12
AUDIO_SR = 48000
CHANNELS = 1

CLIP_OPTIONS = [(30, "30 seconds"), (60, "1 minute"), (120, "2 minutes"), (180, "3 minutes")]
AUDIO_SOURCE_OPTIONS = ["Auto", "Mic only", "Output only", "Mic + Output"]


class ScreenClipperApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # load config
        self._load_config()

        # recording / audio buffers / state
        self.recording = False
        self.lock = threading.Lock()
        self._audio_cb_times = deque(maxlen=2000)
        self._audio_watchdog_thread = None
        self._audio_watchdog_stop = threading.Event()
        self._audio_watchdog_state = None
        self._last_audio_log = 0.0

        self.max_buffer_seconds = 120
        self.frames = deque()  # (timestamp, filepath) compressed JPEGs
        self.audio_raw_path = None
        self.audio_total_samples = 0
        self.audio_start_ts = None
        self.audio_index = deque()
        self.sys_audio_raw_path = None
        self.sys_audio_total_samples = 0
        self.sys_audio_start_ts = None
        self.sys_thread = None
        self.sys_audio_index = deque()
        self.sys_indexer_thread = None
        self.record_tmpdir = None
        self._frame_counter = 0

        self.audio_stream = None
        self.wake_buffer = np.array([], dtype=np.float32)

        # mss/sounddevice contexts are created in worker threads
        # Audio source selection
        self.audio_var = tk.StringVar(value='Mic + Output')
        # Theme and accent state (loaded from config in _load_config)
        self.setup_ui()

        try:
            self.geometry('1600x360')
            self.resizable(False, False)
        except Exception:
            pass

        self._psutil_available = PSUTIL_AVAILABLE
        if self._psutil_available:
            try:
                self._proc = psutil.Process(os.getpid())
                try:
                    self._proc.cpu_percent(interval=None)
                except Exception:
                    pass
                
                self.after(1000, self._update_process_stats)
            except Exception:
                self._psutil_available = False

        # Preview image reference
        self.preview_image = None

        # Threads
        self.video_thread = None

        # Status
        self.last_status = "Not recording"
        self._last_status_time = 0.0
        self.update_status("Not recording")

        # Initialize wake detector if enabled (after status exists)
        try:
            if getattr(self, 'wake_enabled', False):
                self._init_wake_detector()
        except Exception:
            pass

        # Check for ffmpeg availability
        self.ffmpeg_available = bool(shutil.which('ffmpeg'))
        if not self.ffmpeg_available:
            self._set_error('ffmpeg not found in PATH; clipping disabled')
            self.clip_btn.state(['disabled'])

        # Preview update loop
        self.after(int(1000 / PREVIEW_FPS), self._preview_loop)

        # Wakeword disabled by default for UI
        self._last_wake_time = 0.0

    def setup_ui(self):
        style = ttk.Style(self)
        font_main = ('Helvetica', 11)

        if SV_TTK_AVAILABLE:
            # sv_ttk theme setup
            try:
                sv_ttk.set_theme(self._theme if getattr(self, '_theme', 'dark') in ('dark', 'light') else 'dark')
            except Exception:
                pass
            # Keep fonts and weight customizations
            style.configure('TLabel', font=font_main)
            style.configure('TMenubutton', font=font_main)
            style.configure('Primary.TButton', font=(font_main[0], 11, 'bold'))
            style.configure('Clip.TButton', font=(font_main[0], 12, 'bold'))
            style.configure('Danger.TButton', font=(font_main[0], 10, 'bold'))
            style.configure('Success.TButton', font=(font_main[0], 10, 'bold'))
            # Fallback palette variables
            bg = '#0f1115'
            panel = '#14161a'
            card = '#1b1d22'
            fg = '#e6eef3'
            accent = '#1e88e5'
            success = '#43a047'
            danger = '#e53935'
        else:
            style.theme_use('clam')

            # Cohesive dark palette (fallback)
            bg = '#0f1115'          # window background
            panel = '#14161a'       # inner panel background
            card = '#1b1d22'        # card / control background
            fg = '#e6eef3'          # foreground text
            accent = '#1e88e5'      # blue accent for primary actions
            success = '#43a047'
            danger = '#e53935'

            # Apply base styles
            style.configure('TFrame', background=bg)
            style.configure('TLabel', background=bg, foreground=fg, font=font_main)
            style.configure('TMenubutton', background=card, foreground=fg)

            # Button styles
            style.configure('Primary.TButton', background=accent, foreground='white', font=(font_main[0], 11, 'bold'))
            style.map('Primary.TButton', background=[('active', '#1669b8')])
            style.configure('Success.TButton', background=success, foreground='white', font=(font_main[0], 10, 'bold'))
            style.map('Success.TButton', background=[('active', '#2e7d32')])
            style.configure('Danger.TButton', background=danger, foreground='white', font=(font_main[0], 10, 'bold'))
            style.map('Danger.TButton', background=[('active', '#b71c1c')])
            style.configure('Clip.TButton', background=accent, foreground='white', font=(font_main[0], 12, 'bold'))
            style.map('Clip.TButton', background=[('active', '#1669b8')])

        # UI layout and controls
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=12, pady=10)

        # left controls
        left_controls = ttk.Frame(top)
        left_controls.pack(side=tk.LEFT)

        self.toggle_btn = ttk.Button(left_controls, text='Start', style='Primary.TButton', command=self.toggle_recording)
        self.toggle_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.clip_btn = ttk.Button(left_controls, text='Clip Now', style='Clip.TButton', command=self.clip_now)
        self.clip_btn.pack(side=tk.LEFT)
        self.clip_btn.state(['disabled'])

        # middle controls
        mid_controls = ttk.Frame(top)
        mid_controls.pack(side=tk.LEFT, padx=16)

        # accent selector
        self._accent_var = tk.StringVar(value=self._accent_name)
        accent_sel = ttk.Combobox(mid_controls, textvariable=self._accent_var, values=['Orange', 'Green', 'Blue'], width=8)
        accent_sel.pack(side=tk.LEFT, padx=(4, 12))
        accent_sel.bind('<<ComboboxSelected>>', lambda *_: self._apply_accent(self._accent_var.get()))

        self.clip_var = tk.StringVar(value='2 minutes')
        options = [label for _, label in CLIP_OPTIONS]
        self.dropdown = ttk.OptionMenu(mid_controls, self.clip_var, self.clip_var.get(), *options, command=self._on_dropdown_change)
        self.dropdown.pack(side=tk.LEFT, padx=(0, 10))

        audio_opts = AUDIO_SOURCE_OPTIONS
        self.audio_dropdown = ttk.OptionMenu(mid_controls, self.audio_var, self.audio_var.get(), *audio_opts)
        self.audio_dropdown.pack(side=tk.LEFT)

        # removed blocksize selector

        # right controls (gain)
        right_controls = ttk.Frame(top)
        right_controls.pack(side=tk.RIGHT)

        mic_frame = ttk.Frame(right_controls)
        mic_frame.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(mic_frame, text='Mic').pack(side=tk.LEFT)
        self.mic_minus = ttk.Button(mic_frame, text='−', width=2, command=lambda: self._change_mic_gain(-5))
        self.mic_minus.pack(side=tk.LEFT, padx=(6, 0))
        self.mic_gain_label = ttk.Label(mic_frame, text=f'{int(self.mic_gain)}%')
        self.mic_gain_label.pack(side=tk.LEFT, padx=(6, 0))
        self.mic_plus = ttk.Button(mic_frame, text='+', width=2, command=lambda: self._change_mic_gain(5))
        self.mic_plus.pack(side=tk.LEFT, padx=(6, 0))

        sys_frame = ttk.Frame(right_controls)
        sys_frame.pack(side=tk.LEFT)
        ttk.Label(sys_frame, text='PC').pack(side=tk.LEFT)
        self.sys_minus = ttk.Button(sys_frame, text='−', width=2, command=lambda: self._change_sys_gain(-5))
        self.sys_minus.pack(side=tk.LEFT, padx=(6, 0))
        self.sys_gain_label = ttk.Label(sys_frame, text=f'{int(self.sys_gain)}%')
        self.sys_gain_label.pack(side=tk.LEFT, padx=(6, 0))
        self.sys_plus = ttk.Button(sys_frame, text='+', width=2, command=lambda: self._change_sys_gain(5))
        self.sys_plus.pack(side=tk.LEFT, padx=(6, 0))

        # spacer
        top_spacer = ttk.Frame(top)
        top_spacer.pack(side=tk.LEFT, expand=True)

        # Main area: preview + sidebar
        mid = ttk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        # preview area
        preview_frame = tk.Frame(mid, bg=card, width=1000, height=500)
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 12))
        preview_frame.pack_propagate(False)
        self.preview_label = tk.Label(preview_frame, text='No preview', bg=card, fg=fg, font=(font_main[0], 12))
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        self.preview_frame = preview_frame

        # sidebar
        if SV_TTK_AVAILABLE:
            # sidebar sizing
            stats_frame = ttk.Frame(mid, width=300, height=500)
            stats_frame.pack(side=tk.RIGHT, fill=tk.Y)
            stats_frame.pack_propagate(False)

            sb_title = ttk.Label(stats_frame, text='USED BY PROCESS', font=(font_main[0], 11, 'bold'))
            sb_title.pack(pady=(12, 6))
            self.cpu_var = tk.StringVar(value='CPU: 0%')
            self.mem_var = tk.StringVar(value='RAM: 0 MB')
            ttk.Label(stats_frame, textvariable=self.cpu_var, font=font_main).pack(pady=(8, 4))
            ttk.Label(stats_frame, textvariable=self.mem_var, font=font_main).pack(pady=(2, 8))

            # Wake controls
            self.wake_var = tk.BooleanVar(value=self.wake_enabled)
            self.wake_clip_var = tk.BooleanVar(value=self.wake_clip_on_detect)
            wake_chk = ttk.Checkbutton(stats_frame, text='Wake', variable=self.wake_var, command=self._on_wake_toggle)
            wake_chk.pack(pady=(6, 2))
            wake_clip_chk = ttk.Checkbutton(stats_frame, text='Clip on detect', variable=self.wake_clip_var, command=self._on_wake_clip_toggle)
            wake_clip_chk.pack(pady=(0, 6))
            ttk.Button(stats_frame, text='Wake test', command=self._on_wake_test).pack(pady=(2, 8))

            # footer hint
            self.hint_var = tk.StringVar(value='Status: Not recording')
            ttk.Label(stats_frame, textvariable=self.hint_var, font=(font_main[0], 10)).pack(side=tk.BOTTOM, pady=12)
        else:
            # fallback sidebar sizing
            stats_frame = tk.Frame(mid, bg=panel, width=300, height=280)
            stats_frame.pack(side=tk.RIGHT, fill=tk.Y)
            stats_frame.pack_propagate(False)

            sb_title = tk.Label(stats_frame, text='USED BY PROCESS', bg=panel, fg=fg, font=(font_main[0], 11, 'bold'))
            sb_title.pack(pady=(12, 6))
            self.cpu_var = tk.StringVar(value='CPU: 0%')
            self.mem_var = tk.StringVar(value='RAM: 0 MB')
            tk.Label(stats_frame, textvariable=self.cpu_var, bg=panel, fg=fg, font=font_main).pack(pady=(8, 4))
            tk.Label(stats_frame, textvariable=self.mem_var, bg=panel, fg=fg, font=font_main).pack(pady=(2, 8))

            # footer hint
            self.hint_var = tk.StringVar(value='Status: Not recording')
            tk.Label(stats_frame, textvariable=self.hint_var, bg=panel, fg=fg, font=(font_main[0], 10)).pack(side=tk.BOTTOM, pady=12)

            # Wake controls (fallback)
            self.wake_var = tk.BooleanVar(value=self.wake_enabled)
            self.wake_clip_var = tk.BooleanVar(value=self.wake_clip_on_detect)
            tk.Checkbutton(stats_frame, text='Wake', variable=self.wake_var, command=self._on_wake_toggle, bg=panel, fg=fg).pack(pady=(6, 2))
            tk.Checkbutton(stats_frame, text='Clip on detect', variable=self.wake_clip_var, command=self._on_wake_clip_toggle, bg=panel, fg=fg).pack(pady=(0, 6))
            tk.Button(stats_frame, text='Wake test', command=self._on_wake_test).pack(pady=(2, 8))

    # expose widgets
        self.stats_frame = stats_frame
        self.sb_title = sb_title

    # status label
        self.status_var = tk.StringVar(value='Not recording')
        if SV_TTK_AVAILABLE:
            style.configure('Status.TLabel', font=(font_main[0], 10, 'bold'), foreground='white')
        else:
            style.configure('Status.TLabel', background=bg, foreground=fg, font=(font_main[0], 10, 'bold'))

    # theme toggle
        right_controls = ttk.Frame(top)
        right_controls.pack(side=tk.RIGHT)
        self._dark_var = tk.BooleanVar(value=(self._theme == 'dark'))
        theme_chk = ttk.Checkbutton(right_controls, text='Dark', variable=self._dark_var, command=self._toggle_theme)
        theme_chk.pack(side=tk.LEFT)

        # Apply initial accent
        self._apply_accent(self._accent_name)
        self.status_label = ttk.Label(self, textvariable=self.status_var, style='Status.TLabel', anchor=tk.CENTER)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(8, 12))

        # ensure theme applied
        try:
            self._toggle_theme()
        except Exception:
            pass
        # (wake detector will be initialized after setup when app state exists)
        # psutil fallback
        if not getattr(self, '_psutil_available', False):
            self.cpu_var.set('CPU: N/A')
            self.mem_var.set('RAM: N/A')

    def _update_process_stats(self):
        # Periodically update CPU and RAM usage for the running Python process only
        try:
            if not getattr(self, '_psutil_available', False):
                # nothing to do; values already set to N/A
                return
            p = getattr(self, '_proc', None)
            if p is None:
                return
            try:
                cpu = p.cpu_percent(interval=None)
            except Exception:
                cpu = 0.0
            try:
                mem = p.memory_info().rss / (1024.0 * 1024.0)
            except Exception:
                mem = 0.0
            try:
                # Show CPU usage for this process only. psutil reports per-process
                # percentage relative to a single CPU, which can exceed 100 on
                # multi-core machines. Display the raw percent and an equivalent
                # "cores used" value so it's clear this is process-local.
                cores_used = cpu / 100.0
                self.cpu_var.set(f'CPU: {cpu:.1f}% (~{cores_used:.2f} cores)')
                self.mem_var.set(f'RAM: {mem:.1f} MB')
            except Exception:
                pass
        finally:
            # schedule next update
            try:
                self.after(1000, self._update_process_stats)
            except Exception:
                pass

    def _on_dropdown_change(self, value):
        # Set max buffer duration according to dropdown
        for seconds, label in CLIP_OPTIONS:
            if label == value:
                with self.lock:
                    self.max_buffer_seconds = seconds
                break

    def _on_blocksize_change(self, value):
        # Blocksize selector was removed from the UI; keep compatibility if set programmatically.
        try:
            self.audio_blocksize = int(value)
            self._save_config()
            self.update_status(f'Audio blocksize set: {self.audio_blocksize}')
        except Exception:
            pass

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        # Prepare buffers
        with self.lock:
            # create a fresh temp dir for recording files
            try:
                if self.record_tmpdir and os.path.exists(self.record_tmpdir):
                    for f in os.listdir(self.record_tmpdir):
                        try:
                            os.remove(os.path.join(self.record_tmpdir, f))
                        except Exception:
                            pass
                    os.rmdir(self.record_tmpdir)
            except Exception:
                pass

            self.record_tmpdir = tempfile.mkdtemp(prefix='screen_clipper_')
            self.frames.clear()
            self.audio_raw_path = os.path.join(self.record_tmpdir, 'audio.raw')
            # create an empty raw audio file
            open(self.audio_raw_path, 'wb').close()
            self.audio_total_samples = 0
            self.audio_start_ts = time.time()
            # clear per-block indexes
            self.audio_index.clear()
            # Setup system audio according to the chosen audio source option
            sel = self.audio_var.get() if hasattr(self, 'audio_var') else 'Auto'
            self.sys_audio_raw_path = None
            if sel in ('Output only', 'Mic + Output', 'Auto'):
                # prepare sys audio file path
                self.sys_audio_raw_path = os.path.join(self.record_tmpdir, 'sys_audio.raw')
                open(self.sys_audio_raw_path, 'wb').close()
                self.sys_audio_total_samples = 0
                self.sys_audio_start_ts = time.time()
                self.sys_audio_index.clear()
                started = False

                # If user explicitly requested Output, prefer soundcard capture (more direct) then ffmpeg
                if sel in ('Output only', 'Mic + Output'):
                    # attempt dynamic import if available now
                    if not SOUND_CARD_AVAILABLE:
                        try:
                            sc_mod = importlib.import_module('soundcard')
                            globals()['sc'] = sc_mod
                            globals()['SOUND_CARD_AVAILABLE'] = True
                        except Exception:
                            pass

                    if SOUND_CARD_AVAILABLE:
                        # start soundcard-based capture thread
                        try:
                            self.sys_thread = threading.Thread(target=self._sys_audio_worker_soundcard, daemon=True)
                            self.sys_thread.start()
                            started = True
                            self.update_status('System audio capture started via soundcard (output)')
                        except Exception:
                            started = False
                    if not started:
                        started = self._start_ffmpeg_sys_capture()
                        if started:
                            self.update_status('System audio capture started via ffmpeg (output)')

                # If still not started, try sounddevice monitor source detection (Auto or fallback)
                if not started:
                    dev = self._find_system_monitor_device()
                    if dev is not None:
                        idx, name = dev
                        self.sys_thread = threading.Thread(target=self._sys_audio_worker, args=(idx,), daemon=True)
                        self.sys_thread.start()
                        self.update_status(f"System audio capture started: {name}")
                    else:
                        # Try external recorder fallback
                        got_proc = self._try_start_monitor_subprocess()
                        if got_proc:
                            self.update_status("System audio capture started via external recorder")
                        else:
                            # If user explicitly wanted output only, warn; otherwise silently proceed without system audio
                            if sel in ('Output only',):
                                self._set_error('System audio capture not available for the selected option')
                            else:
                                self.update_status("System audio capture not found (desktop audio will not be recorded)")
            self._frame_counter = 0

        # Start recording threads (audio + video) so each can manage their resources
        self.recording = True

        # Start audio thread that opens the InputStream in-thread
        # Start mic audio only if selected
        sel = self.audio_var.get() if hasattr(self, 'audio_var') else 'Auto'
        if sel in ('Mic only', 'Mic + Output', 'Auto'):
            self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
            self.audio_thread.start()
            # start audio watchdog thread
            try:
                if not self._audio_watchdog_thread or not self._audio_watchdog_thread.is_alive():
                    self._audio_watchdog_stop.clear()
                    self._audio_watchdog_thread = threading.Thread(target=self._audio_watchdog_worker, daemon=True)
                    self._audio_watchdog_thread.start()
            except Exception:
                pass

        # Start video thread
        self.video_thread = threading.Thread(target=self._video_worker, daemon=True)
        self.video_thread.start()

        # UI changes
        self.toggle_btn.config(text='Stop Recording', style='Danger.TButton')
        if self.ffmpeg_available:
            self.clip_btn.state(['!disabled'])
        self.update_status('Recording active – Buffer: 0:00 / ' + self._format_seconds(self.max_buffer_seconds))

    def stop_recording(self):
        self.recording = False
        # Stop audio: audio thread will close stream when it observes self.recording == False
        try:
            if hasattr(self, 'audio_thread') and self.audio_thread is not None:
                self.audio_thread.join(timeout=1.0)
                self.audio_thread = None
        except Exception:
            pass

        # Wait for video thread to finish
        if self.video_thread is not None:
            self.video_thread.join(timeout=1.0)
            self.video_thread = None

        self.toggle_btn.config(text='Start Recording', style='Start.TButton')
        self.clip_btn.state(['disabled'])
        self.update_status('Not recording')

        # Clean up recording temp dir and files
        if self.record_tmpdir and os.path.exists(self.record_tmpdir):
            # remove frame files
            for _, path in list(self.frames):
                self._safe_remove(path)
            # remove raw audio files
            self._safe_remove(self.audio_raw_path)
            self._safe_remove(self.sys_audio_raw_path)
            self._safe_rmdir(self.record_tmpdir)

        # Reset recording-specific state
        self.record_tmpdir = None
        self.sys_audio_raw_path = None
        self.sys_audio_total_samples = 0
        self.sys_audio_start_ts = None
        self.sys_thread = None

        # No platform-specific virtual-sink cleanup required here

        # Terminate/close any external capture processes or files
        self._safe_terminate(getattr(self, 'sys_capture_proc', None))
        self._safe_close(getattr(self, 'sys_capture_file', None))
        self.sys_capture_proc = None
        self.sys_capture_file = None
        # stop audio watchdog
        try:
            self._audio_watchdog_stop.set()
            if self._audio_watchdog_thread:
                self._audio_watchdog_thread.join(timeout=0.5)
        except Exception:
            pass

    def _video_worker(self):
        target_dt = 1.0 / CAPTURE_FPS
        # Create mss inside this thread to ensure display context is correct
        try:
            sct = mss.mss()
            monitor = sct.monitors[1]
        except Exception as e:
            self._set_error(f"Screen capture init failed: {e}")
            return

        while self.recording:
            t0 = time.time()
            try:
                img = sct.grab(monitor)
            except Exception as e:
                self._set_error(f"Screen capture failed: {e}")
                break

            arr = np.array(img)  # BGRA or RGB depending on platform
            # Normalize to BGR for cv2
            if arr.shape[2] == 4:
                frame_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            else:
                # assume RGB
                frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            ts = time.time()
            # Compress frame to JPEG and write to temp file to avoid keeping raw frames in RAM
            try:
                ret, buf = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                if not ret:
                    raise RuntimeError('JPEG encode failed')
                filename = os.path.join(self.record_tmpdir, f'frame_{int(ts*1000)}_{self._frame_counter:06d}.jpg')
                tmpname = filename + '.tmp'
                # write to temp file and atomically replace to avoid readers seeing partial file
                with open(tmpname, 'wb') as f:
                    f.write(buf.tobytes())
                    f.flush()
                    os.fsync(f.fileno())
                try:
                    os.replace(tmpname, filename)
                except Exception:
                    # fallback to non-atomic write
                    os.remove(tmpname)
                    with open(filename, 'wb') as f:
                        f.write(buf.tobytes())
                self._frame_counter += 1
                with self.lock:
                    self.frames.append((ts, filename))
                    # purge old frames and files
                    self._trim_buffers_locked()
            except Exception as e:
                self._set_error(f"Frame save failed: {e}")

            # update status buffer progress
            with self.lock:
                buffer_seconds = self._buffer_length_seconds_locked()
            if buffer_seconds < self.max_buffer_seconds:
                # show buffering progress until full
                self.update_status(f"Buffering... {int(buffer_seconds)}s ready")
            else:
                self.update_status(f"Recording active – Buffer: {self._format_seconds(buffer_seconds)} / {self._format_seconds(self.max_buffer_seconds)}")

            elapsed = time.time() - t0
            to_sleep = max(0.0, target_dt - elapsed)
            time.sleep(to_sleep)

        # cleanup mss
        if 'sct' in locals():
            del sct

    def _audio_worker(self):
        # Open InputStream inside this thread to keep audio device context local
        try:
            bs = int(getattr(self, 'audio_blocksize', 0)) or None
            try:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Audio: starting InputStream blocksize={bs}", flush=True)
            except Exception:
                pass
            if bs:
                with sd.InputStream(samplerate=AUDIO_SR, channels=CHANNELS, callback=self._audio_callback, blocksize=bs) as stream:
                    self.audio_stream = stream
                    # Keep the stream running until recording stops
                    while self.recording:
                        time.sleep(0.1)
            else:
                with sd.InputStream(samplerate=AUDIO_SR, channels=CHANNELS, callback=self._audio_callback) as stream:
                    self.audio_stream = stream
                    # Keep the stream running until recording stops
                    while self.recording:
                        time.sleep(0.1)
        except Exception as e:
            self._set_error(f"Audio input failed: {e}")
        finally:
            self.audio_stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            # Post status to GUI
            self._set_error(f"Audio status: {status}")
        # record callback timestamp for watchdog
        try:
            now = time.time()
            self._audio_cb_times.append(now)
        except Exception:
            pass
        # Convert float32 to int16 and append raw bytes to audio file to avoid keeping audio in memory
        try:
            arr = indata.copy()
            # compute simple RMS for diagnostics
            try:
                if arr.size > 0:
                    rms = float(np.sqrt((arr.astype(np.float32) ** 2).mean()))
                else:
                    rms = 0.0
            except Exception:
                rms = 0.0
            # occasionally log diagnostics (throttled to ~1s)
            try:
                if (now - getattr(self, '_last_audio_log', 0)) > 1.0:
                    self._last_audio_log = now
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AudioCB: frames={frames} rms={rms:.6f} total_samples={self.audio_total_samples}", flush=True)
            except Exception:
                pass
            int_samples = np.int16(np.clip(arr, -1.0, 1.0) * 32767)
            # determine timestamp for first sample in this block
            try:
                t = None
                if isinstance(time_info, dict):
                    t = time_info.get('input_buffer_adc_time') or time_info.get('input_stream_time') or time_info.get('current_time')
                if t is None:
                    t = time.time()
            except Exception:
                t = time.time()

            with open(self.audio_raw_path, 'ab') as f:
                f.write(int_samples.tobytes())

            with self.lock:
                start_sample = self.audio_total_samples
                try:
                    self.audio_index.append((start_sample, t))
                except Exception:
                    pass
                self.audio_total_samples += int_samples.shape[0]
                # Trim frames by time (files) and audio if needed
                self._trim_buffers_locked()
                self._trim_audio_files_locked()
                # prune old index entries
                try:
                    cutoff = time.time() - (self.max_buffer_seconds + 1.0)
                    while self.audio_index and self.audio_index[0][1] < cutoff:
                        self.audio_index.popleft()
                except Exception:
                    pass
            # Feed wakeword detector from mic samples if enabled
            try:
                if getattr(self, 'wake_var', None) and self.wake_var.get() and getattr(self, 'wake_detector', None):
                    # indata is float32 shape (frames, channels)
                    samples = arr.reshape(-1)
                    target_sr = getattr(self, 'wake_detector_sr', AUDIO_SR)
                    if target_sr != AUDIO_SR:
                        if AUDIO_SR % target_sr == 0:
                            dec = AUDIO_SR // target_sr
                            proc_samples = samples[::dec]
                        else:
                            old_n = len(samples)
                            new_n = int(round(old_n * target_sr / AUDIO_SR))
                            proc_samples = np.interp(np.linspace(0, old_n, new_n, endpoint=False), np.arange(old_n), samples).astype(np.float32)
                    else:
                        proc_samples = samples

                    det = self.wake_detector
                    hit = False
                    try:
                        # Accumulate into a streaming buffer at detector sample rate and process in 1280-sample frames
                        # proc_samples is already at target_sr
                        if proc_samples.dtype != np.float32:
                            proc = proc_samples.astype(np.float32)
                        else:
                            proc = proc_samples
                        # append
                        try:
                            self.wake_buffer = np.concatenate((self.wake_buffer, proc))
                        except Exception:
                            # fallback: recreate small arrays
                            self.wake_buffer = np.append(self.wake_buffer, proc)

                        # cap buffer to 10s to avoid unbounded growth
                        max_keep = int(target_sr * 10)
                        if self.wake_buffer.shape[0] > max_keep:
                            self.wake_buffer = self.wake_buffer[-max_keep:]

                        FRAME = 1280
                        while self.wake_buffer.shape[0] >= FRAME and not hit:
                            frame = self.wake_buffer[:FRAME]
                            # Try the APIs on this frame
                            try:
                                if hasattr(det, 'accept_waveform'):
                                    try:
                                        r = det.accept_waveform(target_sr, frame)
                                        hit = bool(r)
                                    except TypeError:
                                        try:
                                            r = det.accept_waveform(frame.tobytes())
                                            hit = bool(r)
                                        except Exception:
                                            pass
                                elif hasattr(det, 'process'):
                                    try:
                                        r = det.process(frame)
                                        hit = bool(r)
                                    except Exception:
                                        pass
                                elif hasattr(det, 'run'):
                                    try:
                                        r = det.run(frame)
                                        hit = bool(r)
                                    except Exception:
                                        pass
                                elif hasattr(det, 'detect'):
                                    try:
                                        r = det.detect(frame)
                                        hit = bool(r)
                                    except Exception:
                                        pass
                                elif hasattr(det, 'predict'):
                                    try:
                                        # convert float frame to int16 PCM for predict
                                        if frame.dtype != np.int16:
                                            frame_int = (frame * 32767.0).astype(np.int16)
                                        else:
                                            frame_int = frame
                                        preds = det.predict(frame_int)
                                        if isinstance(preds, dict):
                                            maxscore = max(preds.values()) if preds.values() else 0.0
                                            hit = bool(maxscore >= 0.5)
                                        else:
                                            hit = bool(preds)
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                            # drop processed frame (non-overlapping). Overlap can be added later if needed.
                            self.wake_buffer = self.wake_buffer[FRAME:]

                    except Exception:
                        hit = False

                    if hit:
                        now = time.time()
                        try:
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Wake: live detected", flush=True)
                        except Exception:
                            pass
                        if (now - getattr(self, '_last_wake_time', 0)) > 2.0:
                            self._last_wake_time = now
                            try:
                                self.after(0, self._on_wake_detected)
                            except Exception:
                                pass
            except Exception:
                pass
        except Exception:
            self._set_error("Audio write failed")

    def _sys_audio_worker(self, device_index):
        try:
            with sd.InputStream(samplerate=AUDIO_SR, channels=CHANNELS, device=device_index, callback=self._sys_audio_callback):
                while self.recording:
                    time.sleep(0.1)
        except Exception:
            self._set_error('System audio capture failed')

    def _sys_audio_callback(self, indata, frames, time_info, status):
        if status:
            self._set_error(f"System audio status: {status}")
        try:
            arr = indata.copy()
            int_samples = np.int16(np.clip(arr, -1.0, 1.0) * 32767)
            if self.sys_audio_raw_path:
                with open(self.sys_audio_raw_path, 'ab') as f:
                    f.write(int_samples.tobytes())
            with self.lock:
                start_sample = self.sys_audio_total_samples
                try:
                    ts = None
                    if isinstance(time_info, dict):
                        ts = time_info.get('input_buffer_adc_time') or time_info.get('input_stream_time') or time_info.get('current_time')
                    if ts is None:
                        ts = time.time()
                except Exception:
                    ts = time.time()
                try:
                    self.sys_audio_index.append((start_sample, ts))
                except Exception:
                    pass
                self.sys_audio_total_samples += int_samples.shape[0]
                self._trim_audio_files_locked()
                try:
                    cutoff = time.time() - (self.max_buffer_seconds + 1.0)
                    while self.sys_audio_index and self.sys_audio_index[0][1] < cutoff:
                        self.sys_audio_index.popleft()
                except Exception:
                    pass
        except Exception:
            self._set_error('System audio write failed')

    def _find_system_monitor_device(self):
        # Try to pick a reasonable "monitor" device name (PipeWire often exposes monitor sources)
        try:
            devs = sd.query_devices()
            for i, d in enumerate(devs):
                name = d.get('name', '').lower()
                if 'monitor' in name or 'stereo mix' in name or 'loopback' in name:
                    # confirm it's an input
                    if d.get('max_input_channels', 0) > 0:
                        return i, d.get('name', '')
        except Exception:
            return None
        # Fallback: try to use the default sink's monitor (works when pulseaudio-compatible tools expose sinks)
        try:
            if shutil.which('pactl') is None:
                raise RuntimeError('pactl not available')
            out = subprocess.check_output(['pactl', 'info'], stderr=subprocess.DEVNULL, text=True)
            for line in out.splitlines():
                if line.startswith('Default Sink:'):
                    default_sink = line.split(':', 1)[1].strip().lower()
                    # look for a source that mentions the sink name (monitor flavor)
                    for i, d in enumerate(sd.query_devices()):
                        name = d.get('name', '').lower()
                        if default_sink in name and 'monitor' in name and d.get('max_input_channels', 0) > 0:
                            return i, d.get('name', '')
                    break
        except Exception:
            pass

        return None

    def _try_start_monitor_subprocess(self):
        # Look for a monitor source name via platform-specific helpers and try to spawn a recorder subprocess.
        # On Windows, prefer WASAPI or dshow ffmpeg capture; on Linux attempt PulseAudio monitor sources.
        candidates = []
        if os.name == 'nt':
            # Try WASAPI default capture (loopback may be supported as 'default') and common dshow virtual capture
            candidates.append(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'wasapi', '-i', 'default', '-ac', str(CHANNELS), '-ar', str(AUDIO_SR), '-f', 's16le', '-'])
            candidates.append(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'dshow', '-i', 'audio=virtual-audio-capturer', '-ac', str(CHANNELS), '-ar', str(AUDIO_SR), '-f', 's16le', '-'])
        else:
            # Try to find a monitor source via pactl (PulseAudio / PipeWire), otherwise fall back to ffmpeg pulse default
            monitor_name = None
            try:
                if shutil.which('pactl'):
                    out = subprocess.check_output(['pactl', 'list', 'sources', 'short'], stderr=subprocess.DEVNULL, text=True)
                    for line in out.splitlines():
                        parts = line.split()
                        if len(parts) >= 2:
                            name = parts[1]
                            if '.monitor' in name or 'monitor' in name:
                                monitor_name = name
                                break
            except Exception:
                monitor_name = None

            src = monitor_name if monitor_name else 'default'
            candidates.append(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'pulse', '-i', src, '-ac', str(CHANNELS), '-ar', str(AUDIO_SR), '-f', 's16le', '-'])
            candidates.append(['parec', '-d', monitor_name, '--rate=48000', '--format=s16le', '--channels=1'])
            candidates.append(['pw-record', '-d', monitor_name, '--rate=48000', '--format', 's16', '--channels', '1'])

        for cmd in candidates:
            try:
                # close any previous capture file handle
                try:
                    if hasattr(self, 'sys_capture_file') and self.sys_capture_file:
                        try:
                            self.sys_capture_file.close()
                        except Exception:
                            pass
                except Exception:
                    pass

                f = open(self.sys_audio_raw_path, 'wb')
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.DEVNULL)
                # simple check: give it a moment to start
                time.sleep(0.5)
                if proc.poll() is None:
                    self.sys_capture_proc = proc
                    self.sys_capture_file = f
                    # record a start timestamp reference
                    self.sys_audio_start_ts = time.time()
                    # Start a lightweight indexer thread to timestamp file growth (coarse but avoids changing capture method)
                    try:
                        if not self.sys_indexer_thread or not self.sys_indexer_thread.is_alive():
                            self.sys_indexer_thread = threading.Thread(target=self._sys_indexer_worker, daemon=True)
                            self.sys_indexer_thread.start()
                    except Exception:
                        pass
                    try:
                        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(f"[{ts}] Started external recorder: {' '.join(cmd[:2])}", flush=True)
                    except Exception:
                        pass
                    return True
                else:
                    try:
                        f.close()
                    except Exception:
                        pass
            except Exception:
                continue
        return False

    def _sys_audio_worker_soundcard(self):
        # Use the 'soundcard' library to record from a loopback microphone (monitor) if available
        try:
            # prefer an explicit loopback microphone
            mic = None
            try:
                mic = sc.get_microphone(include_loopback=True)
            except Exception:
                mic = None
            # if the default call didn't give a loopback mic, try to find any mic marked as loopback
            if mic is None or not getattr(mic, 'isloopback', False):
                for cand in sc.all_microphones(include_loopback=True):
                    if getattr(cand, 'isloopback', False) or 'monitor' in cand.name.lower() or 'monitor' in cand.id.lower():
                        mic = cand
                        break
            if mic is None:
                # as last resort try to get a mic matching the default speaker name
                try:
                    speaker = sc.default_speaker()
                    mic = sc.get_microphone(speaker.name, include_loopback=True)
                except Exception:
                    mic = None

            if mic is None:
                self._set_error('No loopback/monitor microphone found via soundcard')
                return
        except Exception as e:
            self._set_error(f"soundcard init failed: {e}")
            return
        # Use a larger chunk and keep file handle open to avoid per-write overhead
        chunk = 4096
        try:
            # open file for append and keep handle for duration of capture
            f = open(self.sys_audio_raw_path, 'ab')
            self.sys_capture_file = f
            try:
                with mic.recorder(samplerate=AUDIO_SR, channels=CHANNELS) as rec:
                    while self.recording:
                        try:
                            data = rec.record(chunk)
                            # data is float32 in [-1,1]; convert to int16
                            int_samples = np.int16(np.clip(data, -1.0, 1.0) * 32767)
                            # record sample offset and timestamp for this chunk
                            with self.lock:
                                start_sample = self.sys_audio_total_samples
                                try:
                                    self.sys_audio_index.append((start_sample, time.time()))
                                except Exception:
                                    pass
                                self.sys_audio_total_samples += int_samples.shape[0]
                                # prune old entries
                                try:
                                    cutoff = time.time() - (self.max_buffer_seconds + 1.0)
                                    while self.sys_audio_index and self.sys_audio_index[0][1] < cutoff:
                                        self.sys_audio_index.popleft()
                                except Exception:
                                    pass
                            f.write(int_samples.tobytes())
                            f.flush()
                            with self.lock:
                                self._trim_audio_files_locked()
                        except Exception:
                            # on error, continue to avoid stopping capture
                            time.sleep(0.01)
            finally:
                try:
                    f.close()
                except Exception:
                    pass
                self.sys_capture_file = None
        except Exception as e:
            self._set_error(f"soundcard capture failed: {e}")

    def _audio_watchdog_worker(self):
        # Monitor recent audio callback timestamps and report if callbacks stop arriving
        last_state = None
        while not self._audio_watchdog_stop.is_set():
            try:
                now = time.time()
                if self._audio_cb_times:
                    last_ts = self._audio_cb_times[-1]
                    gap = now - last_ts
                else:
                    gap = float('inf')

                if gap > 0.6:
                    # considered missing
                    if last_state != 'missing':
                        last_state = 'missing'
                        try:
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AudioWatchdog: missing callbacks (gap={gap:.3f}s)", flush=True)
                        except Exception:
                            pass
                        try:
                            self.update_status('Mic audio missing')
                        except Exception:
                            pass
                else:
                    if last_state != 'ok':
                        last_state = 'ok'
                        try:
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AudioWatchdog: callbacks healthy (gap={gap:.3f}s)", flush=True)
                        except Exception:
                            pass
                        try:
                            # restore normal recording status if appropriate
                            if self.recording:
                                self.update_status('Recording active – Buffer: ' + self._format_seconds(self._buffer_length_seconds_locked()) + ' / ' + self._format_seconds(self.max_buffer_seconds))
                        except Exception:
                            pass
            except Exception:
                pass
            # check every 200ms
            time.sleep(0.2)

    # Virtual sink helpers removed: prefer using soundcard / platform-specific capture.

    # Virtual sink creation removed (platform-specific and fragile).

    # _install_soundcard helper removed - prefer manual installation by the user.

    # _remove_virtual_sink removed.

    def _start_ffmpeg_sys_capture(self, source='default'):
        # Start ffmpeg capture from the default pulse sink monitor into the sys_audio_raw_path
        if not self.ffmpeg_available:
            return False
        try:
            # Choose a platform-appropriate ffmpeg input format
            if os.name == 'nt':
                # Try WASAPI for Windows
                cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'wasapi', '-i', source if source != 'default' else 'default', '-ac', str(CHANNELS), '-ar', str(AUDIO_SR), '-f', 's16le', '-']
            else:
                cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'pulse', '-i', source, '-ac', str(CHANNELS), '-ar', str(AUDIO_SR), '-f', 's16le', '-']
            f = open(self.sys_audio_raw_path, 'wb')
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.DEVNULL)
            time.sleep(0.5)
            if proc.poll() is None:
                self.sys_capture_proc = proc
                self.sys_capture_file = f
                self.sys_audio_start_ts = time.time()
                # Start sys indexer to timestamp file growth
                try:
                    if not self.sys_indexer_thread or not self.sys_indexer_thread.is_alive():
                        self.sys_indexer_thread = threading.Thread(target=self._sys_indexer_worker, daemon=True)
                        self.sys_indexer_thread.start()
                except Exception:
                    pass
                try:
                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{ts}] Started ffmpeg sys capture", flush=True)
                except Exception:
                    pass
                return True
            else:
                try:
                    f.close()
                except Exception:
                    pass
                return False
        except Exception:
            return False

    def _sys_indexer_worker(self):
        # Periodically sample the size of the sys audio raw file and record sample offsets with timestamps.
        bytes_per_sample = CHANNELS * 2
        last_samples = 0
        min_growth = 512  # avoid indexing tiny growths
        while self.recording and getattr(self, 'sys_audio_raw_path', None):
            try:
                if os.path.exists(self.sys_audio_raw_path):
                    sz = os.path.getsize(self.sys_audio_raw_path)
                    samples = sz // bytes_per_sample
                    if samples < last_samples:
                        # file was truncated/rewritten --- reset index
                        last_samples = samples
                        with self.lock:
                            self.sys_audio_index = deque([(0, time.time() - (samples / float(AUDIO_SR)))])
                    elif samples - last_samples >= min_growth:
                        # record the offset of the first new sample with current time
                        with self.lock:
                            self.sys_audio_index.append((last_samples, time.time()))
                            # prune old entries
                            cutoff = time.time() - (self.max_buffer_seconds + 1.0)
                            while self.sys_audio_index and self.sys_audio_index[0][1] < cutoff:
                                self.sys_audio_index.popleft()
                        last_samples = samples
            except Exception:
                # ignore transient file access errors
                pass
            time.sleep(0.25)

    def _trim_buffers_locked(self):
        # Remove frames older than max_buffer_seconds
        cutoff = time.time() - self.max_buffer_seconds
        while self.frames and self.frames[0][0] < cutoff:
            ts, path = self.frames.popleft()
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

        # For audio stored in a raw file, we don't truncate the raw file here (would be expensive).
        # We'll extract the needed portion when saving a clip and can optionally rewrite the raw file later.
    def _trim_audio_files_locked(self):
        # Trim raw audio files to keep at most max_buffer_seconds
        bytes_per_sample = CHANNELS * 2
        bytes_needed = int(self.max_buffer_seconds * AUDIO_SR * bytes_per_sample)
        # mic
        if self.audio_raw_path and os.path.exists(self.audio_raw_path):
            try:
                sz = os.path.getsize(self.audio_raw_path)
                if sz > bytes_needed:
                    # keep only last bytes_needed
                    with open(self.audio_raw_path, 'rb') as rf:
                        rf.seek(max(0, sz - bytes_needed))
                        data = rf.read()
                    with open(self.audio_raw_path, 'wb') as wf:
                        wf.write(data)
                    # adjust start ts/sample count
                    kept_samples = len(data) // bytes_per_sample
                    self.audio_total_samples = kept_samples
                    self.audio_start_ts = time.time() - (kept_samples / AUDIO_SR)
                    # Rebase audio index to a single entry aligned to new start
                    try:
                        self.audio_index = deque([(0, self.audio_start_ts)])
                    except Exception:
                        self.audio_index.clear()
            except Exception:
                pass
        # system
        if self.sys_audio_raw_path and os.path.exists(self.sys_audio_raw_path):
            try:
                sz = os.path.getsize(self.sys_audio_raw_path)
                if sz > bytes_needed:
                    with open(self.sys_audio_raw_path, 'rb') as rf:
                        rf.seek(max(0, sz - bytes_needed))
                        data = rf.read()
                    with open(self.sys_audio_raw_path, 'wb') as wf:
                        wf.write(data)
                    kept_samples = len(data) // bytes_per_sample
                    self.sys_audio_total_samples = kept_samples
                    self.sys_audio_start_ts = time.time() - (kept_samples / AUDIO_SR)
                    try:
                        self.sys_audio_index = deque([(0, self.sys_audio_start_ts)])
                    except Exception:
                        self.sys_audio_index.clear()
            except Exception:
                pass

    # ----------------------------
    # Persistent config helpers
    def _config_path(self):
        cfg_dir = os.path.join(os.path.expanduser('~'), '.config')
        try:
            os.makedirs(cfg_dir, exist_ok=True)
        except Exception:
            pass
        return os.path.join(cfg_dir, 'screen_clipper.json')

    def _resource_path(self, *parts):
        """Return a filesystem path for bundled resources.

        When running from a PyInstaller onefile/onedir bundle the
        unpacked resources are available under sys._MEIPASS. Otherwise
        fall back to the current working directory.
        """
        base = getattr(sys, '_MEIPASS', os.getcwd())
        try:
            return os.path.join(base, *parts)
        except Exception:
            # Fallback to cwd join if anything odd happens
            return os.path.join(os.getcwd(), *parts)

    def _load_config(self):
        # defaults
        self.mic_gain = 100.0
        self.sys_gain = 100.0
        self.wake_enabled = True
        self.wake_clip_on_detect = True
        self.audio_blocksize = 1024
        self._theme = 'dark'
        self._accent_name = 'Orange'
        try:
            p = self._config_path()
            if os.path.exists(p):
                with open(p, 'r') as f:
                    cfg = json.load(f)
                self.mic_gain = float(cfg.get('mic_gain', self.mic_gain))
                self.sys_gain = float(cfg.get('sys_gain', self.sys_gain))
                self.wake_enabled = bool(cfg.get('wake_enabled', self.wake_enabled))
                self.wake_clip_on_detect = bool(cfg.get('wake_clip_on_detect', self.wake_clip_on_detect))
                self.audio_blocksize = int(cfg.get('audio_blocksize', self.audio_blocksize))
                self._theme = cfg.get('theme', self._theme)
                self._accent_name = cfg.get('accent', self._accent_name)
        except Exception:
            # keep defaults on any failure
            self.mic_gain = float(getattr(self, 'mic_gain', 100.0))
            self.sys_gain = float(getattr(self, 'sys_gain', 100.0))
            self.wake_enabled = bool(getattr(self, 'wake_enabled', True))
            self.wake_clip_on_detect = bool(getattr(self, 'wake_clip_on_detect', True))
            self._theme = getattr(self, '_theme', 'dark')
            self._accent_name = getattr(self, '_accent_name', 'Orange')

    def _save_config(self):
        try:
            p = self._config_path()
            cfg = {'mic_gain': float(self.mic_gain), 'sys_gain': float(self.sys_gain), 'wake_enabled': bool(getattr(self, 'wake_enabled', True)), 'wake_clip_on_detect': bool(getattr(self, 'wake_clip_on_detect', True)), 'audio_blocksize': int(getattr(self, 'audio_blocksize', 1024)), 'theme': self._theme, 'accent': self._accent_name}
            with open(p, 'w') as f:
                json.dump(cfg, f)
        except Exception:
            pass

    def _change_mic_gain(self, delta):
        with self.lock:
            self.mic_gain = max(0.0, min(200.0, self.mic_gain + delta))
            self.mic_gain_label.config(text=f'{int(self.mic_gain)}%')
            self._save_config()
        self.update_status(f'Mic gain: {int(self.mic_gain)}%')

    def _change_sys_gain(self, delta):
        with self.lock:
            self.sys_gain = max(0.0, min(200.0, self.sys_gain + delta))
            self.sys_gain_label.config(text=f'{int(self.sys_gain)}%')
            self._save_config()
        self.update_status(f'PC gain: {int(self.sys_gain)}%')

    def _apply_accent(self, name: str):
        """Apply an accent color to primary UI elements."""
        self._accent_name = name
        # define accent colors
        if name == 'Orange':
            accent = '#ff8c42'
            active = '#ff6f17'
        elif name == 'Green':
            accent = '#43a047'
            active = '#2e7d32'
        else:
            accent = '#1e88e5'
            active = '#1669b8'

        style = ttk.Style(self)
        try:
            style.configure('Primary.TButton', background=accent, foreground='white')
            style.map('Primary.TButton', background=[('active', active)])
            style.configure('Clip.TButton', background=accent, foreground='white')
            style.map('Clip.TButton', background=[('active', active)])
        except Exception:
            # ignore styling errors on some platforms/themes
            pass

        # Ensure status text is high-contrast (white on dark)
        try:
            self.status_label.config(foreground='white')
        except Exception:
            pass

    def _toggle_theme(self):
        """Toggle between light and dark themes. Uses sv_ttk when available, otherwise adjusts key styles."""
        desired_dark = bool(self._dark_var.get())
        self._theme = 'dark' if desired_dark else 'light'
        try:
            if SV_TTK_AVAILABLE and sv_ttk:
                sv_ttk.set_theme('dark' if desired_dark else 'light')
                # sv_ttk may change colors; re-apply accent for consistent buttons
                self._apply_accent(self._accent_name)
                # ensure status label remains white in dark mode
                try:
                    self.status_label.config(foreground='white' if desired_dark else '#111')
                except Exception:
                    pass
            else:
                # Fallback: tweak style colors
                style = ttk.Style(self)
                if desired_dark:
                    bg = '#0f1115'
                    fg = '#e6eef3'
                    card = '#1b1d22'
                    status_fg = 'white'
                else:
                    bg = '#f5f7fa'
                    fg = '#111111'
                    card = '#ffffff'
                    status_fg = '#111'

                try:
                    style.configure('TFrame', background=bg)
                    style.configure('TLabel', background=bg, foreground=fg)
                    style.configure('Status.TLabel', background=bg, foreground=status_fg)
                    # refresh preview label colors
                    try:
                        self.preview_label.config(bg=card, fg=fg)
                    except Exception:
                        pass
                    try:
                        self.stats_frame.config(bg=bg)
                    except Exception:
                        pass
                except Exception:
                    pass

        except Exception:
            pass


    def _apply_gain_to_bytes(self, data_bytes, gain_percent):
        if not data_bytes:
            return data_bytes
        try:
            arr = np.frombuffer(data_bytes, dtype=np.int16).astype(np.int32)
            factor = float(gain_percent) / 100.0
            arr = np.clip((arr * factor), -32768, 32767).astype(np.int16)
            return arr.tobytes()
        except Exception:
            return data_bytes

    # ----------------------------
    # Wakeword integration
    def _init_wake_detector(self):
        # Attempt to locate model file (prefer .onnx then .tflite) under ./model
        # dynamic import to avoid requiring the module at top-level (silences Pylance missing-import)
        global ow, OPENWAKE_AVAILABLE
        if not OPENWAKE_AVAILABLE:
            try:
                ow = importlib.import_module('openwakeword')
                OPENWAKE_AVAILABLE = True
            except Exception:
                self.update_status('openwakeword not installed — wakeword disabled')
                return
        # Prefer bundled model directory when frozen by PyInstaller
        model_dir = self._resource_path('model')
        # Also accept a local ./model during development
        if not os.path.isdir(model_dir):
            alt = os.path.join(os.getcwd(), 'model')
            if os.path.isdir(alt):
                model_dir = alt
        if not os.path.isdir(model_dir):
            self.update_status('No model/ directory found for wakeword')
            return
        # Helpful debug when running from a bundled executable
        try:
            if getattr(sys, 'frozen', False):
                print(f"Running from bundled executable: using model dir {model_dir}", flush=True)
        except Exception:
            pass
        model_path = None
        for ext in ('.onnx', '.tflite'):
            for fname in os.listdir(model_dir):
                if fname.lower().endswith(ext):
                    model_path = os.path.join(model_dir, fname)
                    break
            if model_path:
                break
        if not model_path:
            self.update_status('No ONNX/TFLite wake model found in model/')
            return

        try:
            # Prefer the documented openwakeword.Model API
            det = None
            try:
                det = ow.Model(wakeword_model_paths=[model_path])
            except Exception:
                # Fallback to other historical constructor names if present
                try:
                    det = ow.Detector(model_path)
                except Exception:
                    try:
                        det = ow.OpenWakeword(model_path)
                    except Exception:
                        try:
                            det = ow.create_detector(model_path)
                        except Exception:
                            det = None

            if det is None:
                self.update_status('Failed to initialize wake detector')
                return

            # Model expects 16kHz audio; prefer explicit attribute if provided
            sr = getattr(det, 'sample_rate', None) or getattr(det, 'sr', None) or getattr(det, 'samplerate', None) or 16000
            self.wake_detector = det
            self.wake_detector_sr = int(sr)
            self.update_status(f'Wake model loaded ({os.path.basename(model_path)})')
        except Exception as e:
            # Surface the exception details to help debugging
            import traceback
            tb = traceback.format_exc()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Wake init exception:\n{tb}", flush=True)
            self.update_status(f'Wake init failed: {e}')

    def _on_wake_toggle(self):
        enabled = bool(self.wake_var.get())
        self.wake_enabled = enabled
        if enabled:
            self._init_wake_detector()
        else:
            self.wake_detector = None
            self.update_status('Wakeword disabled')
        try:
            self._save_config()
        except Exception:
            pass

    def _on_wake_clip_toggle(self):
        enabled = bool(self.wake_clip_var.get())
        self.wake_clip_on_detect = enabled
        try:
            self._save_config()
        except Exception:
            pass

    def _on_wake_test(self):
        # Run a short test capture and run the detector against it in a worker thread
        t = threading.Thread(target=self._wake_test_thread, daemon=True)
        t.start()

    def _wake_test_thread(self):
        self.update_status('Wake test: capturing...')
        # If recording, snapshot last 3s of mic audio; otherwise record directly
        duration = 3.0
        sr = getattr(self, 'wake_detector_sr', AUDIO_SR)
        samples = None
        if self.recording and self.audio_raw_path and os.path.exists(self.audio_raw_path):
            # copy snapshot
            tmp = tempfile.mktemp(prefix='wake_test_')
            try:
                shutil.copyfile(self.audio_raw_path, tmp)
                with open(tmp, 'rb') as rf:
                    # take last duration seconds
                    bytes_per = CHANNELS * 2
                    take_bytes = int(duration * AUDIO_SR * bytes_per)
                    rf.seek(max(0, os.path.getsize(tmp) - take_bytes))
                    raw = rf.read()
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
                os.remove(tmp)
            except Exception:
                samples = None
        if samples is None:
            # record directly (non-recording state)
            try:
                with sd.InputStream(samplerate=sr, channels=CHANNELS, dtype='float32') as stream:
                    frames = int(duration * sr)
                    data, _ = stream.read(frames)
                    samples = data.reshape(-1)
            except Exception as e:
                self._set_error(f'Wake test capture failed: {e}')
                return

        # Resample if detector has different sr
        det_sr = getattr(self, 'wake_detector_sr', AUDIO_SR)
        if det_sr != AUDIO_SR and det_sr != sr:
            # resample simple linear interpolation
            old_n = len(samples)
            new_n = int(round(old_n * det_sr / float(sr)))
            samples = np.interp(np.linspace(0, old_n, new_n, endpoint=False), np.arange(old_n), samples).astype(np.float32)

        # Run detector on collected samples
        res = self._run_detector_on_samples(samples, det_sr)
        self.update_status(f'Wake test finished: {res}')

    def _run_detector_on_samples(self, samples, sr):
        det = getattr(self, 'wake_detector', None)
        if det is None:
            return 'no detector'
        out_lines = []
        out_lines.append(f'det methods: {[m for m in dir(det) if not m.startswith("_")]}')

        # Normalize samples dtype: openwakeword Model expects 16-bit PCM (int16) at 16kHz
        try:
            if samples.dtype != np.int16:
                # Samples may be float32 in [-1,1] or float in another range
                if np.issubdtype(samples.dtype, np.floating):
                    samples_int = (samples * 32767.0).astype(np.int16)
                else:
                    samples_int = samples.astype(np.int16)
            else:
                samples_int = samples
        except Exception:
            samples_int = samples.astype(np.int16)

        # Try accept_waveform(sr, samples)
        try:
            if hasattr(det, 'accept_waveform'):
                try:
                    r = det.accept_waveform(sr, samples)
                    out_lines.append(f'accept_waveform(sr,samples) -> {r}')
                except Exception:
                    try:
                        r = det.accept_waveform(samples.tobytes())
                        out_lines.append(f'accept_waveform(bytes) -> {r}')
                    except Exception:
                        out_lines.append('accept_waveform -> failed')
        except Exception:
            out_lines.append('accept_waveform -> exception')

        # Try process / detect / run
        for name in ('process', 'detect', 'run'):
            if hasattr(det, name):
                try:
                    fn = getattr(det, name)
                    try:
                        r = fn(samples)
                        out_lines.append(f'{name}(samples) -> {r}')
                    except Exception:
                        try:
                            r = fn(samples.tobytes())
                            out_lines.append(f'{name}(bytes) -> {r}')
                        except Exception:
                            out_lines.append(f'{name} -> failed')
                except Exception:
                    out_lines.append(f'{name} -> exception')

        # Try Model-specific APIs
        try:
            if hasattr(det, 'predict'):
                try:
                    r = det.predict(samples_int)
                    out_lines.append(f'predict(samples_int) -> {r}')
                except Exception:
                    out_lines.append('predict -> failed')
            if hasattr(det, 'predict_clip'):
                try:
                    # predict_clip accepts a numpy array of int16 or WAV path
                    r = det.predict_clip(samples_int)
                    out_lines.append(f'predict_clip(samples_int) -> {r}')
                except Exception:
                    out_lines.append('predict_clip -> failed')
        except Exception:
            out_lines.append('Model predict/predict_clip -> exception')

        # print results to terminal
        for l in out_lines:
            try:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] WakeTest: {l}', flush=True)
            except Exception:
                pass

        # determine success heuristically from outputs
        success = any('True' in str(x) or '1' in str(x) for x in out_lines)
        return 'detected' if success else 'not-detected'

    def _on_wake_detected(self):
        # Called in main thread via after()
        self.update_status('Wakeword detected')
        try:
            self._play_beep()
        except Exception:
            pass
        # detection handler — lightweight: status + beep
        # Optionally trigger a clip when wake is detected
        try:
            if getattr(self, 'wake_clip_on_detect', True) and getattr(self, 'recording', False):
                # Indicate and perform clip
                self.update_status('Wakeword detected — saving clip')
                try:
                    self.clip_now()
                except Exception:
                    pass
        except Exception:
            pass

    def _buffer_length_seconds_locked(self):
        if not self.frames:
            return 0
        return max(0, self.frames[-1][0] - self.frames[0][0])

    def _format_seconds(self, s):
        m = int(s) // 60
        sec = int(s) % 60
        return f"{m}:{sec:02d}"

    def _preview_loop(self):
        # Update preview image from most recent frame
        last_fp = None
        pil = None
        with self.lock:
            if self.frames:
                last_fp = self.frames[-1][1]

        if last_fp and os.path.exists(last_fp):
            # Try a few recent frames in case the most recent file is being written
            pil = None
            for i in range(1, min(6, len(self.frames) + 1)):
                try_fp = self.frames[-i][1]
                if not os.path.exists(try_fp):
                    continue
                try:
                    pil = Image.open(try_fp).convert('RGB')
                    break
                except Exception:
                    pil = None
                    continue

        if pil is not None:
            # scale down to fit label size
            w, h = pil.size
            # Use the preview label size when available so the image fits the rectangular preview
            try:
                pl_w = max(200, self.preview_label.winfo_width())
                pl_h = max(150, self.preview_label.winfo_height())
            except Exception:
                pl_w = max(200, min(800, self.winfo_width() - 20))
                pl_h = max(150, min(600, self.winfo_height() - 120))
            max_w = min(820, pl_w)
            max_h = min(460, pl_h)
            scale = min(max_w / w, max_h / h, 1.0)
            if scale < 1.0:
                pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            self.preview_image = ImageTk.PhotoImage(pil)
            try:
                self.preview_label.configure(image=self.preview_image, text='')
            except Exception:
                self.preview_label.configure(image=self.preview_image)

        # schedule next preview
        self.after(int(1000 / PREVIEW_FPS), self._preview_loop)

    def clip_now(self):
        if not self.recording:
            return
        # Snapshot buffers and run saving in a background thread
        with self.lock:
            frames_copy = list(self.frames)
            duration = self.max_buffer_seconds

        t = threading.Thread(target=self._save_clip_thread, args=(frames_copy, duration), daemon=True)
        t.start()

    def _save_clip_thread(self, frames_copy, duration):
        # Indicate buffering status
        self.update_status("Buffering... preparing clip")

        fb_tmpdir = None

        if not frames_copy:
            # Try to take a synchronous snapshot as a fallback
            try:
                with mss.mss() as sct:
                    monitor = sct.monitors[1]
                    img = sct.grab(monitor)
                    arr = np.array(img)
                    if arr.shape[2] == 4:
                        frame_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
                    else:
                        frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    # write fallback frame to temp dir so downstream code can read a filename
                    tmpdir_fb = tempfile.mkdtemp(prefix='screen_clipper_fb_')
                    fb_tmpdir = tmpdir_fb
                    fname = os.path.join(tmpdir_fb, 'fb.jpg')
                    ret, buf = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    with open(fname, 'wb') as f:
                        f.write(buf.tobytes())
                    frames_copy = [(time.time(), fname)]
            except Exception:
                self._set_error("No frames available to save")
                return

        # Build temp files
        tmpdir = tempfile.mkdtemp(prefix='screen_clipper_')
        video_path = os.path.join(tmpdir, 'clip.avi')
        audio_path = os.path.join(tmpdir, 'clip.wav')
        final_name = datetime.now().strftime('clip_%Y-%m-%d_%H-%M-%S.mp4')
        final_path = os.path.join(os.getcwd(), final_name)

        try:
            # Prepare frames selection for requested duration (operate on filenames)
            end_ts = frames_copy[-1][0]
            start_ts = end_ts - duration
            sel = [(t, fp) for (t, fp) in frames_copy if t >= start_ts]
            if not sel:
                sel = [frames_copy[-1]]

            # Copy selected files into our working tmpdir to avoid races with trimming
            local_files = []
            for i, (_t, fp) in enumerate(sel):
                try:
                    dst = os.path.join(tmpdir, f'frame_{i:06d}.jpg')
                    shutil.copy(fp, dst)
                    local_files.append(dst)
                except Exception:
                    # skip missing frames
                    continue

            if not local_files:
                raise RuntimeError('No frame files could be copied for clip')

            # Determine actual video duration from number of frames we have
            actual_video_duration = max(1.0 / CAPTURE_FPS, len(local_files) / float(CAPTURE_FPS))
            # If requested duration is longer than available video, shorten to actual video duration
            target_duration = min(duration, actual_video_duration)
            # Recompute start timestamp to align audio to actual video segment
            start_ts = end_ts - target_duration

            # Determine resolution from first selected frame
            first_img = cv2.imread(local_files[0])
            if first_img is None:
                raise RuntimeError('Failed to read frame image')
            h, w = first_img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(video_path, fourcc, CAPTURE_FPS, (w, h))
            for fp in local_files:
                img = cv2.imread(fp)
                if img is None:
                    continue
                # if frame size differs, resize to expected
                if img.shape[1] != w or img.shape[0] != h:
                    img = cv2.resize(img, (w, h))
                out.write(img)
            out.release()

            # Prepare audio by extracting the relevant range from the raw audio file
            n_samples = int(AUDIO_SR * target_duration)
            audio_bytes = b''

            # Snapshot live raw audio files to avoid races with ongoing capture (copy under lock)
            audio_snapshot = None
            sys_snapshot = None
            with self.lock:
                audio_raw = self.audio_raw_path
                audio_start_ts = self.audio_start_ts
                sys_raw = self.sys_audio_raw_path
                sys_start_ts = self.sys_audio_start_ts
                audio_index_copy = list(self.audio_index)
                sys_index_copy = list(self.sys_audio_index)

                try:
                    if audio_raw and os.path.exists(audio_raw):
                        audio_snapshot = os.path.join(tmpdir, 'audio_snapshot.raw')
                        shutil.copyfile(audio_raw, audio_snapshot)
                except Exception:
                    audio_snapshot = None
                try:
                    if sys_raw and os.path.exists(sys_raw):
                        sys_snapshot = os.path.join(tmpdir, 'sys_snapshot.raw')
                        shutil.copyfile(sys_raw, sys_snapshot)
                except Exception:
                    sys_snapshot = None

            # Read from snapshot files (safer) using the timestamp snapshots we captured
            def _extract_from_index(snapshot_path, index_list, start_ts_val, n_samples_val):
                bytes_needed_local = n_samples_val * CHANNELS * 2
                if not snapshot_path or not os.path.exists(snapshot_path) or not index_list:
                    return None
                # find the latest index entry at or before start_ts_val
                chosen_off, chosen_ts = index_list[0]
                for off, ts in reversed(index_list):
                    if ts <= start_ts_val:
                        chosen_off, chosen_ts = off, ts
                        break
                desired_sample = int(chosen_off + max(0.0, start_ts_val - chosen_ts) * AUDIO_SR)
                start_byte = desired_sample * CHANNELS * 2
                try:
                    sz = os.path.getsize(snapshot_path)
                    if start_byte >= sz:
                        return b''
                    with open(snapshot_path, 'rb') as rf:
                        rf.seek(start_byte)
                        data = rf.read(bytes_needed_local)
                    if len(data) < bytes_needed_local:
                        data = (b'\x00' * (bytes_needed_local - len(data))) + data
                    # diagnostic
                    try:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AudioIndexMap: start_ts={start_ts_val:.3f} chosen_ts={chosen_ts:.3f} chosen_off={chosen_off} desired_sample={desired_sample}", flush=True)
                    except Exception:
                        pass
                    return data
                except Exception:
                    return None

            # Try to extract audio using per-block index mapping first
            if audio_snapshot:
                audio_bytes = _extract_from_index(audio_snapshot, audio_index_copy, start_ts, n_samples)
            else:
                audio_bytes = None

            # If index-based extraction failed, fall back to legacy timestamp method
            bytes_needed = n_samples * CHANNELS * 2
            if audio_bytes is None:
                if audio_snapshot and audio_start_ts is not None:
                    start_offset = max(0.0, start_ts - audio_start_ts)
                    start_frame = int(start_offset * AUDIO_SR)
                    start_byte = start_frame * CHANNELS * 2
                    try:
                        with open(audio_snapshot, 'rb') as rf:
                            rf.seek(start_byte)
                            audio_bytes = rf.read(bytes_needed)
                    except Exception:
                        audio_bytes = b''
                else:
                    audio_bytes = b''

            # Diagnostic: if we read fewer bytes than requested, log details to help debug choppy audio
            if len(audio_bytes) < bytes_needed:
                try:
                    sz = os.path.getsize(audio_snapshot) if audio_snapshot and os.path.exists(audio_snapshot) else (os.path.getsize(audio_raw) if audio_raw and os.path.exists(audio_raw) else 0)
                    proc_alive = hasattr(self, 'sys_capture_proc') and self.sys_capture_proc and (self.sys_capture_proc.poll() is None)
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AudioReadShort: need={bytes_needed} got={len(audio_bytes)} snapshot_size={sz} start_ts={start_ts:.3f} proc_alive={proc_alive}", flush=True)
                except Exception:
                    pass

            # Pad with silence if needed
            if len(audio_bytes) < bytes_needed:
                audio_bytes = (b'\x00' * (bytes_needed - len(audio_bytes))) + audio_bytes

            # Write WAV
            # apply mic gain to the extracted bytes (affects final recording only)
            try:
                audio_bytes = self._apply_gain_to_bytes(audio_bytes, getattr(self, 'mic_gain', 100.0))
            except Exception:
                pass
            with wave.open(audio_path, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(AUDIO_SR)
                wf.writeframes(audio_bytes)

            # If system audio was captured, extract corresponding segment and write WAV
            sys_audio_path = None
            # Prefer the snapshot if available (copied earlier to avoid races)
            if sys_snapshot and os.path.exists(sys_snapshot):
                try:
                    # Prefer index-based extraction when available
                    sys_bytes = None
                    if 'sys_index_copy' in locals() and sys_index_copy:
                        sys_bytes = _extract_from_index(sys_snapshot, sys_index_copy, start_ts, n_samples)
                    if sys_bytes is None:
                        # fallback to legacy timestamp method
                        if sys_start_ts is not None:
                            start_offset = max(0.0, start_ts - sys_start_ts)
                            start_frame = int(start_offset * AUDIO_SR)
                            start_byte = start_frame * CHANNELS * 2
                            bytes_needed = n_samples * CHANNELS * 2
                            with open(sys_snapshot, 'rb') as rf:
                                rf.seek(start_byte)
                                sys_bytes = rf.read(bytes_needed)
                    if sys_bytes is None:
                        sys_bytes = b''
                    if len(sys_bytes) < (n_samples * CHANNELS * 2):
                        sys_bytes = (b'\x00' * ((n_samples * CHANNELS * 2) - len(sys_bytes))) + sys_bytes
                    # apply system (PC) gain before writing final wav
                    try:
                        sys_bytes = self._apply_gain_to_bytes(sys_bytes, getattr(self, 'sys_gain', 100.0))
                    except Exception:
                        pass
                    sys_audio_path = os.path.join(tmpdir, 'sys_clip.wav')
                    with wave.open(sys_audio_path, 'wb') as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(2)
                        wf.setframerate(AUDIO_SR)
                        wf.writeframes(sys_bytes)
                except Exception:
                    sys_audio_path = None
            # Diagnostic: if system audio exists but is shorter than needed, log details
            if sys_audio_path and os.path.exists(sys_audio_path):
                try:
                    needed = n_samples * CHANNELS * 2
                    got = os.path.getsize(sys_snapshot) if 'sys_snapshot' in locals() and sys_snapshot and os.path.exists(sys_snapshot) else (os.path.getsize(sys_raw) if sys_raw and os.path.exists(sys_raw) else 0)
                    if got < needed:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SysAudioShort: need={needed} avail={got}", flush=True)
                except Exception:
                    pass
            else:
                with self.lock:
                    sys_raw = self.sys_audio_raw_path
                    sys_start_ts = self.sys_audio_start_ts
                if sys_raw and os.path.exists(sys_raw) and sys_start_ts is not None:
                    try:
                        sys_bytes = b''
                        start_offset = max(0.0, start_ts - sys_start_ts)
                        start_frame = int(start_offset * AUDIO_SR)
                        start_byte = start_frame * CHANNELS * 2
                        bytes_needed = n_samples * CHANNELS * 2
                        with open(sys_raw, 'rb') as rf:
                            rf.seek(start_byte)
                            sys_bytes = rf.read(bytes_needed)
                        if len(sys_bytes) < bytes_needed:
                            sys_bytes = (b'\x00' * (bytes_needed - len(sys_bytes))) + sys_bytes
                        sys_audio_path = os.path.join(tmpdir, 'sys_clip.wav')
                        with wave.open(sys_audio_path, 'wb') as wf:
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(2)
                            wf.setframerate(AUDIO_SR)
                            wf.writeframes(sys_bytes)
                    except Exception:
                        sys_audio_path = None

            # Use ffmpeg to mux to mp4. If we have system audio, mix mic + sys into one track.
            if sys_audio_path:
                # Optionally perform automatic sync estimation between mic and system audio
                delay_filter = None
                try:
                    if self.auto_sync_var.get():
                        # analyze last up to 10s of audio to estimate offset
                        def _read_wav_segment(path, max_secs=10.0):
                            with wave.open(path, 'rb') as wf:
                                sr = wf.getframerate()
                                nframes = wf.getnframes()
                                take = int(min(nframes, int(max_secs * sr)))
                                start = max(0, nframes - take)
                                wf.setpos(start)
                                raw = wf.readframes(take)
                            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
                            return arr, sr

                        mic_arr, sr1 = _read_wav_segment(audio_path, max_secs=min(10.0, target_duration))
                        sys_arr, sr2 = _read_wav_segment(sys_audio_path, max_secs=min(10.0, target_duration))
                        if mic_arr.size and sys_arr.size and sr1 == sr2:
                            # decimate to reduce correlation cost
                            decimate = 8
                            mic_dec = mic_arr[::decimate]
                            sys_dec = sys_arr[::decimate]
                            # limit search to +/- 5 seconds
                            max_shift_sec = 5.0
                            max_shift = int(max_shift_sec * sr1 / decimate)

                            # compute cross-correlation using FFT for performance
                            n = len(mic_dec) + len(sys_dec) - 1
                            # pad to next power of two for FFT
                            nfft = 1 << (n - 1).bit_length()
                            import numpy.fft as fft
                            f1 = fft.rfft(mic_dec, n=nfft)
                            f2 = fft.rfft(sys_dec, n=nfft)
                            corr = fft.irfft(f1 * np.conj(f2), n=nfft)
                            corr = np.roll(corr, - (len(sys_dec) - 1))[: (2 * max_shift + 1)]
                            if corr.size:
                                lag_idx = int(np.argmax(np.abs(corr))) - max_shift
                                # mapping: lag_idx * decimate samples at sr1 sampling
                                offset_secs = - (lag_idx * decimate) / float(sr1)
                                # sanity bounds
                                if abs(offset_secs) < 15.0:
                                    ms = int(round(abs(offset_secs) * 1000))
                                    if ms > 20:
                                        if offset_secs > 0:
                                            # sys leads mic -> delay sys
                                            delay_filter = f"[2:a]adelay={ms}|{ms}[sysd];[1:a]anull[a1];[sysd][a1]amix=inputs=2:normalize=1[aout]"
                                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AutoSync: sys leads by {offset_secs:.3f}s -> delaying sys by {ms}ms", flush=True)
                                        else:
                                            # mic leads sys -> delay mic
                                            delay_filter = f"[1:a]adelay={ms}|{ms}[micd];[2:a]anull[a2];[micd][a2]amix=inputs=2:normalize=1[aout]"
                                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AutoSync: sys lags by {-offset_secs:.3f}s -> delaying mic by {ms}ms", flush=True)
                except Exception:
                    delay_filter = None

                if delay_filter:
                    ffmpeg_cmd = [
                        'ffmpeg', '-y', '-i', video_path, '-i', audio_path, '-i', sys_audio_path,
                        '-filter_complex', delay_filter,
                        '-map', '0:v', '-map', '[aout]',
                        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                        '-c:a', 'aac', '-b:a', '128k', '-shortest', final_path
                    ]
                else:
                    # amix the two audio inputs
                    ffmpeg_cmd = [
                        'ffmpeg', '-y', '-i', video_path, '-i', audio_path, '-i', sys_audio_path,
                        '-filter_complex', '[1:a][2:a]amix=inputs=2:normalize=1[aout]',
                        '-map', '0:v', '-map', '[aout]',
                        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                        '-c:a', 'aac', '-b:a', '128k', '-shortest', final_path
                    ]
            else:
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-i', video_path, '-i', audio_path,
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                    '-c:a', 'aac', '-b:a', '128k', '-shortest', final_path
                ]
            res = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                stderr = res.stderr.decode(errors='ignore')
                self._set_error(f"ffmpeg failed: {stderr.splitlines()[-1][:200]}")
                return

            # Play short success beep
            try:
                self._play_beep()
            except Exception:
                pass

            self.update_status(f"Clip saved: {final_name}")

        except Exception as e:
            self._set_error(f"Clip save failed: {e}")
        finally:
            try:
                # Clean up temp dir
                for f in (video_path, audio_path):
                    if os.path.exists(f):
                        os.remove(f)
                os.rmdir(tmpdir)
                # remove fallback tmpdir if created
                if fb_tmpdir and os.path.exists(fb_tmpdir):
                    try:
                        for f in os.listdir(fb_tmpdir):
                            os.remove(os.path.join(fb_tmpdir, f))
                        os.rmdir(fb_tmpdir)
                    except Exception:
                        pass
            except Exception:
                pass

    def _play_beep(self):
        # Short 440Hz sine wave beep
        duration = 0.15
        t = np.linspace(0, duration, int(AUDIO_SR * duration), False)
        tone = 0.2 * np.sin(2 * np.pi * 440 * t)
        sd.play(tone.astype(np.float32), samplerate=AUDIO_SR)
        sd.wait()

    def update_status(self, msg):
        # Print status change to terminal (with timestamp). Also update UI in main thread.
        is_error = any(k in msg.lower() for k in ('failed', 'error', 'ffmpeg'))
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Avoid spamming identical status messages in the terminal.
        now = time.time()
        if msg == self.last_status and (now - self._last_status_time) < 1.0 and not is_error:
            # skip duplicate non-error status update
            return

        if is_error:
            # red text for errors
            RED = '\033[31m'
            RESET = '\033[0m'
            print(f"[{ts}] {RED}{msg}{RESET}", flush=True)
        else:
            print(f"[{ts}] {msg}", flush=True)

        # update last status/time
        self.last_status = msg
        self._last_status_time = now

        # Always update in main thread for GUI
        def _set():
            self.status_var.set(msg)
            if is_error:
                self.status_label.config(foreground='red')
            else:
                # Use white text in dark theme, dark text in light theme
                try:
                    fg = 'white' if getattr(self, '_theme', 'dark') == 'dark' else '#111'
                except Exception:
                    fg = 'white'
                self.status_label.config(foreground=fg)

        try:
            self.after(0, _set)
        except Exception:
            pass

    def _set_error(self, msg):
        self.update_status(msg)

    # ---------------------------------
    # Small helpers for safe cleanup
    def _safe_close(self, obj):
        try:
            if obj:
                obj.close()
        except Exception:
            pass

    def _safe_terminate(self, proc):
        try:
            if proc:
                proc.terminate()
        except Exception:
            pass

    def _safe_remove(self, path):
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    def _safe_rmdir(self, path):
        try:
            if path and os.path.isdir(path):
                os.rmdir(path)
        except Exception:
            pass

    def on_close(self):
        if self.recording:
            if not messagebox.askyesno("Quit", "Recording is active. Stop and quit?"):
                return
            self.stop_recording()
        # Ensure any external recorder or virtual sinks are cleaned up
        try:
            # stop/close external recorder and remove virtual sink and temp dir
            self._safe_terminate(getattr(self, 'sys_capture_proc', None))
            self._safe_close(getattr(self, 'sys_capture_file', None))
            self.sys_capture_proc = None
            self.sys_capture_file = None
            # No virtual sink cleanup required (platform-specific virtual sink support removed)
            if self.record_tmpdir and os.path.exists(self.record_tmpdir):
                for f in os.listdir(self.record_tmpdir):
                    self._safe_remove(os.path.join(self.record_tmpdir, f))
                self._safe_rmdir(self.record_tmpdir)
        finally:
            # persist config on exit
            try:
                self._save_config()
            except Exception:
                pass
            self.destroy()

def main():
    app = ScreenClipperApp()
    app.mainloop()

if __name__ == '__main__':
    main()
