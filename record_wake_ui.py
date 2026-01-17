#!/usr/bin/env python3
"""Simple Wake Recorder UI

Click "Start" to begin recording from the default microphone and "Stop" to stop and
save the recording as a 16 kHz mono 16-bit WAV into the project `model/` directory.

Usage: python record_wake_ui.py
"""
import os
import sys
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox

try:
    import sounddevice as sd
except Exception as e:
    sd = None

import wave

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPWIDTH = 2  # bytes for 16-bit
MODEL_DIR = os.path.join(os.getcwd(), 'model')

class RecorderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Wake Recorder')
        self.protocol('WM_DELETE_WINDOW', self.on_close)

        frame = ttk.Frame(self, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value='Idle')
        self.status_label = ttk.Label(frame, textvariable=self.status_var, font=('Segoe UI', 11))
        self.status_label.pack(pady=(0,8))

        self.toggle_btn = ttk.Button(frame, text='Start', command=self.toggle)
        self.toggle_btn.pack(fill=tk.X)

        self.duration_var = tk.StringVar(value='00:00.0')
        self.duration_label = ttk.Label(frame, textvariable=self.duration_var, font=('Segoe UI', 10))
        self.duration_label.pack(pady=(8,0))

        self.is_recording = False
        self._start_ts = None
        self._update_job = None
        self._stream = None
        self._buf = bytearray()
        # ensure model dir exists
        os.makedirs(MODEL_DIR, exist_ok=True)

        if sd is None:
            messagebox.showerror('Missing dependency', 'sounddevice not available; pip install sounddevice')
            self.toggle_btn.config(state='disabled')

    def toggle(self):
        if not self.is_recording:
            self.start()
        else:
            self.stop()

    def start(self):
        if sd is None:
            return
        # reset buffer
        self._buf = bytearray()
        try:
            self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', callback=self._callback)
            self._stream.start()
        except Exception as e:
            messagebox.showerror('Record failed', f'Failed to start input stream: {e}')
            return
        self.is_recording = True
        self._start_ts = time.time()
        self.toggle_btn.config(text='Stop')
        self.status_var.set('Recording...')
        self._schedule_update()

    def _callback(self, indata, frames, time_info, status):
        # indata is an ndarray of shape (frames, channels) dtype=int16
        try:
            self._buf.extend(indata.tobytes())
        except Exception:
            pass

    def stop(self):
        if not self.is_recording:
            return
        # stop stream
        try:
            if self._stream is not None:
                try:
                    self._stream.stop()
                except Exception:
                    pass
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
        except Exception:
            pass

        duration = time.time() - (self._start_ts or time.time())
        self.is_recording = False
        self.toggle_btn.config(text='Start')
        self.status_var.set('Saving...')
        if self._update_job:
            self.after_cancel(self._update_job)
            self._update_job = None

        # write to WAV file in models dir
        fname = datetime.now().strftime('wake_sample_%Y%m%d_%H%M%S.wav')
        outpath = os.path.join(MODEL_DIR, fname)
        try:
            with wave.open(outpath, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPWIDTH)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(bytes(self._buf))
            self.status_var.set(f'Saved: {os.path.basename(outpath)}')
            messagebox.showinfo('Saved', f'Saved recording to {outpath}')
        except Exception as e:
            self.status_var.set('Save failed')
            messagebox.showerror('Save failed', f'Could not write WAV file: {e}')

    def _schedule_update(self):
        # update duration label
        if not self.is_recording:
            self.duration_var.set('00:00.0')
            return
        dt = time.time() - (self._start_ts or time.time())
        m = int(dt) // 60
        s = int(dt) % 60
        frac = int((dt - int(dt)) * 10)
        self.duration_var.set(f'{m:02d}:{s:02d}.{frac}')
        self._update_job = self.after(100, self._schedule_update)

    def on_close(self):
        # ensure stream closed
        try:
            if self._stream is not None:
                try:
                    self._stream.stop()
                except Exception:
                    pass
                try:
                    self._stream.close()
                except Exception:
                    pass
        except Exception:
            pass
        self.destroy()

if __name__ == '__main__':
    app = RecorderApp()
    app.mainloop()
