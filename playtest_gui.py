"""
playtest_gui.py
---------------
Real-time GUI playtest environment for the dual-axis formant pedal.

This runs on your local machine (not in the sandbox) where you have:
  - PortAudio / sounddevice  → pip install sounddevice
  - tkinter                  → usually bundled with Python; on Ubuntu:
                                sudo apt-get install python3-tk
  - soundfile                → pip install soundfile
  - numpy, scipy, matplotlib → pip install numpy scipy matplotlib

Run it:
  python3 playtest_gui.py                   # live mic/guitar input
  python3 playtest_gui.py --file riff.wav   # play from WAV file in loop
  python3 playtest_gui.py --list-devices    # show audio device list

Controls
--------
  X slider      Wah position (heel → toe)
  Y slider      Vowel / formant position (OO → EE)
  Wah wet       Dry/wet for wah BPF
  Formant wet   Dry/wet for formant pair
  Wah Q         Q of wah bandpass
  F1 Q / F2 Q   Q of formant peaks
  Env depth     How much pick attack modulates F2
  Model blend   0=genre model prediction, 1=manual joystick

The bottom-left panel shows a live F1/F2 vowel space plot with the
current filter position animated in real time.
"""

import argparse
import queue
import threading
import time
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyArrowPatch
import matplotlib.animation as animation

from formant_engine import (
    PedalEngine, PedalParams, interpolate_vowel,
    VOWELS, VOWEL_TRAJECTORY
)

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("WARNING: sounddevice/soundfile not available. "
          "GUI will launch in 'no-audio' mode.")


# ---------------------------------------------------------------------------
# Audio I/O thread
# ---------------------------------------------------------------------------

class AudioThread(threading.Thread):
    """
    Runs the sounddevice stream in a dedicated thread so the GUI stays
    responsive. The PedalEngine lives here and is called per-block.
    """

    BLOCK_SIZE = 128
    FS         = 48000

    def __init__(self, engine: PedalEngine, input_device=None,
                 output_device=None, file_path=None):
        super().__init__(daemon=True)
        self.engine    = engine
        self.in_dev    = input_device
        self.out_dev   = output_device
        self.file_path = file_path
        self._stop_evt = threading.Event()

        # For waveform display (ring buffer of recent samples)
        self.display_buf = np.zeros(self.FS // 4, dtype=np.float32)
        self._buf_lock   = threading.Lock()
        self._write_pos  = 0

        # File playback state
        self._file_data = None
        self._file_pos  = 0
        if file_path:
            self._load_file(file_path)

    def _load_file(self, path):
        data, fs = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if fs != self.FS:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(self.FS), int(fs))
            mono = resample_poly(mono, self.FS // g, fs // g)
        self._file_data = mono.astype(np.float32)
        self._file_pos  = 0

    def _get_file_block(self, n):
        if self._file_data is None:
            return np.zeros(n, dtype=np.float32)
        d   = self._file_data
        pos = self._file_pos
        if pos + n > len(d):
            # Wrap around (loop)
            part1 = d[pos:]
            part2 = d[:n - len(part1)]
            block = np.concatenate([part1, part2])
            self._file_pos = n - len(part1)
        else:
            block = d[pos: pos + n].copy()
            self._file_pos = pos + n
        return block

    def _push_display(self, samples: np.ndarray):
        with self._buf_lock:
            n = len(samples)
            end = self._write_pos + n
            if end <= len(self.display_buf):
                self.display_buf[self._write_pos:end] = samples
            else:
                split = len(self.display_buf) - self._write_pos
                self.display_buf[self._write_pos:] = samples[:split]
                self.display_buf[:n - split] = samples[split:]
            self._write_pos = end % len(self.display_buf)

    def get_display_buf(self) -> np.ndarray:
        with self._buf_lock:
            return self.display_buf.copy()

    def run(self):
        if not AUDIO_AVAILABLE:
            return

        def callback(indata, outdata, frames, time_info, status):
            if self._file_data is not None:
                x = self._get_file_block(frames)
            else:
                x = indata[:, 0].copy()
            y = self.engine.process_block(x)
            outdata[:, 0] = y
            if outdata.shape[1] > 1:
                outdata[:, 1] = y
            self._push_display(y)

        with sd.Stream(
            samplerate=self.FS,
            blocksize=self.BLOCK_SIZE,
            dtype="float32",
            channels=(1, 2),
            device=(self.in_dev, self.out_dev),
            callback=callback,
        ):
            while not self._stop_evt.is_set():
                time.sleep(0.05)

    def stop(self):
        self._stop_evt.set()


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class PlaytestGUI:

    def __init__(self, engine: PedalEngine, audio_thread: AudioThread = None):
        self.engine       = engine
        self.audio_thread = audio_thread

        self.root = tk.Tk()
        self.root.title("Formant Pedal Playtest")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(True, True)

        self._build_layout()
        self._build_vowel_plot()
        self._build_waveform_plot()

        # Periodic GUI update
        self.root.after(50, self._update_display)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TScale",   background="#1e1e2e", troughcolor="#313244",
                        sliderthickness=16)
        style.configure("TLabel",   background="#1e1e2e", foreground="#cdd6f4",
                        font=("Helvetica", 10))
        style.configure("TFrame",   background="#1e1e2e")
        style.configure("Header.TLabel", background="#1e1e2e",
                        foreground="#89b4fa", font=("Helvetica", 11, "bold"))

        # ── Main columns ─────────────────────────────────────────────
        left  = ttk.Frame(self.root)
        right = ttk.Frame(self.root)
        left.grid( row=0, column=0, sticky="nsew", padx=8, pady=8)
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        # ── Left: sliders ────────────────────────────────────────────
        ttk.Label(left, text="Joystick axes", style="Header.TLabel"
                  ).grid(row=0, column=0, columnspan=2, pady=(0, 4))

        self._sliders = {}
        slider_defs = [
            # (label, attr, min, max, default, row)
            ("X — wah",         "x_pos",         0.0,   1.0,   0.5,  1),
            ("Y — formant",     "y_pos",          0.0,   1.0,   0.5,  2),
            ("",                None,             0,     1,     0,    3),
            ("Pre drive",       "pre_drive",      1.0,   8.0,   2.5,  4),
            ("Pre gain",        "pre_gain",       0.1,   4.0,   1.0,  5),
            ("Output gain",     "output_gain",    0.1,   2.0,   0.7,  6),
            ("",                None,             0,     1,     0,    7),
            ("Wah wet",         "wah_wet",        0.0,   1.0,   0.8,  8),
            ("Formant wet",     "formant_wet",    0.0,   1.0,   0.85, 9),
            ("Formant gain",    "formant_gain",   0.1,   4.0,   1.5, 10),
            ("",                None,             0,     1,     0,   11),
            ("Wah Q",           "wah_Q",          0.5,  20.0,   4.0, 12),
            ("F1 Q",            "f1_Q",           1.0,  25.0,  12.0, 13),
            ("F2 Q",            "f2_Q",           1.0,  25.0,  14.0, 14),
            ("",                None,             0,     1,     0,   15),
            ("Env depth (Hz)",  "env_f2_depth_hz", 0,  500,  150.0, 16),
            ("Env attack (ms)", "env_attack_ms",  1.0, 100.0,   5.0, 17),
            ("Env release (ms)","env_release_ms", 5.0, 500.0,  80.0, 18),
            ("",                None,             0,     1,     0,   19),
            ("Model blend",     "model_blend",    0.0,   1.0,   1.0, 20),
        ]

        for label, attr, lo, hi, default, row in slider_defs:
            if attr is None:
                ttk.Label(left, text="").grid(row=row, column=0)
                continue

            ttk.Label(left, text=label, width=18, anchor="e"
                      ).grid(row=row, column=0, sticky="e", padx=4, pady=1)

            val_var = tk.DoubleVar(value=default)
            val_lbl = ttk.Label(left, text=f"{default:.2f}", width=6)
            val_lbl.grid(row=row, column=2, sticky="w", padx=4)

            def _make_cb(a=attr, v=val_var, lbl=val_lbl):
                def cb(val):
                    f = float(val)
                    lbl.config(text=f"{f:.2f}")
                    setattr(self.engine.params, a, f)
                    # If attack/release change, update env follower
                    if a in ("env_attack_ms", "env_release_ms"):
                        self.engine.env_follower.set_times(
                            self.engine.params.env_attack_ms,
                            self.engine.params.env_release_ms,
                        )
                return cb

            s = ttk.Scale(left, from_=lo, to=hi, orient="horizontal",
                          variable=val_var, length=200, command=_make_cb())
            s.grid(row=row, column=1, sticky="ew", padx=4)
            self._sliders[attr] = (s, val_var, val_lbl)

        # ── Right top: waveform ───────────────────────────────────────
        self.wave_frame = ttk.Frame(right)
        self.wave_frame.grid(row=0, column=0, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=2)
        right.columnconfigure(0, weight=1)

        # ── Right bottom: vowel space ─────────────────────────────────
        self.vowel_frame = ttk.Frame(right)
        self.vowel_frame.grid(row=1, column=0, sticky="nsew")

        # ── Status bar ────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief="sunken", anchor="w")
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _build_vowel_plot(self):
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#1e1e2e")
        ax.set_facecolor("#181825")

        ax.set_title("F1 / F2 vowel space", color="#cdd6f4", fontsize=10)
        ax.set_xlabel("F1 (Hz)", color="#a6adc8", fontsize=9)
        ax.set_ylabel("F2 (Hz)", color="#a6adc8", fontsize=9)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.tick_params(colors="#a6adc8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")

        # Static vowel positions
        for v, (f1, f2) in VOWELS.items():
            ax.scatter(f1, f2, s=60, color="#89b4fa", zorder=3)
            ax.annotate(v, (f1, f2), xytext=(5, 5),
                        textcoords="offset points", fontsize=9,
                        color="#89b4fa")

        # Trajectory
        traj_pts = [VOWELS[v] for v in VOWEL_TRAJECTORY]
        ax.plot([p[0] for p in traj_pts], [p[1] for p in traj_pts],
                color="#313244", lw=1.5, zorder=1)

        # Live cursor (will be updated)
        self._vowel_cursor, = ax.plot([], [], "o", color="#f38ba8",
                                      ms=10, zorder=5)
        # Wah marker (horizontal line showing wah freq position)
        self._wah_line = ax.axhline(0, color="#a6e3a1", lw=1,
                                    alpha=0.5, linestyle="--")

        ax.set_xlim(900, 200)
        ax.set_ylim(2800, 600)
        ax.grid(True, alpha=0.1, color="#45475a")
        fig.tight_layout()

        self._vowel_fig = fig
        self._vowel_ax  = ax
        canvas = FigureCanvasTkAgg(fig, master=self.vowel_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._vowel_canvas = canvas

    def _build_waveform_plot(self):
        fig, ax = plt.subplots(figsize=(5, 1.8))
        fig.patch.set_facecolor("#1e1e2e")
        ax.set_facecolor("#181825")
        ax.set_title("Output waveform", color="#cdd6f4", fontsize=9)
        ax.tick_params(colors="#a6adc8", labelsize=7)
        ax.set_ylim(-1.1, 1.1)
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")
        ax.axhline(0, color="#45475a", lw=0.5)

        n_display = 1024
        self._wave_x = np.arange(n_display)
        self._wave_line, = ax.plot(self._wave_x,
                                   np.zeros(n_display),
                                   color="#a6e3a1", lw=0.8)
        ax.set_xlim(0, n_display)
        fig.tight_layout()

        self._wave_fig    = fig
        self._wave_canvas = FigureCanvasTkAgg(fig, master=self.wave_frame)
        self._wave_canvas.draw()
        self._wave_canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Periodic update
    # ------------------------------------------------------------------

    def _update_display(self):
        state = self.engine.get_display_state()

        # Update vowel cursor
        f1, f2 = state["f1_hz"], state["f2_hz"]
        self._vowel_cursor.set_data([f1], [f2])

        # Update wah line (show wah frequency on Y axis as horizontal ref)
        # We repurpose it as a vertical dashed line at wah_hz on F1 axis
        wah_hz = state["wah_hz"]
        self._wah_line.set_ydata([wah_hz, wah_hz])
        self._wah_line.set_xdata([0, 1])    # ignored for axhline but set anyway

        self._vowel_canvas.draw_idle()

        # Update waveform
        if self.audio_thread is not None:
            buf = self.audio_thread.get_display_buf()
            n = min(1024, len(buf))
            self._wave_line.set_ydata(buf[-n:])
            self._wave_canvas.draw_idle()

        # Status bar
        self.status_var.set(
            f"Wah: {wah_hz:.0f} Hz  |  F1: {f1:.0f} Hz  |  "
            f"F2: {f2:.0f} Hz  |  Env: {state['envelope']:.3f}"
        )

        self.root.after(50, self._update_display)   # ~20 Hz refresh

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real-time formant pedal playtest GUI")
    parser.add_argument("--file",         help="WAV file to loop instead of mic")
    parser.add_argument("--input-device",  type=int, default=None,
                        help="Input audio device index (see --list-devices)")
    parser.add_argument("--output-device", type=int, default=None,
                        help="Output audio device index")
    parser.add_argument("--list-devices", action="store_true",
                        help="List audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        if AUDIO_AVAILABLE:
            print(sd.query_devices())
        else:
            print("sounddevice not available")
        return

    engine = PedalEngine(fs=48000, block_size=128)

    audio_thread = None
    if AUDIO_AVAILABLE:
        audio_thread = AudioThread(
            engine,
            input_device=args.input_device,
            output_device=args.output_device,
            file_path=args.file,
        )
        audio_thread.start()
    else:
        print("No audio I/O. Launching GUI in display-only mode.")

    gui = PlaytestGUI(engine, audio_thread)
    gui.run()

    if audio_thread:
        audio_thread.stop()


if __name__ == "__main__":
    main()