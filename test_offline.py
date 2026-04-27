"""
test_offline.py
---------------
Offline playtest tool. Feed it a WAV file and it will:
  1. Process the whole file through the pedal engine at several
     (x_pos, y_pos) settings that you specify.
  2. Write output WAV files you can listen to.
  3. Plot a spectrogram comparison so you can see the filter action.

Usage
-----
  python3 test_offline.py --input guitar_riff.wav

  # Optionally specify output directory
  python3 test_offline.py --input guitar_riff.wav --outdir ./renders

  # Sweep through a vowel trajectory automatically
  python3 test_offline.py --input guitar_riff.wav --sweep

Requirements
------------
  pip install soundfile numpy scipy matplotlib
"""

import argparse
import os
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from formant_engine import PedalEngine, PedalParams, interpolate_vowel, VOWELS, VOWEL_TRAJECTORY


# ---------------------------------------------------------------------------
# Test presets — edit these to explore different sounds
# ---------------------------------------------------------------------------

PRESETS = {
    # name: (x_pos, y_pos, wah_wet, formant_wet, wah_Q, f1_Q, f2_Q, pre_drive)
    # ── Listen in this order ─────────────────────────────────────────────────
    # 1. Dry reference
    "dry":                  (0.5, 0.5, 0.0, 0.0,  4.0, 12.0, 14.0, 1.0),

    # 2. Pure vocal tract — 100% wet, driven. Should sound like vowels.
    #    OO vs EE should be OBVIOUSLY different here. If not, something is wrong.
    "vowel_OO_pure":        (0.5, 0.0, 0.0, 1.0,  4.0, 12.0, 14.0, 2.5),
    "vowel_AH_pure":        (0.5, 0.5, 0.0, 1.0,  4.0, 12.0, 14.0, 2.5),
    "vowel_EE_pure":        (0.5, 1.0, 0.0, 1.0,  4.0, 12.0, 14.0, 2.5),

    # 3. Blended — sounds more musical, less robotic
    "vowel_OO_blend":       (0.5, 0.0, 0.0, 0.7,  4.0, 12.0, 14.0, 2.5),
    "vowel_EE_blend":       (0.5, 1.0, 0.0, 0.7,  4.0, 12.0, 14.0, 2.5),

    # 4. Effect of drive — compare these
    "vowel_AH_nodrive":     (0.5, 0.5, 0.0, 1.0,  4.0, 12.0, 14.0, 1.0),
    "vowel_AH_drive2":      (0.5, 0.5, 0.0, 1.0,  4.0, 12.0, 14.0, 2.0),
    "vowel_AH_drive4":      (0.5, 0.5, 0.0, 1.0,  4.0, 12.0, 14.0, 4.0),

    # 5. Q comparison — higher Q = more nasal but more audible
    "vowel_EE_Q10":         (0.5, 1.0, 0.0, 0.8,  4.0, 10.0, 10.0, 2.5),
    "vowel_EE_Q18":         (0.5, 1.0, 0.0, 0.8,  4.0, 18.0, 18.0, 2.5),

    # 6. Full combined effect
    "combined_OO":          (0.2, 0.0, 0.7, 0.7,  5.0, 12.0, 14.0, 2.5),
    "combined_EE":          (0.8, 1.0, 0.7, 0.7,  5.0, 12.0, 14.0, 2.5),
    "combined_funky":       (0.4, 0.35, 0.7, 0.7, 5.5, 14.0, 16.0, 3.0),
}


def load_audio(path: str, target_fs: float = 48000.0):
    """Load audio file, convert to mono float64, resample if needed."""
    data, fs = sf.read(path, dtype="float32", always_2d=True)
    # Mix to mono
    mono = data.mean(axis=1).astype(np.float64)
    # Simple resample if sample rate differs
    if fs != target_fs:
        from scipy.signal import resample_poly
        from math import gcd
        ratio_n = int(target_fs)
        ratio_d = int(fs)
        g = gcd(ratio_n, ratio_d)
        mono = resample_poly(mono, ratio_n // g, ratio_d // g)
        print(f"  Resampled {fs}Hz → {target_fs}Hz")
    return mono.astype(np.float32), int(target_fs)


def process_preset(audio: np.ndarray, fs: int,
                   x_pos, y_pos, wah_wet, formant_wet,
                   wah_Q, f1_Q, f2_Q, pre_drive=1.0,
                   block_size: int = 128) -> np.ndarray:
    """Run the full engine over an audio array with given params."""
    engine = PedalEngine(fs=fs, block_size=block_size)
    engine.params.x_pos       = x_pos
    engine.params.y_pos       = y_pos
    engine.params.wah_wet     = wah_wet
    engine.params.formant_wet = formant_wet
    engine.params.wah_Q       = wah_Q
    engine.params.f1_Q        = f1_Q
    engine.params.f2_Q        = f2_Q
    engine.params.pre_drive   = pre_drive

    n_blocks = len(audio) // block_size
    out = np.zeros(n_blocks * block_size, dtype=np.float32)
    for i in range(n_blocks):
        block = audio[i * block_size: (i + 1) * block_size]
        out[i * block_size: (i + 1) * block_size] = engine.process_block(block)
    return out


def process_sweep(audio: np.ndarray, fs: int,
                  n_steps: int = 10, block_size: int = 128) -> np.ndarray:
    """
    Sweep Y position (vowel axis) over the full file duration.
    X stays at 0.5 (wah mid). Useful for hearing the vowel transition.
    """
    engine = PedalEngine(fs=fs, block_size=block_size)
    engine.params.wah_wet     = 0.0
    engine.params.formant_wet = 0.8
    engine.params.x_pos       = 0.5

    n_blocks  = len(audio) // block_size
    out = np.zeros(n_blocks * block_size, dtype=np.float32)
    for i in range(n_blocks):
        engine.params.y_pos = i / max(n_blocks - 1, 1)   # 0 → 1
        block = audio[i * block_size: (i + 1) * block_size]
        out[i * block_size: (i + 1) * block_size] = engine.process_block(block)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_spectrogram(ax, audio, fs, title, fmax=4000):
    from scipy.signal import spectrogram as sg
    f, t, Sxx = sg(audio.astype(np.float64), fs=fs,
                   nperseg=1024, noverlap=768, scaling="spectrum")
    mask = f <= fmax
    Sxx_db = 10 * np.log10(np.maximum(Sxx[mask], 1e-12))
    ax.pcolormesh(t, f[mask], Sxx_db, shading="gouraud",
                  cmap="magma", vmin=-80, vmax=0)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Hz")
    ax.set_xlabel("s")
    # Mark vowel formant bands
    for v, (f1, f2) in VOWELS.items():
        ax.axhline(f1, color="cyan", lw=0.5, alpha=0.4)
        ax.axhline(f2, color="lime", lw=0.5, alpha=0.4)


def plot_vowel_space(ax, highlights: dict[str, tuple[float, float]]):
    """Plot F1/F2 vowel space with standard vowel positions."""
    ax.set_title("F1/F2 vowel space", fontsize=9)
    ax.set_xlabel("F1 (Hz)")
    ax.set_ylabel("F2 (Hz)")
    ax.invert_xaxis()   # Phonetic convention: low F1 on right
    ax.invert_yaxis()   # Low F2 at top (close vowels)

    # Plot standard vowels
    for v, (f1, f2) in VOWELS.items():
        ax.scatter(f1, f2, s=80, zorder=3, color="steelblue")
        ax.annotate(v, (f1, f2), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color="steelblue")

    # Draw trajectory
    traj_pts = [VOWELS[v] for v in VOWEL_TRAJECTORY]
    xs = [p[0] for p in traj_pts]
    ys = [p[1] for p in traj_pts]
    ax.plot(xs, ys, "steelblue", lw=1, alpha=0.4, linestyle="--")

    # Highlight active positions for this render
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(highlights)))
    for (label, (f1, f2)), c in zip(highlights.items(), colors):
        ax.scatter(f1, f2, s=120, zorder=4, color=c, marker="D",
                   edgecolors="white", linewidth=0.5)
        ax.annotate(label, (f1, f2), textcoords="offset points",
                    xytext=(5, -10), fontsize=7, color=c)

    ax.set_xlim(900, 200)
    ax.set_ylim(2600, 700)
    ax.grid(True, alpha=0.2)


def make_report(input_path, outdir, results: dict, fs: int, sweep_out=None):
    """
    Generate a multi-panel PDF/PNG report showing spectrograms for each
    preset and the vowel space positions.
    """
    n_presets = len(results)
    fig = plt.figure(figsize=(16, 3 * (n_presets // 2 + 1) + 3))
    gs  = gridspec.GridSpec(n_presets // 2 + 2, 2,
                            hspace=0.55, wspace=0.35)

    # Spectrograms
    for i, (name, audio) in enumerate(results.items()):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        plot_spectrogram(ax, audio, fs, name)

    # Vowel space (bottom left)
    ax_v = fig.add_subplot(gs[-1, 0])
    highlights = {}
    for name, preset in PRESETS.items():
        if preset[3] > 0:   # formant_wet > 0
            f1, f2 = interpolate_vowel(preset[1])
            highlights[name] = (f1, f2)
    plot_vowel_space(ax_v, highlights)

    # Sweep spectrogram (bottom right) if available
    if sweep_out is not None:
        ax_s = fig.add_subplot(gs[-1, 1])
        plot_spectrogram(ax_s, sweep_out, fs, "vowel sweep (Y: 0→1)")

    fig.suptitle(f"Formant Pedal Playtest — {os.path.basename(input_path)}",
                 fontsize=11, y=1.01)
    out_path = os.path.join(outdir, "report.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Report saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline formant pedal tester")
    parser.add_argument("--input",  required=True, help="Input WAV file")
    parser.add_argument("--outdir", default="./renders",
                        help="Output directory (default: ./renders)")
    parser.add_argument("--sweep",  action="store_true",
                        help="Also render a Y-axis vowel sweep")
    parser.add_argument("--fs",     type=float, default=48000.0,
                        help="Processing sample rate (default: 48000)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Loading {args.input}...")
    audio, fs = load_audio(args.input, args.fs)
    print(f"  {len(audio)/fs:.2f}s  @{fs}Hz  ({len(audio)} samples)")

    results = {}
    for name, preset in PRESETS.items():
        x_pos, y_pos, wah_wet, formant_wet, wah_Q, f1_Q, f2_Q, pre_drive = preset
        print(f"  Rendering preset: {name}")
        out = process_preset(audio, fs, x_pos, y_pos,
                             wah_wet, formant_wet, wah_Q, f1_Q, f2_Q, pre_drive)
        wav_path = os.path.join(args.outdir, f"{name}.wav")
        sf.write(wav_path, out, fs)
        results[name] = out

    sweep_out = None
    if args.sweep:
        print("  Rendering vowel sweep...")
        sweep_out = process_sweep(audio, fs)
        sf.write(os.path.join(args.outdir, "vowel_sweep.wav"), sweep_out, fs)

    print("Generating report...")
    make_report(args.input, args.outdir, results, fs, sweep_out)
    print("Done.")


if __name__ == "__main__":
    main()