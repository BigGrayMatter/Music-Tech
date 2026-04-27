"""
simple_formant.py
-----------------
Offline vowel/formant filter demo for electric guitar.
Processes a WAV file and writes an output you can listen to.

Signal path:
  clean input
    → fuzz (harmonic enrichment)
    → high-shelf pre-emphasis (tilt spectral energy toward F2/F3)
    → F1 + F2 + F3 peaking EQ filters (parallel vocal-tract resonances)
    → wet/dry blend with clean input
    → RMS normalise

Install deps:  pip install numpy scipy soundfile matplotlib
Run:           python3 simple_formant.py
"""

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

# ─────────────────────────────────────────────
#  SETTINGS — edit these
# ─────────────────────────────────────────────

INPUT_FILE  = "ElecGtr.wav"
OUTPUT_FILE = "output.wav"

# Pick one key from VOWELS below
VOWEL_TARGET = "AH"

VOWELS = {
    "OO": (300,   900),   # "boot"   — dark, hollow
    "OH": (500,  1000),   # "go"     — rounded
    "AH": (800,  1200),   # "father" — open, neutral (good test vowel)
    "AE": (700,  1800),   # "cat"    — nasal, forward
    "EE": (300,  2300),   # "feet"   — bright, sharp
}

# F3 (third formant) — adds clarity/brightness, especially for EE/AE.
# These are real acoustic phonetics values, NOT uniform-step placeholders.
VOWEL_F3 = {
    "OO": 2500,
    "OH": 2600,
    "AH": 2550,   # dips for open back vowels — does NOT climb linearly
    "AE": 2900,
    "EE": 3100,
}

# ── Filter bandwidth (Q) ──────────────────────────────────────────────────────
# Lower Q = wider peak = more natural vowel sound.
# Q=12 (old default) was far too narrow — it produced comb-filter artefacts,
# not vowel resonances. Real speech formant bandwidths:
#   F1: ~100-200 Hz → Q ≈ 3-5   at 300-800 Hz
#   F2: ~100-150 Hz → Q ≈ 10-15 at 1000-2300 Hz
#   F3: ~150-200 Hz → Q ≈ 8-12  at 2500-3100 Hz
F1_Q = 4.0
F2_Q = 10.0
F3_Q = 8.0

# ── Formant peak boost ────────────────────────────────────────────────────────
# Peaking EQ (not pure BPF): boosts at the formant frequency while preserving
# the rest of the spectrum. F2 gets 3 dB less, F3 gets 6 dB less than F1.
# 18 dB is a good balance — audible vowel colour without sounding like a synth.
FORMANT_PEAK_DB = 18.0

# ── Wet/dry ───────────────────────────────────────────────────────────────────
# 1.0 = pure vowel filter (dry signal is pre-drive clean input).
# 0.0 = clean dry pass-through (no processing).
FORMANT_WET = 0.95

# ── Fuzz ─────────────────────────────────────────────────────────────────────
# Drive MUST be significant (4-8) for formants to be audible on guitar.
# Clean guitar is harmonically sparse — the fuzz fills in the harmonic series
# that the formant filters then shape. Without it, vowels are inaudible.
FUZZ_DRIVE = 6.0
FUZZ_MODE  = "diode"   # "tanh" | "asymmetric" | "hardclip" | "foldback" | "diode"

# ── Pre-emphasis ──────────────────────────────────────────────────────────────
# High-shelf boost (dB) above 1 kHz applied after fuzz, before formant filters.
# Guitar harmonics naturally roll off; this tilt compensates so F2/F3
# have energy to shape. 8 dB is a reasonable starting point.
PRE_EMPHASIS_DB = 8.0


# ─────────────────────────────────────────────
#  DSP
# ─────────────────────────────────────────────

def make_peaking_coeffs(f0, Q, gain_db, fs):
    """
    Peaking EQ biquad (Audio EQ Cookbook).
    Boosts at f0 by gain_db dB with bandwidth controlled by Q.
    Unlike a BPF, passes the full signal and adds a resonant peak.
    Returns (b, a) for scipy.signal.lfilter.
    """
    f0 = np.clip(f0, 20.0, fs * 0.49)
    Q  = max(Q, 0.1)
    A     = 10 ** (gain_db / 40.0)
    w0    = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    b0 =  1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 =  1 - alpha * A
    a0 =  1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 =  1 - alpha / A
    return (np.array([b0, b1, b2]) / a0,
            np.array([1.0, a1 / a0, a2 / a0]))


def make_high_shelf_coeffs(shelf_hz, gain_db, fs):
    """High-shelf biquad for pre-emphasis (Audio EQ Cookbook §HS)."""
    A      = 10 ** (gain_db / 40.0)
    w0     = 2 * np.pi * np.clip(shelf_hz, 20, fs * 0.49) / fs
    alpha  = np.sin(w0) / 2 * np.sqrt(2)   # slope = 1
    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)
    b0 =      A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 =      A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 =           (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 =  2 *     ((A - 1) - (A + 1) * cos_w0)
    a2 =           (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


# ── Fuzz models ───────────────────────────────────────────────────────────────

def fuzz_tanh(x, drive):
    """Smooth symmetric soft clip. Only odd harmonics."""
    return np.tanh(drive * x) / np.tanh(drive)


def fuzz_asymmetric(x, drive):
    """Asymmetric clip — adds even harmonics, warmer/more organic."""
    gained = x * drive
    pos = np.where(gained > 0, np.tanh(gained * 1.4) * 0.7, 0.0)
    neg = np.where(gained <= 0, -np.tanh(-gained * 0.9), 0.0)
    return (pos + neg) / max(drive * 0.9, 1e-10)


def fuzz_hardclip(x, drive):
    """Brick-wall clip. Very buzzy — lots of high-order harmonics."""
    return np.clip(x * drive, -1.0, 1.0) / max(drive, 1e-10)


def fuzz_foldback(x, drive):
    """Signal folds back when it exceeds threshold. Harsh, spitting."""
    gained = x * drive
    threshold = 0.8
    s_norm = gained / threshold
    folded = (np.abs(((s_norm - 1) % 4) - 2) - 1) * threshold
    return folded / max(drive, 1e-10)


def fuzz_diode(x, drive):
    """
    Diode clipper approximation — exponential soft knee, asymmetric.
    Closest to a real fuzz/overdrive circuit. Recommended default.
    """
    gained = x * drive
    vf_pos, vf_neg = 0.4, 0.8

    def diode_clip(v, vf):
        return np.where(
            np.abs(v) < vf,
            v,
            np.sign(v) * (vf + np.log1p(np.abs(v) - vf + 1e-10) * 0.4),
        )

    out  = np.where(gained >= 0, diode_clip(gained, vf_pos), 0.0)
    out += np.where(gained <  0, diode_clip(gained, vf_neg), 0.0)
    peak = np.percentile(np.abs(out), 99) + 1e-10
    return out / peak * 0.85


FUZZ_MODELS = {
    "tanh":       fuzz_tanh,
    "asymmetric": fuzz_asymmetric,
    "hardclip":   fuzz_hardclip,
    "foldback":   fuzz_foldback,
    "diode":      fuzz_diode,
}


def apply_formant(signal, vowel, vowels, vowel_f3,
                  f1_q, f2_q, f3_q, formant_peak_db,
                  fuzz_drive, fuzz_mode, pre_emphasis_db, wet, fs):
    """
    Apply the vocal-tract formant filter to a mono signal.

    Key design choices vs. earlier versions:
      - Peaking EQ (not pure BPF): preserves the full-bandwidth signal and
        adds resonant peaks. Pure BPF at high wet removed everything between
        the bands, which sounded robotic rather than vowel-like.
      - Low F1_Q (4): natural F1 bandwidth ~75-200 Hz matches real speech.
        The old Q=12 gave ~25-67 Hz — impossibly narrow, not a vowel.
      - Weighted sum: F1 > F2 > F3, matching natural vowel spectral shape.
      - Dry blend is against the pre-drive clean signal, not post-fuzz.
    """
    f1_hz, f2_hz = vowels[vowel]
    f3_hz = vowel_f3[vowel]
    x = signal.astype(np.float64)   # clean reference for dry blend

    # Fuzz: fills the harmonic series so formant filters have energy to shape.
    driven = FUZZ_MODELS[fuzz_mode](x, fuzz_drive)

    # Pre-emphasis: boost highs so F2/F3 have comparable energy to F1.
    if pre_emphasis_db > 0.0:
        b_shelf, a_shelf = make_high_shelf_coeffs(1000.0, pre_emphasis_db, fs)
        emphasized = lfilter(b_shelf, a_shelf, driven)
    else:
        emphasized = driven

    # Three parallel peaking EQ filters (vocal tract resonances).
    # F2 gets -3 dB, F3 gets -6 dB relative to F1 — natural amplitude taper.
    b1, a1 = make_peaking_coeffs(f1_hz, f1_q, formant_peak_db,       fs)
    b2, a2 = make_peaking_coeffs(f2_hz, f2_q, formant_peak_db - 3.0, fs)
    b3, a3 = make_peaking_coeffs(f3_hz, f3_q, formant_peak_db - 6.0, fs)

    f1_out = lfilter(b1, a1, emphasized)
    f2_out = lfilter(b2, a2, emphasized)
    f3_out = lfilter(b3, a3, emphasized)

    # Weighted sum: F1 loudest, F2 slightly less, F3 subtle.
    vocal_tract = f1_out * 0.55 + f2_out * 0.35 + f3_out * 0.10

    # Blend clean (pre-drive) input with the formant-filtered signal.
    # At wet=1.0: pure vowel filter. At wet=0.0: clean dry guitar.
    out = (1.0 - wet) * x + wet * vocal_tract

    # RMS normalise to -12 dBFS (more robust than peak normalisation).
    rms = np.sqrt(np.mean(out ** 2))
    if rms > 1e-8:
        out = out * (10 ** (-12 / 20) / rms)

    return np.clip(out, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────
#  OPTIONAL: plot fuzz transfer curves
# ─────────────────────────────────────────────

def plot_transfer_curves():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    x = np.linspace(-1, 1, 1000)
    fig, axes = plt.subplots(1, len(FUZZ_MODELS), figsize=(14, 3))
    fig.suptitle(f"Fuzz transfer curves  (drive={FUZZ_DRIVE})", fontsize=11)

    for ax, (name, fn) in zip(axes, FUZZ_MODELS.items()):
        y = fn(x, FUZZ_DRIVE)
        ax.plot(x, y, linewidth=1.5)
        ax.plot(x, x, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(name, fontsize=9)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(0, color="black", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.3)
        ax.set_xlabel("input")
        if ax is axes[0]:
            ax.set_ylabel("output")
        ax.grid(True, alpha=0.2)
        asymmetry = np.mean(np.abs(y + y[::-1]))
        ax.text(0.05, 0.92, f"asymm={asymmetry:.3f}",
                transform=ax.transAxes, fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig("fuzz_curves.png", dpi=120, bbox_inches="tight")
    print("Saved fuzz_curves.png")
    plt.close()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    assert VOWEL_TARGET in VOWELS, \
        f"'{VOWEL_TARGET}' not in VOWELS. Choose from: {list(VOWELS)}"
    assert FUZZ_MODE in FUZZ_MODELS, \
        f"'{FUZZ_MODE}' not in FUZZ_MODELS. Choose from: {list(FUZZ_MODELS)}"

    plot_transfer_curves()

    print(f"Loading  {INPUT_FILE}")
    audio, fs = sf.read(INPUT_FILE, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    print(f"  {len(mono)/fs:.2f}s  @{fs}Hz")

    f1, f2 = VOWELS[VOWEL_TARGET]
    f3 = VOWEL_F3[VOWEL_TARGET]
    print(f"Vowel '{VOWEL_TARGET}'  F1={f1}Hz  F2={f2}Hz  F3={f3}Hz")
    print(f"Q     F1={F1_Q}  F2={F2_Q}  F3={F3_Q}  peak={FORMANT_PEAK_DB}dB")
    print(f"Fuzz  mode={FUZZ_MODE}  drive={FUZZ_DRIVE}  emphasis={PRE_EMPHASIS_DB}dB")
    print(f"Wet={FORMANT_WET}")

    out = apply_formant(
        mono, VOWEL_TARGET, VOWELS, VOWEL_F3,
        F1_Q, F2_Q, F3_Q, FORMANT_PEAK_DB,
        FUZZ_DRIVE, FUZZ_MODE, PRE_EMPHASIS_DB, FORMANT_WET, fs,
    )

    sf.write(OUTPUT_FILE, out, fs)
    print(f"Written  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()