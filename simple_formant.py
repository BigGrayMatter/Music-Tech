# """
# simple_formant.py
# -----------------
# Self-contained formant pedal demo.
# Change INPUT_FILE and VOWEL_TARGET to try different sounds.

# Install deps:  pip install numpy scipy soundfile
# Run:           python3 simple_formant.py
# """

# import numpy as np
# import soundfile as sf

# # ─────────────────────────────────────────────
# #  SETTINGS — edit these
# # ─────────────────────────────────────────────

# INPUT_FILE  = "ElecGtr.wav"   # path to any mono or stereo WAV
# OUTPUT_FILE = "output.wav"

# # Pick one key from VOWELS below, or add your own (F1_hz, F2_hz)
# VOWEL_TARGET = "EE"

# VOWELS = {
#     "OO": (300,   900),   # dark, hollow  — as in "boot"
#     "OH": (500,  1000),   # rounded       — as in "go"
#     "AH": (800,  1200),   # open, neutral — as in "father"
#     "AE": (700,  1800),   # nasal         — as in "cat"
#     "EE": (300,  2300),   # bright, sharp — as in "feet"
# }

# # Filter shape — higher Q = narrower, more pronounced vowel character
# F1_Q = 12.0
# F2_Q = 14.0

# # 1.0 = clean, 2-4 = mild drive that fills in harmonics (makes vowels audible)
# PRE_DRIVE = 2.5

# # 0.0 = dry guitar, 1.0 = pure vowel filter (no dry signal)
# FORMANT_WET = 1


# # ─────────────────────────────────────────────
# #  DSP
# # ─────────────────────────────────────────────

# def make_bpf_coeffs(f0, Q, fs):
#     """
#     Audio EQ Cookbook bandpass — constant 0 dB peak gain.
#     Returns (b, a) as numpy arrays for scipy.signal.lfilter.
#     """
#     w0    = 2 * np.pi * f0 / fs
#     alpha = np.sin(w0) / (2 * Q)
#     b0    =  np.sin(w0) / 2
#     b2    = -np.sin(w0) / 2
#     a0    =  1 + alpha
#     a1    = -2 * np.cos(w0)
#     a2    =  1 - alpha
#     return (np.array([b0, 0.0, b2]) / a0,
#             np.array([1.0, a1 / a0, a2 / a0]))


# def apply_formant(signal, vowel_name, vowels, f1_q, f2_q, pre_drive, wet, fs):
#     """
#     Run the vocal-tract filter on a mono float32 signal.

#     Signal path:
#         input → soft-clip (pre_drive) → F1 BPF + F2 BPF (parallel, summed)
#                                           └── wet/dry blend ──► output
#     """
#     from scipy.signal import lfilter

#     f1_hz, f2_hz = vowels[vowel_name]
#     x = signal.astype(np.float64)

#     # Soft clip to enrich harmonics — makes the vowel filter audible
#     # on guitar (which is harmonically sparse when clean).
#     # tanh(d*x)/tanh(d) preserves unity gain at small amplitudes.
#     driven = np.tanh(pre_drive * x) / np.tanh(pre_drive)

#     # Two parallel bandpass filters — one per formant
#     b1, a1 = make_bpf_coeffs(f1_hz, f1_q, fs)
#     b2, a2 = make_bpf_coeffs(f2_hz, f2_q, fs)

#     f1_out = lfilter(b1, a1, driven)
#     f2_out = lfilter(b2, a2, driven)

#     # Sum the two formant bands — this is the "vocal tract" output
#     vocal_tract = (f1_out + f2_out) * 0.5

#     # Blend: wet=1.0 → pure vocal tract, wet=0.0 → dry pass-through
#     out = (1.0 - wet) * driven + wet * vocal_tract

#     # Normalise and soft-clip to prevent overs
#     peak = np.max(np.abs(out))
#     if peak > 0:
#         out = out / peak * 0.85
#     return out.astype(np.float32)


# # ─────────────────────────────────────────────
# #  MAIN
# # ─────────────────────────────────────────────

# def main():
#     audio, fs = sf.read(INPUT_FILE, dtype="float32", always_2d=True)
#     mono = audio.mean(axis=1)   # mix to mono if stereo

#     f1, f2 = VOWELS[VOWEL_TARGET]
#     print(f"Applying vowel '{VOWEL_TARGET}'  F1={f1}Hz  F2={f2}Hz")
#     print(f"  pre_drive={PRE_DRIVE}  Q=({F1_Q},{F2_Q})  wet={FORMANT_WET}")

#     out = apply_formant(mono, VOWEL_TARGET, VOWELS,
#                         F1_Q, F2_Q, PRE_DRIVE, FORMANT_WET, fs)

#     sf.write(OUTPUT_FILE, out, fs)
#     print(f"Written  {OUTPUT_FILE}")


# if __name__ == "__main__":
#     main()

"""
simple_formant_v3.py
--------------------
Adds a choice of fuzz/distortion character before the formant filters.
The distortion model matters a lot — tanh sounds "mathematical" at
high gain. Real fuzz circuits use asymmetric clipping which adds even
harmonics and sounds warmer and more vocal.

Install deps:  pip install numpy scipy soundfile matplotlib
Run:           python3 simple_formant_v3.py
"""

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

# ─────────────────────────────────────────────
#  SETTINGS — edit these
# ─────────────────────────────────────────────

INPUT_FILE   = "ElecGtr.wav"
OUTPUT_FILE  = "output_v3.wav"

VOWEL_TARGET = "OO"

VOWELS = {
    "OO": (300,   900),
    "OH": (500,  1000),
    "AH": (800,  1200),
    "AE": (700,  1800),
    "EE": (300,  2300),
}

F1_Q            = 12.0
F2_Q            = 14.0
FORMANT_WET     = 0.92
CASCADE         = True
FILTER_STAGES   = 2
PRE_EMPHASIS_DB = 8.0

# ── Fuzz settings ─────────────────────────────────────────────────────────

# How hard the fuzz is driven. 1.0 = very mild, 4.0 = medium, 10.0 = cranked.
FUZZ_DRIVE = 6.0

# Choose the fuzz character. Options:
#
#   "tanh"        — smooth soft clip, odd harmonics only. Sounds "mathematical"
#                   at high gain. The original approach.
#
#   "asymmetric"  — different clip threshold on positive vs negative half-cycle.
#                   Adds even harmonics (2nd, 4th). Warmer, more germanium-fuzz
#                   character. Good starting point.
#
#   "hardclip"    — hard brick-wall clip, symmetrical. Very buzzy and aggressive.
#                   Lots of high-order harmonics. Silicon fuzz / distortion pedal.
#
#   "foldback"    — when signal exceeds threshold, it folds back (reflects) rather
#                   than clipping. Produces a harsh, spitting character at high gain.
#                   Think Maestro FZ-1 territory.
#
#   "diode"       — approximates a diode clipper (exponential soft knee).
#                   Asymmetric by default (one diode per direction with different
#                   forward voltages). Closest to a real fuzz circuit.
#
FUZZ_MODE = "diode"


# ─────────────────────────────────────────────
#  FUZZ / DISTORTION MODELS
# ─────────────────────────────────────────────

def fuzz_tanh(x, drive):
    """
    Smooth symmetric soft clip.
    Only odd harmonics (3rd, 5th, 7th...).
    Normalised so unity-gain at small signals.
    """
    return np.tanh(drive * x) / np.tanh(drive)


def fuzz_asymmetric(x, drive):
    """
    Different clip thresholds on positive and negative halves.
    Positive clips at 0.7, negative clips harder at -1.0 (after gain).
    The asymmetry produces even harmonics — warmer, more organic character.
    This is the most immediately useful replacement for tanh.
    """
    gained = x * drive
    # Soft clip positive half at 0.7 threshold
    pos = np.where(gained >  0, np.tanh(gained * 1.4) * 0.7, 0.0)
    # Harder clip on negative half — different shape = asymmetry
    neg = np.where(gained <= 0, -np.tanh(-gained * 0.9),      0.0)
    out = pos + neg
    # Normalise to roughly unity gain
    return out / (drive * 0.9)


def fuzz_hardclip(x, drive):
    """
    Brick-wall clip after gain. Symmetrical.
    Very buzzy — lots of high-order odd harmonics.
    Sounds like a silicon Big Muff or ProCo Rat.
    """
    gained = x * drive
    return np.clip(gained, -1.0, 1.0) / drive


def fuzz_foldback(x, drive):
    """
    When the signal exceeds the threshold, it folds back (reflects)
    rather than clipping. Produces a harsh, spitting, almost ring-mod
    quality at high gain. Unusual harmonic content — lots of both even
    and odd harmonics. Useful for aggressive/metal fuzz tones.
    """
    gained = x * drive
    threshold = 0.8
    # Fold: reflect amplitude back down when it exceeds threshold
    def fold(s, t):
        # Map into [-t, t] by reflecting
        s_norm = s / t
        # Triangle wave folding
        s_fold = np.abs(((s_norm - 1) % 4) - 2) - 1
        return s_fold * t
    out = fold(gained, threshold)
    return out / drive


def fuzz_diode(x, drive):
    """
    Approximation of a diode clipper — the building block of most
    real fuzz and overdrive circuits. A diode's I-V curve is exponential,
    giving a soft knee that transitions smoothly from linear to clipping.

    We use two diodes (one per half-cycle) with slightly different
    forward voltages (Vf), which creates natural asymmetry.

    The characteristic: I = Is * (exp(V/Vt) - 1)
    Inverted to give output voltage from input current.
    We approximate this as a soft exponential knee.
    """
    gained = x * drive
    Vf_pos = 0.4    # forward voltage, positive half (slightly lower = clips sooner)
    Vf_neg = 0.8    # forward voltage, negative half (clips later = asymmetric)

    def diode_clip(v, vf):
        # Soft exponential knee around the forward voltage
        # Below vf: mostly linear. Above vf: exponential compression.
        knee = np.where(
            np.abs(v) < vf,
            v,
            np.sign(v) * (vf + np.log(1 + np.abs(v) - vf + 1e-10) * 0.4)
        )
        return knee

    pos_clipped = np.where(gained >= 0, diode_clip( gained,  Vf_pos),  0.0)
    neg_clipped = np.where(gained <  0, diode_clip( gained,  Vf_neg),  0.0)
    out = pos_clipped + neg_clipped

    # Normalise
    peak = np.percentile(np.abs(out), 99) + 1e-10
    return out / peak * 0.85


FUZZ_MODELS = {
    "tanh":       fuzz_tanh,
    "asymmetric": fuzz_asymmetric,
    "hardclip":   fuzz_hardclip,
    "foldback":   fuzz_foldback,
    "diode":      fuzz_diode,
}


# ─────────────────────────────────────────────
#  FILTER DSP (unchanged from v2)
# ─────────────────────────────────────────────

def make_bpf_coeffs(f0, Q, fs):
    w0    = 2 * np.pi * np.clip(f0, 20, fs * 0.49) / fs
    alpha = np.sin(w0) / (2 * max(Q, 0.1))
    b0    =  np.sin(w0) / 2
    a0    =  1 + alpha
    a1    = -2 * np.cos(w0)
    a2    =  1 - alpha
    return (np.array([b0, 0.0, -b0]) / a0,
            np.array([1.0, a1 / a0, a2 / a0]))


def make_high_shelf_coeffs(shelf_hz, gain_db, fs):
    A  = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * shelf_hz / fs
    alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * (1/1.0 - 1) + 2)
    b0 =      A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
    b2 =      A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
    a0 =           (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
    a1 =  2 *     ((A - 1) - (A + 1) * np.cos(w0))
    a2 =           (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def apply_bpf_stages(signal, f0, Q, fs, stages):
    out = signal.copy()
    b, a = make_bpf_coeffs(f0, Q, fs)
    for _ in range(stages):
        out = lfilter(b, a, out)
    return out


def apply_formant(signal, vowel_name, vowels, f1_q, f2_q,
                  fuzz_drive, fuzz_mode, wet, fs,
                  cascade, filter_stages, pre_emphasis_db):
    """
    Signal path:
        input
          → fuzz model (character + harmonic enrichment)
          → high-shelf pre-emphasis
          → formant filters (cascade or parallel, multi-stage)
          → wet/dry blend
          → normalise
    """
    f1_hz, f2_hz = vowels[vowel_name]
    x = signal.astype(np.float64)

    # Fuzz / distortion stage
    fuzz_fn = FUZZ_MODELS[fuzz_mode]
    driven  = fuzz_fn(x, fuzz_drive)

    # Pre-emphasis
    if pre_emphasis_db > 0.0:
        b_shelf, a_shelf = make_high_shelf_coeffs(1000.0, pre_emphasis_db, fs)
        emphasised = lfilter(b_shelf, a_shelf, driven)
    else:
        emphasised = driven

    # Formant filters
    if cascade:
        f1_out = apply_bpf_stages(emphasised, f1_hz, f1_q, fs, filter_stages)
        vocal_tract = apply_bpf_stages(f1_out, f2_hz, f2_q, fs, filter_stages)
    else:
        f1_out = apply_bpf_stages(emphasised, f1_hz, f1_q, fs, filter_stages)
        f2_out = apply_bpf_stages(emphasised, f2_hz, f2_q, fs, filter_stages)
        vocal_tract = (f1_out + f2_out) * 0.5

    # Wet/dry blend
    out = (1.0 - wet) * driven + wet * vocal_tract

    # Normalise
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.85
    return out.astype(np.float32)


# ─────────────────────────────────────────────
#  OPTIONAL: plot the transfer curves
# ─────────────────────────────────────────────

def plot_transfer_curves():
    """
    Show what each fuzz model's transfer function looks like.
    Helpful for understanding why they sound different.
    Run this to see a visual comparison before committing to a mode.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    x = np.linspace(-1, 1, 1000)
    drive = 6.0

    fig, axes = plt.subplots(1, len(FUZZ_MODELS), figsize=(14, 3))
    fig.suptitle(f"Fuzz transfer curves  (drive={drive})", fontsize=11)

    for ax, (name, fn) in zip(axes, FUZZ_MODELS.items()):
        y = fn(x, drive)
        ax.plot(x, y, linewidth=1.5)
        ax.plot(x, x, color="gray", linewidth=0.5, linestyle="--",
                label="linear (no clip)")
        ax.set_title(name, fontsize=9)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(0, color="black", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.3)
        ax.set_xlabel("input")
        if ax == axes[0]:
            ax.set_ylabel("output")
        ax.grid(True, alpha=0.2)

        # Symmetry indicator
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
    print(f"Vowel '{VOWEL_TARGET}'  F1={f1}Hz  F2={f2}Hz")
    print(f"Fuzz  mode={FUZZ_MODE}  drive={FUZZ_DRIVE}")
    print(f"Wet={FORMANT_WET}  cascade={CASCADE}  stages={FILTER_STAGES}"
          f"  emphasis={PRE_EMPHASIS_DB}dB")

    out = apply_formant(
        mono, VOWEL_TARGET, VOWELS,
        F1_Q, F2_Q, FUZZ_DRIVE, FUZZ_MODE, FORMANT_WET, fs,
        CASCADE, FILTER_STAGES, PRE_EMPHASIS_DB,
    )

    sf.write(OUTPUT_FILE, out, fs)
    print(f"Written  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()