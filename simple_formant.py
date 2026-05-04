"""
simple_formant.py
-----------------
All-pole vocal tract filter for guitar — talk box emulation.

WHY THIS SOUNDS DIFFERENT FROM THE PREVIOUS VERSION
----------------------------------------------------
Old: parallel peaking EQs at F1/F2/F3 → sounds like wah (it IS a wah).
New: cascade of all-pole resonators (F1–F5) → real vocal tract model.

A vocal tract is a tube — a physical all-pole resonant system. Cascading
resonators creates anti-formant notches BETWEEN the peaks through the
natural interaction of the sections. Those notches are exactly what makes
vowels sound like vowels rather than a wah pedal.

The source signal also must be near-square-wave rich in harmonics. A talk
box uses a power amp driven to full saturation. We approximate this with a
hard-clip saturator at high drive.

Install: pip install numpy scipy soundfile
Run:     python3 simple_formant.py
"""

import numpy as np
import soundfile as sf
from scipy.signal import sosfilt

# ─────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────

INPUT_FILE  = "ElecGtr.wav"
OUTPUT_FILE = "output.wav"

VOWEL_TARGET = "AH"

# Full 5-formant vocal tract data (F1–F5 Hz).
# From Hillenbrand et al. (1995), American English male speaker averages.
# All 5 formants needed for convincing vowel identity — F4/F5 add the
# "air" and brightness that distinguish vowels from wah-filter shapes.
VOWEL_FORMANTS = {
    #         F1    F2    F3    F4    F5
    "OO": (  300,   870, 2240, 3180, 3800),   # "boot"  — dark, hollow
    "OH": (  500,  1000, 2500, 3300, 4000),   # "go"    — rounded
    "AH": (  800,  1200, 2500, 3300, 4000),   # "father"— open, neutral
    "AE": (  600,  1900, 2600, 3300, 4200),   # "cat"   — nasal, forward
    "EE": (  300,  2300, 3000, 3600, 4300),   # "feet"  — bright, sharp
}

# Formant bandwidths (Hz) — independent of formant frequency.
# Real vocal tract bandwidths: F1≈60Hz, F2≈90Hz, F3≈150Hz, F4≈200Hz, F5≈250Hz.
# These are NOT Q-based (BW doesn't scale with frequency here).
FORMANT_BW = (60.0, 90.0, 150.0, 200.0, 250.0)

# ── Source saturation ────────────────────────────────────────────────────────
# A talk box uses a saturated power amp — output is near a square wave.
# "saturate" = hard clip, no output normalization → maximum harmonic density.
# Drive 8–12 recommended. Lower values reduce vowel audibility.
FUZZ_DRIVE = 10.0
FUZZ_MODE  = "saturate"

# ── Spectral tilt compensation ───────────────────────────────────────────────
# Guitar harmonics roll off at ~-6 dB/oct. This shelf tilts the saturated
# signal back up so F3/F4/F5 have enough energy to be shaped by the filter.
PRE_EMPHASIS_DB = 6.0

# ── Wet/dry blend ─────────────────────────────────────────────────────────────
# 1.0 = pure vocal tract (talk box). 0.8–0.9 = blended, less aggressive.
FORMANT_WET = 1.0


# ─────────────────────────────────────────────
#  DSP — ALL-POLE VOCAL TRACT FILTER
# ─────────────────────────────────────────────

def make_vocal_tract_sos(formants_hz, bandwidths_hz, fs):
    """
    Build the vocal tract filter as a cascade of 2nd-order all-pole
    resonators in SOS (second-order sections) form.

    Each formant becomes a conjugate pole pair at radius r, angle w:
        H_k(z) = b0_k / (1 + a1_k*z^-1 + a2_k*z^-2)
        r_k    = exp(-π * BW_k / fs)
        b0_k   = 1 + a1_k + a2_k  →  unity DC gain per section

    The FULL CASCADE (not parallel sum) naturally produces anti-formant
    notches between peaks through interaction of the sections — this is
    what the vocal tract actually does as a physical tube, and what
    distinguishes this from a parallel peaking EQ (= wah pedal).

    Uses SOS form for numerical stability (avoids coefficient round-off
    in a single high-order polynomial representation).

    Returns SOS array (n_formants × 6) for scipy.signal.sosfilt.
    """
    sos = []
    for fk, bwk in zip(formants_hz, bandwidths_hz):
        fk = float(np.clip(fk, 20.0, fs * 0.49))
        rk = float(np.exp(-np.pi * bwk / fs))
        wk = 2.0 * np.pi * fk / fs
        a1 = -2.0 * rk * np.cos(wk)
        a2 = rk ** 2
        b0 = 1.0 + a1 + a2   # A(1): unity DC gain
        sos.append([b0, 0.0, 0.0, 1.0, a1, a2])
    return np.array(sos, dtype=np.float64)


def make_high_shelf_sos(shelf_hz, gain_db, fs):
    """High-shelf filter for spectral tilt compensation (SOS form)."""
    A      = 10 ** (gain_db / 40.0)
    w0     = 2 * np.pi * np.clip(shelf_hz, 20, fs * 0.49) / fs
    alpha  = np.sin(w0) / 2 * np.sqrt(2)
    cw     = np.cos(w0)
    sA     = np.sqrt(A)
    b0 =      A * ((A+1) + (A-1)*cw + 2*sA*alpha)
    b1 = -2 * A * ((A-1) + (A+1)*cw)
    b2 =      A * ((A+1) + (A-1)*cw - 2*sA*alpha)
    a0 =           (A+1) - (A-1)*cw + 2*sA*alpha
    a1 =  2 *     ((A-1) - (A+1)*cw)
    a2 =           (A+1) - (A-1)*cw - 2*sA*alpha
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


# ── Fuzz / saturation models ──────────────────────────────────────────────────

def fuzz_saturate(x, drive):
    """
    Hard clip to ±1 after gain. No output normalization.
    At drive=10, any guitar signal > 0.1 amplitude clips to ±1 →
    output approaches a square wave → maximum harmonic density.
    This most closely matches a saturated power amp in a real talk box.
    """
    return np.clip(x * drive, -1.0, 1.0)


def fuzz_tanh(x, drive):
    return np.tanh(drive * x) / np.tanh(drive)


def fuzz_asymmetric(x, drive):
    gained = x * drive
    pos = np.where(gained > 0, np.tanh(gained * 1.4) * 0.7, 0.0)
    neg = np.where(gained <= 0, -np.tanh(-gained * 0.9), 0.0)
    return (pos + neg) / max(drive * 0.9, 1e-10)


def fuzz_hardclip(x, drive):
    return np.clip(x * drive, -1.0, 1.0) / max(drive, 1e-10)


def fuzz_diode(x, drive):
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
    "saturate":   fuzz_saturate,
    "tanh":       fuzz_tanh,
    "asymmetric": fuzz_asymmetric,
    "hardclip":   fuzz_hardclip,
    "diode":      fuzz_diode,
}


def apply_formant(signal, vowel, vowel_formants, formant_bw,
                  fuzz_drive, fuzz_mode, pre_emphasis_db, wet, fs):
    """
    All-pole vocal tract filter.

    Signal path:
      x_clean
        → hard-clip saturator  (near-square-wave source)
        → high-shelf emphasis  (tilt spectrum for F3–F5 energy)
        → all-pole vocal tract (cascade of F1–F5 resonators via sosfilt)
        → wet/dry blend with x_clean
        → RMS normalise to -12 dBFS
    """
    formants = vowel_formants[vowel]
    x = signal.astype(np.float64)

    # Saturate: creates a harmonic-dense source the vocal tract can shape.
    driven = FUZZ_MODELS[fuzz_mode](x, fuzz_drive)

    # Pre-emphasis: tilt spectrum so high formants have energy to shape.
    if pre_emphasis_db > 0.0:
        shelf_sos = make_high_shelf_sos(1000.0, pre_emphasis_db, fs)
        emphasized = sosfilt(shelf_sos, driven)
    else:
        emphasized = driven

    # All-pole vocal tract: cascade of 5 resonators applied in series.
    vt_sos  = make_vocal_tract_sos(formants, formant_bw, fs)
    filtered = sosfilt(vt_sos, emphasized)

    # Wet/dry: dry = pre-drive clean signal.
    out = (1.0 - wet) * x + wet * filtered

    # RMS normalise to -12 dBFS.
    rms = np.sqrt(np.mean(out ** 2))
    if rms > 1e-8:
        out = out * (10 ** (-12.0 / 20.0) / rms)

    return np.clip(out, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    assert VOWEL_TARGET in VOWEL_FORMANTS, \
        f"'{VOWEL_TARGET}' not in VOWEL_FORMANTS. Choose from: {list(VOWEL_FORMANTS)}"
    assert FUZZ_MODE in FUZZ_MODELS, \
        f"'{FUZZ_MODE}' not in FUZZ_MODELS. Choose from: {list(FUZZ_MODELS)}"

    print(f"Loading  {INPUT_FILE}")
    audio, fs = sf.read(INPUT_FILE, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    print(f"  {len(mono)/fs:.2f}s  @{fs}Hz")

    formants = VOWEL_FORMANTS[VOWEL_TARGET]
    labels   = ["F1", "F2", "F3", "F4", "F5"]
    print(f"Vowel '{VOWEL_TARGET}'")
    print(f"  " + "  ".join(f"{l}={f:.0f}Hz" for l, f in zip(labels, formants)))
    print(f"  BW: " + "  ".join(f"{b:.0f}Hz" for b in FORMANT_BW))
    print(f"Fuzz  mode={FUZZ_MODE}  drive={FUZZ_DRIVE}  emphasis={PRE_EMPHASIS_DB}dB")
    print(f"Wet={FORMANT_WET}")

    out = apply_formant(
        mono, VOWEL_TARGET, VOWEL_FORMANTS, FORMANT_BW,
        FUZZ_DRIVE, FUZZ_MODE, PRE_EMPHASIS_DB, FORMANT_WET, fs,
    )

    sf.write(OUTPUT_FILE, out, fs)
    print(f"Written  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()