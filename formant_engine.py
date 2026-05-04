"""
formant_engine.py
-----------------
Platform-independent DSP engine for the dual-axis wah+formant pedal.

All processing is done in pure numpy/scipy so this module runs identically
in the playtest GUI, offline batch tests, and (transpiled) on the Daisy Seed.

Architecture
------------
Signal path:
  dry → [pre_gain] → [pre_drive soft clip] → [wah BPF]
                                               └─ [formant F1] + [formant F2]
                                                  └─ wet/dry mixed ─┘ → output

  Envelope follower runs on the post-drive signal and modulates F2.

Why pre_drive matters
---------------------
Clean guitar is harmonically sparse — formant filters have almost nothing to
shape if their target frequencies fall between harmonics. A soft clipper (tanh)
before the filters fills in the harmonic series, giving the BPFs energy to work
with. This is how talk boxes work in practice. Even drive=1.5-2.0 makes a
dramatic difference in audibility. pre_drive=1.0 means no added distortion.

Key design choices
------------------
- All filters are direct-form II transposed biquads (numerically stable, good
  for time-varying coefficients updated per block).
- Coefficients are smoothed with a 1-pole LP on the *frequency* target, not
  on the coefficients themselves. This avoids nonlinear zipper artefacts.
- Vowel interpolation is done in F1/F2 Hz space (linear). Research suggests
  log-frequency interpolation is slightly more perceptually uniform, but linear
  is close enough for a controller and simpler to reason about.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Vowel targets (American English approximate, Hz)
# These are the *base* targets. Per-genre models will predict offsets/warps.
# ---------------------------------------------------------------------------

VOWELS: dict[str, tuple[float, float]] = {
    "OO": (300,  900),   # as in "boot"
    "OH": (500, 1000),   # as in "go"
    "AH": (800, 1200),   # as in "father"
    "AE": (700, 1800),   # as in "cat"
    "EE": (300, 2300),   # as in "feet"
}

VOWEL_F3: dict[str, float] = {
    "OO": 2500,   # "boot"   — high, rounded
    "OH": 2600,   # "go"     — slightly higher
    "AH": 2550,   # "father" — dips, does NOT climb linearly
    "AE": 2900,   # "cat"    — rises toward front vowels
    "EE": 3100,   # "feet"   — highest F3
}

# Default sweep trajectory (Y=0 → OO, Y=1 → EE)
# You can reorder this or make it genre-dependent later.
VOWEL_TRAJECTORY = ["OO", "OH", "AH", "AE", "EE"]

# Full 5-formant vocal tract data (F1–F5 Hz).
# From Hillenbrand et al. (1995), American English male speaker averages.
VOWEL_FORMANTS: dict[str, tuple] = {
    #         F1    F2    F3    F4    F5
    "OO": (  300,   870, 2240, 3180, 3800),
    "OH": (  500,  1000, 2500, 3300, 4000),
    "AH": (  800,  1200, 2500, 3300, 4000),
    "AE": (  600,  1900, 2600, 3300, 4200),
    "EE": (  300,  2300, 3000, 3600, 4300),
}

# Formant bandwidths (Hz) — real vocal tract values, not Q-derived.
FORMANT_BW: tuple = (60.0, 90.0, 150.0, 200.0, 250.0)

FUZZ_MODES = ("saturate", "tanh", "asymmetric", "hardclip", "foldback", "diode")


# ---------------------------------------------------------------------------
# Biquad filter (direct-form II transposed)
# ---------------------------------------------------------------------------

class Biquad:
    """
    Single second-order IIR section. State is maintained per-instance so
    multiple Biquad objects can run in parallel (F1 and F2 filters).

    Coefficients follow the Audio EQ Cookbook convention:
        H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
    normalised by a0, stored as [b0,b1,b2,a1,a2].
    """

    def __init__(self):
        self.b = np.array([1.0, 0.0, 0.0])  # [b0, b1, b2]
        self.a = np.array([0.0, 0.0])        # [a1, a2] (a0 normalised out)
        self.w = np.zeros(2)                 # delay-line state

    def set_bandpass(self, f0: float, Q: float, fs: float):
        """
        Constant-0dB-peak bandpass (Audio EQ Cookbook §BPF).
        Peak gain = 0 dB regardless of Q.
        """
        f0 = np.clip(f0, 20.0, fs * 0.49)
        Q  = max(Q, 0.1)
        w0    = 2 * np.pi * f0 / fs
        alpha = np.sin(w0) / (2 * Q)
        b0    =  np.sin(w0) / 2
        b1    =  0.0
        b2    = -np.sin(w0) / 2
        a0    =  1 + alpha
        a1    = -2 * np.cos(w0)
        a2    =  1 - alpha
        self.b = np.array([b0 / a0, b1 / a0, b2 / a0])
        self.a = np.array([a1 / a0, a2 / a0])

    def set_peaking(self, f0: float, Q: float, gain_db: float, fs: float):
        """
        Peaking EQ — useful for adding positive gain at F1/F2 rather than
        treating it as a pure bandpass. Optional, but handy for tuning.
        """
        f0 = np.clip(f0, 20.0, fs * 0.49)
        Q  = max(Q, 0.1)
        A     = 10 ** (gain_db / 40.0)
        w0    = 2 * np.pi * f0 / fs
        alpha = np.sin(w0) / (2 * Q)
        b0    =  1 + alpha * A
        b1    = -2 * np.cos(w0)
        b2    =  1 - alpha * A
        a0    =  1 + alpha / A
        a1    = -2 * np.cos(w0)
        a2    =  1 - alpha / A
        self.b = np.array([b0 / a0, b1 / a0, b2 / a0])
        self.a = np.array([a1 / a0, a2 / a0])

    def set_resonator(self, f0: float, bw: float, fs: float):
        """
        All-pole 2nd-order resonator for vocal tract formant modeling.
        H(z) = b0 / (1 + a1*z^-1 + a2*z^-2)
        where b0 = 1 + a1 + a2 normalises DC gain to unity.
        Formant peaks will be above 0 dB — the resonance boost is the effect.
        Applied in CASCADE (series), not parallel, to create anti-formant
        notches between peaks through section interaction.
        """
        f0 = np.clip(f0, 20.0, fs * 0.49)
        bw = max(bw, 1.0)
        r  = np.exp(-np.pi * bw / fs)
        w0 = 2.0 * np.pi * f0 / fs
        a1 = -2.0 * r * np.cos(w0)
        a2 = r ** 2
        b0 = 1.0 + a1 + a2   # = A(1), so H(1) = b0/A(1) = 1
        self.b = np.array([b0, 0.0, 0.0])
        self.a = np.array([a1, a2])

    def set_high_shelf(self, f0: float, gain_db: float, fs: float, slope: float = 1.0):
        """High-shelf EQ for feeding more upper harmonics into the formants."""
        f0 = np.clip(f0, 20.0, fs * 0.49)
        A = 10 ** (gain_db / 40.0)
        w0 = 2 * np.pi * f0 / fs
        alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / max(slope, 0.1) - 1) + 2)
        cos_w0 = np.cos(w0)
        sqrt_A = np.sqrt(A)

        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha

        self.b = np.array([b0 / a0, b1 / a0, b2 / a0])
        self.a = np.array([a1 / a0, a2 / a0])

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """
        Process a block of samples using direct-form II transposed.
        Modifies state in-place. Returns output array same shape as x.
        """
        y = np.empty_like(x)
        w0, w1 = self.w
        b0, b1, b2 = self.b
        a1, a2 = self.a
        for n in range(len(x)):
            xn  = x[n]
            yn  = b0 * xn + w0
            w0  = b1 * xn - a1 * yn + w1
            w1  = b2 * xn - a2 * yn
            y[n] = yn
        self.w = np.array([w0, w1])
        return y

    def reset(self):
        self.w[:] = 0.0


# ---------------------------------------------------------------------------
# Envelope follower
# ---------------------------------------------------------------------------

class EnvelopeFollower:
    """
    Peak envelope detector with independent attack/release time constants.
    Output is a smoothed amplitude estimate in [0, 1] range.
    """

    def __init__(self, attack_ms: float = 5.0, release_ms: float = 80.0,
                 fs: float = 48000.0):
        self.fs = fs
        self.envelope = 0.0
        self.set_times(attack_ms, release_ms)

    def set_times(self, attack_ms: float, release_ms: float):
        self.attack_coeff  = np.exp(-1.0 / (self.fs * attack_ms  / 1000.0))
        self.release_coeff = np.exp(-1.0 / (self.fs * release_ms / 1000.0))

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """Returns per-sample envelope values (same length as x)."""
        env = np.empty_like(x)
        e = self.envelope
        for n in range(len(x)):
            level = abs(x[n])
            if level > e:
                e = self.attack_coeff  * e + (1 - self.attack_coeff)  * level
            else:
                e = self.release_coeff * e + (1 - self.release_coeff) * level
            env[n] = e
        self.envelope = e
        return env

    def reset(self):
        self.envelope = 0.0


# ---------------------------------------------------------------------------
# Frequency smoother (1-pole LP on Hz values)
# ---------------------------------------------------------------------------

class FreqSmoother:
    """
    Smooths a target frequency value to avoid zipper noise when
    coefficients are updated every block.
    tau_ms sets the ~63% rise time.
    """

    def __init__(self, initial_hz: float, tau_ms: float = 30.0,
                 fs: float = 48000.0, block_size: int = 128):
        # Time constant in blocks
        blocks_per_sec = fs / block_size
        tau_blocks = (tau_ms / 1000.0) * blocks_per_sec
        self.coeff = np.exp(-1.0 / tau_blocks)
        self.value = initial_hz

    def update(self, target: float) -> float:
        self.value = self.coeff * self.value + (1 - self.coeff) * target
        return self.value


# ---------------------------------------------------------------------------
# Vowel interpolation
# ---------------------------------------------------------------------------

def interpolate_vowel(y_pos: float,
                      trajectory: list[str] = VOWEL_TRAJECTORY,
                      vowels: dict = VOWELS) -> tuple[float, float]:
    """
    Map a scalar y_pos in [0, 1] to (F1, F2) Hz by linearly interpolating
    along the vowel trajectory.

    y_pos=0 → first vowel in trajectory
    y_pos=1 → last vowel in trajectory
    """
    y_pos = np.clip(y_pos, 0.0, 1.0)
    n_segments = len(trajectory) - 1
    scaled = y_pos * n_segments
    idx    = int(scaled)
    t      = scaled - idx
    if idx >= n_segments:
        return vowels[trajectory[-1]]
    f1_a, f2_a = vowels[trajectory[idx]]
    f1_b, f2_b = vowels[trajectory[idx + 1]]
    return (f1_a + t * (f1_b - f1_a),
            f2_a + t * (f2_b - f2_a))


def interpolate_f3(y_pos: float,
                   trajectory: list[str] = VOWEL_TRAJECTORY,
                   f3_targets: dict = VOWEL_F3) -> float:
    """Map a scalar y_pos in [0, 1] to an F3 / singer-resonance target."""
    y_pos = np.clip(y_pos, 0.0, 1.0)
    n_segments = len(trajectory) - 1
    scaled = y_pos * n_segments
    idx    = int(scaled)
    t      = scaled - idx
    if idx >= n_segments:
        return f3_targets[trajectory[-1]]
    f3_a = f3_targets[trajectory[idx]]
    f3_b = f3_targets[trajectory[idx + 1]]
    return f3_a + t * (f3_b - f3_a)


def fuzz_saturate(x: np.ndarray, drive: float) -> np.ndarray:
    """
    Hard clip to ±1 after gain. No output normalization.
    Approaches a square wave at drive >= 8 — maximum harmonic density.
    This most closely replicates the saturated power amp in a real talk box.
    """
    return np.clip(x * drive, -1.0, 1.0)


def fuzz_tanh(x: np.ndarray, drive: float) -> np.ndarray:
    return np.tanh(drive * x) / np.tanh(drive)


def fuzz_asymmetric(x: np.ndarray, drive: float) -> np.ndarray:
    gained = x * drive
    pos = np.where(gained > 0, np.tanh(gained * 1.4) * 0.7, 0.0)
    neg = np.where(gained <= 0, -np.tanh(-gained * 0.9), 0.0)
    return (pos + neg) / max(drive * 0.9, 1e-10)


def fuzz_hardclip(x: np.ndarray, drive: float) -> np.ndarray:
    return np.clip(x * drive, -1.0, 1.0) / max(drive, 1e-10)


def fuzz_foldback(x: np.ndarray, drive: float) -> np.ndarray:
    gained = x * drive
    threshold = 0.8
    s_norm = gained / threshold
    folded = (np.abs(((s_norm - 1) % 4) - 2) - 1) * threshold
    return folded / max(drive, 1e-10)


def fuzz_diode(x: np.ndarray, drive: float) -> np.ndarray:
    gained = x * drive
    vf_pos = 0.4
    vf_neg = 0.8

    def diode_clip(v, vf):
        return np.where(
            np.abs(v) < vf,
            v,
            np.sign(v) * (vf + np.log1p(np.abs(v) - vf + 1e-10) * 0.4),
        )

    out = np.where(gained >= 0, diode_clip(gained, vf_pos), 0.0)
    out += np.where(gained < 0, diode_clip(gained, vf_neg), 0.0)
    peak = np.percentile(np.abs(out), 99) + 1e-10
    return out / peak * 0.85


FUZZ_MODEL_FUNCS = {
    "saturate":   fuzz_saturate,
    "tanh":       fuzz_tanh,
    "asymmetric": fuzz_asymmetric,
    "hardclip":   fuzz_hardclip,
    "foldback":   fuzz_foldback,
    "diode":      fuzz_diode,
}


def interpolate_formants(y_pos: float,
                         trajectory: list[str] = VOWEL_TRAJECTORY,
                         formants: dict = VOWEL_FORMANTS) -> tuple:
    """
    Map y_pos in [0, 1] to (F1, F2, F3, F4, F5) Hz by linearly
    interpolating along the vowel trajectory.
    """
    y_pos = np.clip(y_pos, 0.0, 1.0)
    n_segments = len(trajectory) - 1
    scaled = y_pos * n_segments
    idx    = int(scaled)
    t      = scaled - idx
    if idx >= n_segments:
        return formants[trajectory[-1]]
    fa = formants[trajectory[idx]]
    fb = formants[trajectory[idx + 1]]
    return tuple(a + t * (b - a) for a, b in zip(fa, fb))


def blend_with_model(model_f1: float, model_f2: float,
                     manual_f1: float, manual_f2: float,
                     blend: float) -> tuple[float, float]:
    """
    blend=0.0 → fully model-driven (genre mapping)
    blend=1.0 → fully manual (raw joystick Y position)
    """
    blend = np.clip(blend, 0.0, 1.0)
    f1 = (1 - blend) * model_f1 + blend * manual_f1
    f2 = (1 - blend) * model_f2 + blend * manual_f2
    return f1, f2


# ---------------------------------------------------------------------------
# Main pedal engine
# ---------------------------------------------------------------------------

@dataclass
class PedalParams:
    """All user-controllable parameters. Mutate these in the GUI."""

    # Joystick axes [0, 1]
    x_pos: float = 0.5       # Wah axis
    y_pos: float = 0.5       # Formant/vowel axis

    # Wah BPF
    wah_f_min:  float = 350.0
    wah_f_max:  float = 2500.0
    wah_Q:      float = 4.0

    # Pre-drive: fuzz applied BEFORE filters.
    # For talk box emulation, drive should be high (8–12) to approach a
    # square wave — maximum harmonic density for the formant filter to shape.
    pre_drive:  float = 10.0
    fuzz_mode:  str = "saturate"
    # High-shelf boost above 1 kHz, applied after fuzz before formant filters.
    # Compensates for spectral roll-off so F3/F4/F5 have energy to shape.
    pre_emphasis_db: float = 6.0

    # Formant bandwidths (Hz) — independent of formant frequency.
    # These match real vocal tract measurements, not Q-derived values.
    # Narrower = more resonant peak, wider = more natural/breathy.
    f1_bw: float = 60.0
    f2_bw: float = 90.0
    f3_bw: float = 150.0
    f4_bw: float = 200.0
    f5_bw: float = 250.0
    # Overall gain of the vocal tract signal before wet/dry blend.
    formant_gain: float = 1.0

    # Envelope follower → F2 modulation
    env_attack_ms:   float = 5.0
    env_release_ms:  float = 80.0
    env_f2_depth_hz: float = 150.0   # max F2 shift from pick attack

    # Mix
    wah_wet:     float = 0.8    # wah dry/wet
    formant_wet: float = 0.85   # raised from 0.6 — less dry masking the effect
    pre_gain:    float = 1.0    # input gain before all processing
    output_gain: float = 0.7    # output attenuation (drive raises level)

    # Model blend (0=genre model, 1=manual)
    model_blend: float = 1.0

    # Smoothing time constants (ms)
    wah_smooth_ms:     float = 20.0
    formant_smooth_ms: float = 25.0

    # Active genre model prediction (set by ML model, None = bypass)
    model_f1: Optional[float] = None
    model_f2: Optional[float] = None
    model_f3: Optional[float] = None


class PedalEngine:
    """
    The complete signal processor. Instantiate once; call process_block()
    on each audio block. Mutate params freely between calls.
    """

    def __init__(self, fs: float = 48000.0, block_size: int = 128):
        self.fs         = fs
        self.block_size = block_size
        self.params     = PedalParams()

        # Filters
        self.pre_emphasis = Biquad()
        self.wah_bpf      = Biquad()
        # Five all-pole resonators — applied in SERIES (cascade) in process_block
        self.f1_bpf = Biquad()
        self.f2_bpf = Biquad()
        self.f3_bpf = Biquad()
        self.f4_bpf = Biquad()
        self.f5_bpf = Biquad()

        # Envelope followers — fast tracks attacks, slow tracks sustain.
        self.env_follower = EnvelopeFollower(
            attack_ms=self.params.env_attack_ms,
            release_ms=self.params.env_release_ms,
            fs=fs,
        )
        self.env_follower_slow = EnvelopeFollower(
            attack_ms=self.params.env_attack_ms,
            release_ms=500.0,
            fs=fs,
        )

        # Frequency smoothers (one per formant + wah)
        wah_init = self._wah_target()
        self.wah_smoother = FreqSmoother(wah_init, self.params.wah_smooth_ms,
                                         fs, block_size)
        f1i, f2i, f3i, f4i, f5i = interpolate_formants(self.params.y_pos)
        sm = self.params.formant_smooth_ms
        self.f1_smoother = FreqSmoother(f1i, sm, fs, block_size)
        self.f2_smoother = FreqSmoother(f2i, sm, fs, block_size)
        self.f3_smoother = FreqSmoother(f3i, sm, fs, block_size)
        self.f4_smoother = FreqSmoother(f4i, sm, fs, block_size)
        self.f5_smoother = FreqSmoother(f5i, sm, fs, block_size)

        # Update coefficients with initial params
        self._update_coefficients(f2_env_offset=0.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wah_target(self) -> float:
        p = self.params
        return p.wah_f_min + p.x_pos * (p.wah_f_max - p.wah_f_min)

    def _formant_targets(self, f2_env_offset: float) -> tuple:
        p = self.params
        fmts = list(interpolate_formants(p.y_pos))
        if p.model_f1 is not None and p.model_f2 is not None:
            fmts[0] = (1-p.model_blend)*p.model_f1 + p.model_blend*fmts[0]
            fmts[1] = (1-p.model_blend)*p.model_f2 + p.model_blend*fmts[1]
        fmts[1] += f2_env_offset   # envelope follower modulates F2
        return tuple(fmts)

    def _update_coefficients(self, f2_env_offset: float):
        p  = self.params
        fs = self.fs

        wah_f = self.wah_smoother.update(self._wah_target())
        f1_t, f2_t, f3_t, f4_t, f5_t = self._formant_targets(f2_env_offset)
        f1_f = self.f1_smoother.update(f1_t)
        f2_f = self.f2_smoother.update(f2_t)
        f3_f = self.f3_smoother.update(f3_t)
        f4_f = self.f4_smoother.update(f4_t)
        f5_f = self.f5_smoother.update(f5_t)

        self.pre_emphasis.set_high_shelf(1000.0, p.pre_emphasis_db, fs)
        self.wah_bpf.set_bandpass(wah_f, p.wah_Q, fs)
        # All-pole resonators: applied in CASCADE in process_block.
        self.f1_bpf.set_resonator(f1_f, p.f1_bw, fs)
        self.f2_bpf.set_resonator(f2_f, p.f2_bw, fs)
        self.f3_bpf.set_resonator(f3_f, p.f3_bw, fs)
        self.f4_bpf.set_resonator(f4_f, p.f4_bw, fs)
        self.f5_bpf.set_resonator(f5_f, p.f5_bw, fs)

        self._current_wah_f = wah_f
        self._current_f1    = f1_f
        self._current_f2    = f2_f
        self._current_f3    = f3_f
        self._current_f4    = f4_f
        self._current_f5    = f5_f

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """
        Process one block of mono audio samples.
        x: float32 array, shape (block_size,), expected range [-1, 1]
        Returns: float32 array, same shape.

        Signal path:
          x_clean (pre-drive) ──────────────────────────────────────────┐
          x_clean → fuzz → pre_emphasis → wah BPF (wet/dry)             │
                                            └→ F1 peak + F2 peak + F3 peak (parallel)
                                               weighted sum = vocal_tract │
                                                                           ↓
                                    formant_wet blend ← x_clean ─────────┘
                                      → output_gain (tanh safety clip)

        Formant filters are PEAKING EQ (not pure BPF): the full-bandwidth
        signal is preserved with resonant boosts at the vowel frequencies.
        Dry blend uses x_clean (pre-drive) so wet=0 gives unprocessed guitar.
        """
        p = self.params
        x_clean = x.astype(np.float64) * p.pre_gain   # save pre-drive for dry blend

        # Pre-drive: harmonic enrichment — essential for audible formants
        if p.pre_drive > 1.001:
            fuzz_fn = FUZZ_MODEL_FUNCS.get(p.fuzz_mode, fuzz_tanh)
            driven = fuzz_fn(x_clean, p.pre_drive)
        else:
            driven = x_clean

        # Envelope follower on post-drive signal.
        # Use TRANSIENT DELTA (fast - slow) so F2 is only offset on pick
        # attacks, returning to zero during sustain. A constant signal produces
        # zero offset — the vowel position stays clean and stable.
        fast_env = self.env_follower.process_block(driven)
        slow_env = self.env_follower_slow.process_block(driven)
        transient = np.maximum(fast_env - slow_env, 0.0)
        f2_env_offset = float(np.mean(transient)) * p.env_f2_depth_hz
        x = x_clean   # alias so remaining code reads cleanly

        # Update filter coefficients (smoothed, once per block)
        self._update_coefficients(f2_env_offset)

        # Wah BPF — applied as wet/dry on the driven signal
        if abs(p.pre_emphasis_db) > 0.01:
            formant_input = self.pre_emphasis.process_block(driven)
        else:
            formant_input = driven

        wah_out = self.wah_bpf.process_block(formant_input)
        wah_mix = (1 - p.wah_wet) * formant_input + p.wah_wet * wah_out

        # All-pole vocal tract: five resonators in CASCADE (series).
        # Each resonator's output feeds the next — this is what creates
        # anti-formant notches between peaks, giving vowel identity rather
        # than wah character. Signal passes through all five sections.
        vt = self.f1_bpf.process_block(wah_mix)
        vt = self.f2_bpf.process_block(vt)
        vt = self.f3_bpf.process_block(vt)
        vt = self.f4_bpf.process_block(vt)
        vt = self.f5_bpf.process_block(vt)
        vocal_tract = vt * p.formant_gain

        # Blend: formant_wet=1.0 → full vowel filter, =0.0 → clean dry.
        # Blend against x_clean (pre-drive) so at wet=0 you hear unprocessed guitar.
        output = (1.0 - p.formant_wet) * x_clean + p.formant_wet * vocal_tract

        # Output gain + safety clip
        output = np.tanh(output * p.output_gain)
        return output.astype(np.float32)

    def get_display_state(self) -> dict:
        """
        Returns current internal state for the GUI to display.
        Safe to call between process_block() calls.
        """
        return {
            "wah_hz":   getattr(self, "_current_wah_f", 0.0),
            "f1_hz":    getattr(self, "_current_f1",    0.0),
            "f2_hz":    getattr(self, "_current_f2",    0.0),
            "f3_hz":    getattr(self, "_current_f3",    0.0),
            "f4_hz":    getattr(self, "_current_f4",    0.0),
            "f5_hz":    getattr(self, "_current_f5",    0.0),
            "x_pos":    self.params.x_pos,
            "y_pos":    self.params.y_pos,
            "envelope": self.env_follower.envelope,
            "transient": max(0.0, self.env_follower.envelope - self.env_follower_slow.envelope),
        }

    def reset(self):
        for bq in (self.pre_emphasis, self.wah_bpf,
                   self.f1_bpf, self.f2_bpf, self.f3_bpf,
                   self.f4_bpf, self.f5_bpf):
            bq.reset()
        self.env_follower.reset()
        self.env_follower_slow.reset()
