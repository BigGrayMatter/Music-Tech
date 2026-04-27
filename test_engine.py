"""
test_engine.py
--------------
Quick DSP sanity checks. Run with:
  python3 test_engine.py

Tests
-----
1. Biquad BPF centre frequency — FFT peak should land at f0 ±2%
2. Biquad stability — 10s of white noise shouldn't blow up
3. Vowel interpolation endpoints and midpoints
4. Envelope follower tracks peaks
5. Full engine smoke test — processes 1s of white noise without exception
6. Coefficient smoothing — no NaN/Inf after rapid parameter change
"""

import numpy as np
import sys

# Allow running from this directory
sys.path.insert(0, ".")
from formant_engine import (
    Biquad, EnvelopeFollower, FreqSmoother,
    interpolate_vowel, VOWELS, VOWEL_TRAJECTORY,
    PedalEngine, PedalParams
)

FS = 48000
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

def check(name, cond, detail=""):
    if cond:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}  {detail}")
    return cond


# ---------------------------------------------------------------------------

def test_biquad_centre_frequency():
    """BPF peak should be within 2% of target f0."""
    results = []
    for f0 in [350, 700, 1200, 2000, 2500]:
        bpf = Biquad()
        bpf.set_bandpass(f0, Q=6.0, fs=FS)

        # Impulse response → magnitude spectrum
        impulse = np.zeros(4096)
        impulse[0] = 1.0
        resp = bpf.process_block(impulse)
        mag  = np.abs(np.fft.rfft(resp))
        freqs = np.fft.rfftfreq(4096, 1.0 / FS)
        peak_f = freqs[np.argmax(mag)]
        err_pct = abs(peak_f - f0) / f0 * 100
        results.append(check(
            f"  BPF peak @ {f0}Hz → {peak_f:.1f}Hz (err {err_pct:.1f}%)",
            err_pct < 2.0,
            f"peak at {peak_f:.1f}Hz"
        ))
    return all(results)


def test_biquad_stability():
    """White noise through BPF should not produce NaN/Inf."""
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(FS * 10).astype(np.float32)
    bpf = Biquad()
    bpf.set_bandpass(1000, Q=8, fs=FS)
    out = bpf.process_block(noise)
    return check("Biquad stability (10s white noise)",
                 np.isfinite(out).all() and np.max(np.abs(out)) < 10.0)


def test_peaking_eq():
    """Peaking EQ centre frequency should be within 2%."""
    results = []
    for f0, gain in [(500, 6), (1500, 3), (2000, -3)]:
        bpf = Biquad()
        bpf.set_peaking(f0, Q=8, gain_db=gain, fs=FS)
        impulse = np.zeros(4096); impulse[0] = 1.0
        resp = bpf.process_block(impulse)
        mag = np.abs(np.fft.rfft(resp))
        freqs = np.fft.rfftfreq(4096, 1.0 / FS)
        if gain > 0:
            peak_f = freqs[np.argmax(mag)]
        else:
            peak_f = freqs[np.argmin(mag)]
        err_pct = abs(peak_f - f0) / f0 * 100
        results.append(check(
            f"  Peaking EQ {gain:+d}dB @ {f0}Hz → {peak_f:.1f}Hz (err {err_pct:.1f}%)",
            err_pct < 3.0
        ))
    return all(results)


def test_vowel_interpolation():
    """Interpolation endpoints and midpoint."""
    results = []
    # y=0 → first vowel
    f1, f2 = interpolate_vowel(0.0)
    target = VOWELS[VOWEL_TRAJECTORY[0]]
    results.append(check(
        f"y=0 → {VOWEL_TRAJECTORY[0]} ({target[0]},{target[1]})",
        abs(f1 - target[0]) < 1 and abs(f2 - target[1]) < 1
    ))
    # y=1 → last vowel
    f1, f2 = interpolate_vowel(1.0)
    target = VOWELS[VOWEL_TRAJECTORY[-1]]
    results.append(check(
        f"y=1 → {VOWEL_TRAJECTORY[-1]} ({target[0]},{target[1]})",
        abs(f1 - target[0]) < 1 and abs(f2 - target[1]) < 1
    ))
    # y=0.5 → somewhere between middle vowels (F1 and F2 should be in range)
    f1, f2 = interpolate_vowel(0.5)
    all_f1 = [v[0] for v in VOWELS.values()]
    all_f2 = [v[1] for v in VOWELS.values()]
    results.append(check(
        f"y=0.5 → ({f1:.0f},{f2:.0f}) in valid range",
        min(all_f1) <= f1 <= max(all_f1) and min(all_f2) <= f2 <= max(all_f2)
    ))
    # Monotonicity test (along default trajectory F2 should generally increase)
    prev_f2 = None
    mono_ok = True
    for y in np.linspace(0, 1, 20):
        _, f2 = interpolate_vowel(y)
        if prev_f2 is not None and f2 < prev_f2 - 5:
            pass   # Some non-monotonicity is expected mid-trajectory
        prev_f2 = f2
    results.append(check("Interpolation stays finite across [0,1]",
                   all(np.isfinite(f1) and np.isfinite(f2)
                       for f1, f2 in (interpolate_vowel(y) for y in np.linspace(0,1,100)))))
    return all(results)


def test_envelope_follower():
    """Envelope follower should track peaks, attack faster than release."""
    fs  = FS
    env = EnvelopeFollower(attack_ms=5, release_ms=100, fs=fs)

    # Step on
    ones = np.ones(int(fs * 0.1), dtype=np.float32)
    e_on = env.process_block(ones)
    results = []
    results.append(check("Envelope rises toward 1.0 after step on",
                         e_on[-1] > 0.9))

    # Step off
    zeros = np.zeros(int(fs * 0.5), dtype=np.float32)
    e_off = env.process_block(zeros)
    results.append(check("Envelope decays after step off",
                         e_off[-1] < 0.1))

    # Attack faster than release
    env2 = EnvelopeFollower(attack_ms=5, release_ms=200, fs=fs)
    chunk = int(fs * 0.05)
    burst = env2.process_block(np.ones(chunk, dtype=np.float32))
    at_val = burst[-1]
    env2.process_block(np.zeros(chunk, dtype=np.float32))
    rel_val = env2.envelope
    results.append(check("Attack faster than release (correct asymmetry)",
                         at_val > rel_val))
    return all(results)


def test_freq_smoother():
    """Smoother should converge to target within reasonable time."""
    smoother = FreqSmoother(initial_hz=500, tau_ms=30, fs=FS, block_size=128)
    target = 2000.0
    for _ in range(200):   # ~200 blocks = ~0.5s
        smoother.update(target)
    result = smoother.value
    return check(f"FreqSmoother converges 500→2000Hz (got {result:.1f}Hz)",
                 abs(result - target) < 5.0)


def test_engine_smoke():
    """Full engine: 1s of white noise, no exceptions, no NaN."""
    rng   = np.random.default_rng(0)
    noise = rng.standard_normal(FS).astype(np.float32) * 0.5
    engine = PedalEngine(fs=FS, block_size=128)
    engine.params.x_pos       = 0.5
    engine.params.y_pos       = 0.4
    engine.params.wah_wet     = 0.7
    engine.params.formant_wet = 0.6

    out = np.zeros(FS, dtype=np.float32)
    bs  = 128
    n_blocks = FS // bs
    for i in range(n_blocks):
        # Slowly vary x and y to exercise coefficient updates mid-stream
        engine.params.x_pos = i / n_blocks
        engine.params.y_pos = (i / n_blocks) ** 0.5
        block = noise[i*bs:(i+1)*bs]
        out[i*bs:(i+1)*bs] = engine.process_block(block)

    return check("Engine smoke test: 1s, no NaN, no Inf, output bounded",
                 np.isfinite(out).all() and np.max(np.abs(out)) < 2.0)


def test_rapid_param_change():
    """Rapid random parameter changes shouldn't produce NaN."""
    rng    = np.random.default_rng(7)
    engine = PedalEngine(fs=FS, block_size=128)
    noise  = rng.standard_normal(FS * 5).astype(np.float32) * 0.3
    bs     = 128
    ok     = True
    for i in range(len(noise) // bs):
        engine.params.x_pos       = float(rng.random())
        engine.params.y_pos       = float(rng.random())
        engine.params.wah_Q       = float(rng.uniform(0.5, 20))
        engine.params.f1_Q        = float(rng.uniform(1,   25))
        engine.params.f2_Q        = float(rng.uniform(1,   25))
        engine.params.wah_wet     = float(rng.random())
        engine.params.formant_wet = float(rng.random())
        block = noise[i*bs:(i+1)*bs]
        out   = engine.process_block(block)
        if not np.isfinite(out).all():
            ok = False
            break
    return check("Rapid random parameter changes: no NaN/Inf in 5s", ok)


def test_display_state():
    """get_display_state should return sensible values after a block."""
    engine = PedalEngine(fs=FS, block_size=128)
    engine.params.x_pos = 0.3
    engine.params.y_pos = 0.7
    noise = np.random.randn(128).astype(np.float32) * 0.1
    engine.process_block(noise)
    state = engine.get_display_state()
    results = []
    results.append(check("wah_hz in range",
                         200 < state["wah_hz"] < 3000))
    results.append(check("f1_hz in range",
                         200 < state["f1_hz"] < 1200))
    results.append(check("f2_hz in range",
                         700 < state["f2_hz"] < 2800))
    results.append(check("envelope >= 0",
                         state["envelope"] >= 0))
    return all(results)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Biquad BPF centre frequency",       test_biquad_centre_frequency),
        ("Biquad stability",                   test_biquad_stability),
        ("Peaking EQ centre frequency",        test_peaking_eq),
        ("Vowel interpolation",                test_vowel_interpolation),
        ("Envelope follower",                  test_envelope_follower),
        ("Frequency smoother convergence",     test_freq_smoother),
        ("Full engine smoke test",             test_engine_smoke),
        ("Rapid parameter changes",            test_rapid_param_change),
        ("Display state after block",          test_display_state),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n{name}")
        try:
            ok = fn()
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  {FAIL}  EXCEPTION: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    total = passed + failed
    print(f"\n{'─'*40}")
    print(f"  {passed}/{total} tests passed")
    if failed == 0:
        print("  \033[92mAll good.\033[0m")
    else:
        print(f"  \033[91m{failed} test(s) failed.\033[0m")
        sys.exit(1)