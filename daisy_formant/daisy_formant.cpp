// daisy_formant.cpp
// -----------------
// Dual-effect pedal for Daisy Pod:
//
//   KNOB 1 (left)  — Wah filter
//                    Sweeps a bandpass filter from 350 Hz (heel) to 2500 Hz (toe).
//
//   KNOB 2 (right) — Vowel / talk-box filter
//                    Sweeps through OO → OH → AH → AE → EE by turning clockwise.
//                    Uses a 5-resonator all-pole vocal tract cascade — the same
//                    physical model that makes a talk box work.
//
// Signal chain (mono, applied to both output channels):
//
//   input
//     → hard-clip saturator    (drive = 10, approaches square wave)
//     → high-shelf pre-emphasis (+6 dB above 1 kHz)
//     → wah BPF                (knob1, Q = 4, wet = 0.8)
//     → F1→F2→F3→F4→F5         (knob2, cascade of 5 all-pole resonators)
//     → output
//
// Build: same Makefile pattern as HW4.
// Flash: make && make program-dfu
//
// References: HW4_Q1.cpp (delay/LFO pattern), HW4_Q4/HW4_Q3.cpp (BiquadLocal struct)

#include "daisysp.h"
#include "daisy_pod.h"
#include <cmath>

using namespace daisysp;
using namespace daisy;

static DaisyPod pod;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static const float DRIVE          = 10.0f;   // hard-clip drive (square-wave source)
static const float PRE_EMPHASIS_DB = 6.0f;   // high-shelf gain above 1 kHz
static const float WAH_Q           = 4.0f;   // wah bandpass Q
static const float WAH_F_MIN       = 350.0f; // knob1 = 0 → 350 Hz
static const float WAH_F_MAX       = 2500.0f;// knob1 = 1 → 2500 Hz
static const float WAH_WET         = 0.8f;   // wah dry/wet blend
static const float FORMANT_WET     = 0.95f;  // formant dry/wet blend
static const int   N_FORMANTS      = 5;

// Envelope follower / consonant onset
static const float ONSET_THRESHOLD = 0.05f;  // transient level that triggers consonant snap
static const float ONSET_DEPTH     = 0.6f;   // how far toward OO to snap [0, 1]
static const float ONSET_DECAY_MS  = 30.0f;  // consonant onset release time (ms)
static const float ENV_F2_DEPTH_HZ = 150.0f; // max F2 shift from pick attack (Hz)

// Vowel formant table: 5 vowels × 5 formants (F1–F5) in Hz.
// From Hillenbrand et al. (1995), American English male averages.
//                      F1      F2      F3      F4      F5
static const float VOWEL_F[5][5] = {
    {  300.0f,   870.0f, 2240.0f, 3180.0f, 3800.0f },   // OO  "boot"
    {  500.0f,  1000.0f, 2500.0f, 3300.0f, 4000.0f },   // OH  "go"
    {  800.0f,  1200.0f, 2500.0f, 3300.0f, 4000.0f },   // AH  "father"
    {  600.0f,  1900.0f, 2600.0f, 3300.0f, 4200.0f },   // AE  "cat"
    {  300.0f,  2300.0f, 3000.0f, 3600.0f, 4300.0f },   // EE  "feet"
};

// Formant bandwidths (Hz) — real vocal tract values, NOT Q-derived.
static const float FORMANT_BW[5] = { 60.0f, 90.0f, 150.0f, 200.0f, 250.0f };

// ---------------------------------------------------------------------------
// Biquad filter — direct-form II transposed
// Identical to HW4_Q4/HW4_Q3.cpp BiquadLocal, changed to float for speed.
// ---------------------------------------------------------------------------

struct Bq {
    float b0, b1, b2, a1, a2, z1, z2;

    Bq() {
        b0 = 1.0f; b1 = 0.0f; b2 = 0.0f;
        a1 = 0.0f; a2 = 0.0f;
        Reset();
    }

    void SetCoefs(float* c) {
        // c = [b0, b1, b2, a1, a2]
        b0 = c[0]; b1 = c[1]; b2 = c[2];
        a1 = c[3]; a2 = c[4];
    }

    void Reset() { z1 = 0.0f; z2 = 0.0f; }

    float Process(float x) {
        float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }
};

// ---------------------------------------------------------------------------
// Envelope follower — peak detector with independent attack / release
// ---------------------------------------------------------------------------

struct EnvFollower {
    float attackCoeff, releaseCoeff, value;

    void Init(float attack_ms, float release_ms, float fs) {
        attackCoeff  = expf(-1.0f / (fs * attack_ms  / 1000.0f));
        releaseCoeff = expf(-1.0f / (fs * release_ms / 1000.0f));
        value = 0.0f;
    }

    float Process(float x) {
        float level = fabsf(x);
        float coeff = (level > value) ? attackCoeff : releaseCoeff;
        value = coeff * value + (1.0f - coeff) * level;
        return value;
    }
};

// ---------------------------------------------------------------------------
// Filter bank (stereo pairs)
// ---------------------------------------------------------------------------

static Bq wahL,      wahR;       // wah bandpass
static Bq shelfL,    shelfR;     // high-shelf pre-emphasis (fixed)
static Bq resonL[N_FORMANTS];   // formant cascade — left
static Bq resonR[N_FORMANTS];   // formant cascade — right

// Envelope follower state
static EnvFollower envFast, envSlow;  // transient = fast − slow
static float onsetEnv       = 0.0f;  // consonant onset depth [0,1], decays per-sample
static float onsetDecay     = 0.0f;  // per-sample decay coeff, set in main()
static float blockTransient = 0.0f;  // peak transient accumulated over current block

// ---------------------------------------------------------------------------
// Coefficient design functions
// ---------------------------------------------------------------------------

// Wah bandpass — Audio EQ Cookbook BPF, constant 0 dB peak gain.
void DesignBPF(float* c, float fc, float Q, float fs)
{
    float w0    = 2.0f * M_PI * fc / fs;
    float alpha = sinf(w0) / (2.0f * Q);
    float a0    = 1.0f + alpha;
    c[0] =  sinf(w0) / 2.0f / a0;   // b0
    c[1] =  0.0f;                    // b1
    c[2] = -sinf(w0) / 2.0f / a0;   // b2
    c[3] = -2.0f * cosf(w0)  / a0;  // a1
    c[4] =  (1.0f - alpha)   / a0;  // a2
}

// All-pole resonator — one vocal tract formant.
// H(z) = b0 / (1 + a1*z^-1 + a2*z^-2)
// b0 = A(1) gives unity DC gain; formant peak rises above 0 dB.
// Sections applied in CASCADE to create anti-formant notches between peaks.
void DesignResonator(float* c, float fc, float bw, float fs)
{
    float r  = expf(-M_PI * bw / fs);
    float w0 = 2.0f * M_PI * fc / fs;
    float a1 = -2.0f * r * cosf(w0);
    float a2 = r * r;
    float b0 = 1.0f + a1 + a2;      // A(1): H(DC) = b0/A(1) = 1
    c[0] = b0;   // b0
    c[1] = 0.0f; // b1
    c[2] = 0.0f; // b2
    c[3] = a1;   // a1
    c[4] = a2;   // a2
}

// High-shelf filter — Audio EQ Cookbook §HS, slope = 1.
void DesignHighShelf(float* c, float fc, float gain_db, float fs)
{
    float A   = powf(10.0f, gain_db / 40.0f);
    float w0  = 2.0f * M_PI * fc / fs;
    float cw  = cosf(w0);
    float sA  = sqrtf(A);
    float alp = sinf(w0) / 2.0f * sqrtf(2.0f); // slope = 1

    float b0 =       A * ((A+1.0f) + (A-1.0f)*cw + 2.0f*sA*alp);
    float b1 = -2.0f*A * ((A-1.0f) + (A+1.0f)*cw);
    float b2 =       A * ((A+1.0f) + (A-1.0f)*cw - 2.0f*sA*alp);
    float a0 =            (A+1.0f) - (A-1.0f)*cw + 2.0f*sA*alp;
    float a1 =  2.0f *   ((A-1.0f) - (A+1.0f)*cw);
    float a2 =            (A+1.0f) - (A-1.0f)*cw - 2.0f*sA*alp;

    c[0] = b0/a0; c[1] = b1/a0; c[2] = b2/a0;
    c[3] = a1/a0; c[4] = a2/a0;
}

// ---------------------------------------------------------------------------
// Vowel interpolation
// ---------------------------------------------------------------------------

// Map pos [0, 1] to 5 formant frequencies by linear interpolation
// along the OO → OH → AH → AE → EE trajectory.
void InterpFormants(float pos, float* freqs_out)
{
    pos         = fclamp(pos, 0.0f, 1.0f);
    float scaled = pos * 4.0f;            // 4 segments, 5 vowels
    int   idx    = (int)scaled;
    if(idx >= 4) idx = 3;
    float t = scaled - (float)idx;

    for(int k = 0; k < N_FORMANTS; k++)
        freqs_out[k] = VOWEL_F[idx][k] + t * (VOWEL_F[idx+1][k] - VOWEL_F[idx][k]);
}

// ---------------------------------------------------------------------------
// Control update — called once per audio block
// ---------------------------------------------------------------------------

static float k1 = 0.5f, k2 = 0.0f;

void UpdateKnobs()
{
    k1 = pod.knob1.Process();   // 0..1 → wah frequency
    k2 = pod.knob2.Process();   // 0..1 → vowel position (OO → EE)

    float fs = pod.AudioSampleRate();
    float coefs[5];

    // Wah BPF: map knob1 linearly to WAH_F_MIN..WAH_F_MAX
    float wahFreq = WAH_F_MIN + k1 * (WAH_F_MAX - WAH_F_MIN);
    DesignBPF(coefs, wahFreq, WAH_Q, fs);
    wahL.SetCoefs(coefs);
    wahR.SetCoefs(coefs);

    // Consonant onset snaps vowel position toward OO on pick attack.
    // onsetEnv decays per-sample in AudioCallback; here we just read it.
    float vowelPos = k2 * (1.0f - onsetEnv * ONSET_DEPTH);

    // Formant resonators: interpolate onset-adjusted vowel position
    float freqs[N_FORMANTS];
    InterpFormants(vowelPos, freqs);

    // F2 coupling to wah
    freqs[1] = fclamp(freqs[1] + 0.15f * (wahFreq - 1200.0f), 300.0f, 3500.0f);
    // F2 modulation from pick attack transient
    freqs[1] = fclamp(freqs[1] + blockTransient * ENV_F2_DEPTH_HZ, 300.0f, 3500.0f);

    for(int i = 0; i < N_FORMANTS; i++) {
        DesignResonator(coefs, freqs[i], FORMANT_BW[i], fs);
        resonL[i].SetCoefs(coefs);
        resonR[i].SetCoefs(coefs);
    }

    blockTransient = 0.0f;  // reset accumulator for the next block
}

void Controls()
{
    pod.ProcessAnalogControls();
    pod.ProcessDigitalControls();
    UpdateKnobs();
}

// ---------------------------------------------------------------------------
// Audio callback
// ---------------------------------------------------------------------------

void AudioCallback(AudioHandle::InterleavingInputBuffer  in,
                   AudioHandle::InterleavingOutputBuffer out,
                   size_t                                size)
{
    Controls();

    for(size_t i = 0; i < size; i += 2)
    {
        float inl = in[i];
        float inr = in[i + 1];

        // ── 1. Hard-clip saturation ──────────────────────────────────────
        // Wah position gates drive: higher wah (brighter) = more saturation.
        float effectiveDrive = DRIVE * (1.0f + 0.3f * k1);
        float drvL = fclamp(inl * effectiveDrive, -1.0f, 1.0f);
        float drvR = fclamp(inr * effectiveDrive, -1.0f, 1.0f);

        // ── Envelope follower / consonant onset ─────────────────────────
        // Mono abs value feeds two followers with different release times.
        // Transient = fast − slow: non-zero only on pick attacks.
        float envIn    = 0.5f * (fabsf(drvL) + fabsf(drvR));
        float fast     = envFast.Process(envIn);
        float slow     = envSlow.Process(envIn);
        float transient = (fast > slow) ? (fast - slow) : 0.0f;
        if(transient > blockTransient) blockTransient = transient;

        // Trigger onset on sharp transient; don't re-trigger while decaying.
        if(transient > ONSET_THRESHOLD && onsetEnv < 0.1f)
            onsetEnv = 1.0f;
        onsetEnv *= onsetDecay;

        // ── 2. High-shelf pre-emphasis ──────────────────────────────────
        // Tilts the spectrum so F3/F4/F5 (2–4 kHz) have enough energy.
        float empL = shelfL.Process(drvL);
        float empR = shelfR.Process(drvR);

        // ── 3. Wah BPF (knob1) ─────────────────────────────────────────
        float wahOutL = wahL.Process(empL);
        float wahOutR = wahR.Process(empR);
        float wahMixL = (1.0f - WAH_WET) * empL + WAH_WET * wahOutL;
        float wahMixR = (1.0f - WAH_WET) * empR + WAH_WET * wahOutR;

        // ── 4. Vocal tract: F1→F2→F3→F4→F5 cascade (knob2) ───────────
        // Each resonator's output feeds the next (series, not parallel).
        // This creates anti-formant notches between peaks — the key
        // difference from a wah pedal, and what gives vowels their identity.
        float vtL = wahMixL;
        float vtR = wahMixR;
        for(int k = 0; k < N_FORMANTS; k++) {
            vtL = resonL[k].Process(vtL);
            vtR = resonR[k].Process(vtR);
        }

        // ── 5. Wet/dry blend ───────────────────────────────────────────
        // Dry signal is the pre-drive input for a clean bypass at wet=0.
        float outL = (1.0f - FORMANT_WET) * inl + FORMANT_WET * vtL;
        float outR = (1.0f - FORMANT_WET) * inr + FORMANT_WET * vtR;

        // ── 6. Safety clip ─────────────────────────────────────────────
        out[i]     = fclamp(outL, -1.0f, 1.0f);
        out[i + 1] = fclamp(outR, -1.0f, 1.0f);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(void)
{
    pod.Init();
    pod.SetAudioBlockSize(4);   // 4-sample blocks, same as HW4

    float fs = pod.AudioSampleRate();
    float coefs[5];

    // Envelope followers: fast tracks attacks (5ms/80ms), slow tracks sustain (5ms/500ms).
    // Transient delta = fast − slow is non-zero only on pick attacks.
    envFast.Init(5.0f,  80.0f, fs);
    envSlow.Init(5.0f, 500.0f, fs);
    onsetDecay = expf(-1.0f / (fs * ONSET_DECAY_MS / 1000.0f));

    // High-shelf pre-emphasis: fixed +6 dB above 1 kHz.
    // Computed once here since it never changes.
    DesignHighShelf(coefs, 1000.0f, PRE_EMPHASIS_DB, fs);
    shelfL.SetCoefs(coefs);
    shelfR.SetCoefs(coefs);

    // Initialize wah and formants with knobs at their default positions
    // (heel position: wah at min freq, vowel at OO).
    float wahFreq = WAH_F_MIN;
    DesignBPF(coefs, wahFreq, WAH_Q, fs);
    wahL.SetCoefs(coefs);
    wahR.SetCoefs(coefs);

    float freqs[N_FORMANTS];
    InterpFormants(0.0f, freqs);   // OO at startup
    for(int i = 0; i < N_FORMANTS; i++) {
        DesignResonator(coefs, freqs[i], FORMANT_BW[i], fs);
        resonL[i].SetCoefs(coefs);
        resonR[i].SetCoefs(coefs);
    }

    pod.StartAdc();
    pod.StartAudio(AudioCallback);

    while(1) {}
}