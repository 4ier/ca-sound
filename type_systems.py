#!/usr/bin/env python3
"""Phase 17: Type Systems — inference, subtyping, dependent types.

Three pieces:
1. Hindley-Milner (55s, stereo) — Type inference as harmonic convergence.
   Polymorphic type variables are wavering tones (frequency uncertain),
   constraints narrow the spectrum, unification snaps to a clear chord.
2. Subtype Lattice (50s, stereo) — A hierarchy from Top to Bottom.
   Covariance ascends, contravariance descends. Subtypes share the
   harmonic DNA of their supertypes.
3. Curry-Howard (55s, stereo) — Propositions-as-types, proofs-as-programs.
   Logical connectives become harmonic operators.
"""
import numpy as np
import os

SR = 44100


def write_wav(path, data):
    import wave
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    data = np.clip(data, -1, 1)
    pcm = (data * 32767).astype(np.int16)
    with wave.open(path, 'w') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"  → {path} ({len(data)/SR:.1f}s)")


def sine(freq, t, phase=0.0):
    return np.sin(2 * np.pi * freq * t + phase)


def fm_tone(carrier, mod_freq, mod_depth, t):
    return np.sin(2 * np.pi * carrier * t + mod_depth * np.sin(2 * np.pi * mod_freq * t))


def envelope(n, attack=0.01, release=0.05):
    env = np.ones(n)
    att = int(attack * SR)
    rel = int(release * SR)
    if att > 0:
        env[:att] = np.linspace(0, 1, att)
    if rel > 0:
        env[-rel:] = np.linspace(1, 0, rel)
    return env


# ─────────────────────────────────────────────────
# 1. Hindley-Milner Inference (vectorized)
# ─────────────────────────────────────────────────
def hindley_milner():
    """Type inference as harmonic convergence.

    5 type variables each start as a band of uncertain frequencies
    (chorus-like wavering). Constraints arrive one by one, each narrowing
    the band until the variable snaps to a definite pitch.
    """
    dur = 55.0
    n = int(dur * SR)
    out = np.zeros((n, 2))
    t = np.arange(n) / SR

    # Type variables: (name, target_freq, snap_time, pan, waver_rate)
    variables = [
        ('alpha',   220.0,   8.0, -0.7, 1.5),
        ('beta',    277.18, 16.0, -0.3, 1.9),
        ('gamma',   329.63, 24.0,  0.0, 1.3),
        ('delta',   440.0,  32.0,  0.3, 2.1),
        ('epsilon', 277.18, 38.0,  0.7, 1.7),
    ]

    for idx, (name, target, snap_t, pan, waver_rate) in enumerate(variables):
        # Phase masks
        pre_mask = t < (snap_t - 2.0)
        narrow_mask = (t >= (snap_t - 2.0)) & (t < snap_t)
        post_mask = t >= snap_t

        signal = np.zeros(n)
        amp = np.zeros(n)

        # Pre-constraint: chorus of detuned voices (uncertainty)
        if np.any(pre_mask):
            uncertainty = 0.30
            center = target * (1.0 + uncertainty * 0.5 * np.sin(2 * np.pi * waver_rate * t))
            for k, detune in enumerate([-0.12, 0.0, 0.12]):
                freq = center * (1.0 + detune * uncertainty)
                signal += pre_mask * np.sin(2 * np.pi * freq * t + k * 1.3) * 0.33
            amp += pre_mask * 0.12

        # Narrowing: uncertainty shrinks over 2 seconds
        if np.any(narrow_mask):
            progress = np.clip((t - (snap_t - 2.0)) / 2.0, 0, 1)
            uncertainty = 0.30 * (1.0 - progress)
            center = target * (1.0 + uncertainty * 0.3 * np.sin(2 * np.pi * 2.0 * t))
            for k, detune in enumerate([-0.12, 0.0, 0.12]):
                freq = center * (1.0 + detune * uncertainty)
                signal += narrow_mask * np.sin(2 * np.pi * freq * t + k * 1.3) * 0.33
            amp += narrow_mask * (0.12 + 0.06 * progress)

        # Post-constraint: locked tone with light vibrato + 2nd harmonic
        if np.any(post_mask):
            vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * 5.0 * t)
            freq = target * vibrato
            tone = 0.7 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
            signal += post_mask * tone
            amp += post_mask * 0.18

        # Snap click
        click_mask = np.abs(t - snap_t) < 0.015
        signal += click_mask * 0.8 * np.sin(2 * np.pi * 2000 * t) * np.exp(-200 * np.abs(t - snap_t))

        # Apply to stereo
        left_gain = 0.5 * (1.0 - pan)
        right_gain = 0.5 * (1.0 + pan)
        out[:, 0] += signal * amp * left_gain
        out[:, 1] += signal * amp * right_gain

    # Principal type chord (last 11 seconds): all variables ground → A major
    chord_start = 44.0
    chord_mask = t >= chord_start
    chord_progress = np.clip((t - chord_start) / 11.0, 0, 1)
    chord_amp = 0.06 * np.minimum(1.0, chord_progress * 3.0)
    fade_zone = chord_progress > 0.7
    chord_amp = np.where(fade_zone, chord_amp * (1.0 - chord_progress) / 0.3, chord_amp)
    chord_amp *= chord_mask

    for freq in [220.0, 277.18, 329.63, 440.0]:
        tone = np.sin(2 * np.pi * freq * t) * chord_amp
        out[:, 0] += tone
        out[:, 1] += tone

    # 55Hz sub drone
    drone = sine(55.0, t) * 0.04 * envelope(n, attack=2.0, release=3.0)
    out[:, 0] += drone
    out[:, 1] += drone

    # Global envelope
    env_g = envelope(n, attack=0.5, release=2.0)
    out[:, 0] *= env_g
    out[:, 1] *= env_g

    return out


# ─────────────────────────────────────────────────
# 2. Subtype Lattice (vectorized)
# ─────────────────────────────────────────────────
def subtype_lattice():
    """A hierarchy of types from Top to Bottom.

    Traverse the lattice depth-first. At each node, the type sounds for
    ~3-4 seconds. Subtypes inherit their parent's harmonics but add
    their own character.
    """
    dur = 50.0
    n = int(dur * SR)
    out = np.zeros((n, 2))
    t = np.arange(n) / SR

    rng = np.random.RandomState(42)

    # Lattice nodes: (name, base_freq, pan, start, duration, type)
    # type: 'additive', 'fm', 'pulse'
    nodes = [
        ('Top',      165.0, 0.0,  0.0,  4.0, 'additive', [(k, 0.7/k) for k in range(1, 16)]),
        ('Number',   220.0, -0.4, 4.5,  3.5, 'additive', [(k, 0.8/k) for k in range(1, 9)]),
        ('Int',      261.6, -0.6, 8.5,  3.0, 'additive', [(k, 0.9/k) for k in range(1, 12, 2)]),
        ('Float',    293.7, -0.3, 12.0, 3.0, 'additive', [(k, 0.7/k) for k in range(1, 14)]),
        ('Sequence', 220.0, 0.3,  15.5, 3.5, 'additive', [(1, 0.6), (3, 0.4), (5, 0.25), (7, 0.15)]),
        ('List',     329.6, 0.5,  19.5, 3.0, 'pulse_regular', [(1, 0.7), (2, 0.5), (4, 0.3)]),
        ('Stream',   349.2, 0.6,  23.0, 3.0, 'pulse_random', [(1, 0.6), (3, 0.5), (6, 0.3), (11, 0.15)]),
        ('Function', 196.0, 0.0,  26.5, 3.5, 'fm', []),
        ('Pure',     247.0, -0.2, 30.5, 3.0, 'fm_clean', []),
        ('Effect',   277.2, 0.2,  34.0, 3.0, 'fm_noisy', []),
        ('Bottom',   55.0,  0.0,  37.5, 3.0, 'additive', [(1, 0.1)]),
    ]

    for name, base_freq, pan, start, node_dur, synth_type, harmonics in nodes:
        i0 = int(start * SR)
        i1 = min(int((start + node_dur) * SR), n)
        seg_n = i1 - i0
        if seg_n <= 0:
            continue

        seg_t = t[i0:i1]
        seg_env = envelope(seg_n, attack=0.08, release=0.3)
        left_g = 0.5 * (1.0 - pan)
        right_g = 0.5 * (1.0 + pan)

        if synth_type.startswith('fm'):
            if synth_type == 'fm':
                tone = fm_tone(base_freq, base_freq * 1.5, 2.0, seg_t) * 0.14 * seg_env
            elif synth_type == 'fm_clean':
                tone = fm_tone(base_freq, base_freq * 2.0, 1.5, seg_t) * 0.14 * seg_env
            else:  # fm_noisy
                tone = fm_tone(base_freq, base_freq * 1.618, 4.0, seg_t) * 0.14 * seg_env
                tone += rng.randn(seg_n) * 0.03 * seg_env

            out[i0:i1, 0] += tone * left_g
            out[i0:i1, 1] += tone * right_g
            continue

        # Additive synthesis
        signal = np.zeros(seg_n)
        for h_num, h_amp in harmonics:
            signal += h_amp * sine(base_freq * h_num, seg_t)
        signal *= 0.12 * seg_env

        # Pulse overlay for List/Stream
        if synth_type == 'pulse_regular':
            local_t = seg_t - start
            pulse = np.zeros(seg_n)
            pulse_period = 0.15
            pulse_phase = local_t % pulse_period
            pulse_on = pulse_phase < 0.02
            pulse += pulse_on * 0.15 * np.exp(-50 * pulse_phase) * np.sin(2 * np.pi * 2000 * seg_t)
            out[i0:i1, 0] += pulse
            out[i0:i1, 1] += pulse
        elif synth_type == 'pulse_random':
            triggers = rng.random(seg_n) < 0.0005
            pulse = triggers.astype(float) * 0.12 * np.sin(2 * np.pi * 2000 * seg_t)
            out[i0:i1, 0] += pulse
            out[i0:i1, 1] += pulse

        out[i0:i1, 0] += signal * left_g
        out[i0:i1, 1] += signal * right_g

    # Transition clicks
    for i_node in range(1, len(nodes)):
        click_t = nodes[i_node][3]
        ci = int(click_t * SR)
        if ci < n - 100:
            click_samples = np.arange(100)
            click = np.exp(-click_samples / 10.0) * 0.15 * np.sin(2 * np.pi * 1500 * click_samples / SR)
            out[ci:ci+100, 0] += click
            out[ci:ci+100, 1] += click

    # Coda chord (41-50s)
    coda_mask = t >= 41.0
    coda_env = envelope(n, attack=1.0, release=3.0)
    for cf in [165.0, 220.0, 261.6, 329.6]:
        tone = sine(cf, t) * 0.08 * coda_env * coda_mask
        out[:, 0] += tone
        out[:, 1] += tone

    # Sub drone
    drone = sine(55.0, t) * 0.03 * envelope(n, attack=1.0, release=2.0)
    out[:, 0] += drone
    out[:, 1] += drone

    env_g = envelope(n, attack=0.3, release=2.0)
    out[:, 0] *= env_g
    out[:, 1] *= env_g

    return out


# ─────────────────────────────────────────────────
# 3. Curry-Howard Correspondence (vectorized)
# ─────────────────────────────────────────────────
def curry_howard():
    """Propositions-as-types, proofs-as-programs.

    Section A (0-18s): Conjunction (A ∧ B)
    Section B (18-36s): Implication (A → B)
    Section C (36-55s): Disjunction case analysis (A ∨ B → C)
    """
    dur = 55.0
    n = int(dur * SR)
    out = np.zeros((n, 2))
    t = np.arange(n) / SR

    freq_A = 220.0    # A3
    freq_B = 329.63   # E4 (perfect fifth)
    freq_C = 392.0    # G4

    # ── Section A: Conjunction (A ∧ B) 0-18s ──
    # Proof of A builds harmonics over 0-8s, proof of B over 8-15s, chord 15-18s
    sa_mask = t < 18.0

    # Proof of A: harmonics accumulate over 0-8s
    a_building = (t < 8.0) & (t >= 0.0)
    a_done = t >= 8.0
    harmonic_count_a = np.clip(t / 1.5 + 1, 1, 6).astype(int)

    sig_a = np.zeros(n)
    for h in range(1, 7):
        h_active = harmonic_count_a >= h
        sig_a += h_active * (0.7 / h) * sine(freq_A * h, t)

    amp_a_build = np.clip(t / 0.5, 0, 1) * 0.15 * a_building
    amp_a_done = a_done * sa_mask * 0.15

    # Conjunction fade at end
    conj_fade = np.where(t > 17.0, np.clip(1.0 - (t - 17.0), 0, 1), 1.0)
    amp_a_conj = np.where(t >= 15.0, 0.18 * conj_fade * sa_mask, 0.0)

    # Proof of B: builds over 8-15s
    b_building = (t >= 8.0) & (t < 15.0)
    b_progress = np.clip((t - 8.0) / 7.0, 0, 1)
    harmonic_count_b = np.clip(b_progress * 6 + 1, 1, 6).astype(int)

    sig_b = np.zeros(n)
    for h in range(1, 7):
        h_active = harmonic_count_b >= h
        sig_b += h_active * (0.7 / h) * sine(freq_B * h, t)

    amp_b_build = np.clip((t - 8.0) / 0.5, 0, 1) * 0.15 * b_building
    amp_b_conj = np.where(t >= 15.0, 0.18 * conj_fade * sa_mask, 0.0)

    # Combine section A
    total_a = sig_a * (amp_a_build + amp_a_done * np.where(b_building, 0.15, 0) + amp_a_conj)
    total_b = sig_b * (amp_b_build + amp_b_conj)

    # Stereo: A left-biased, B right-biased
    out[:, 0] += sa_mask * (total_a * 0.7 + total_b * 0.3)
    out[:, 1] += sa_mask * (total_a * 0.3 + total_b * 0.7)

    # ── Section B: Implication (A → B) 18-36s ──
    sb_mask = (t >= 18.0) & (t < 36.0)
    local_b = t - 18.0

    # Phase 1 (0-6s): Present hypothesis A
    p1 = sb_mask & (local_b < 6.0)
    va_hyp = sine(freq_A, t) * 0.18 * (1.0 + 0.1 * sine(5.0, t))
    out[:, 0] += p1 * va_hyp * 0.6
    out[:, 1] += p1 * va_hyp * 0.4

    # Phase 2 (6-12s): Arrow construction — FM modulation A→B
    p2 = sb_mask & (local_b >= 6.0) & (local_b < 12.0)
    arrow_prog = np.clip((local_b - 6.0) / 6.0, 0, 1)
    carrier = freq_A + (freq_B - freq_A) * arrow_prog
    mod_depth = arrow_prog * 3.0
    arrow_sig = fm_tone(carrier, freq_A, mod_depth, t) * 0.16
    out[:, 0] += p2 * arrow_sig * 0.5
    out[:, 1] += p2 * arrow_sig * 0.5

    # Phase 3 (12-15s): Modus ponens — apply arrow, A consumed, B emerges
    p3 = sb_mask & (local_b >= 12.0) & (local_b < 15.0)
    mp_prog = np.clip((local_b - 12.0) / 3.0, 0, 1)
    va_fade = sine(freq_A, t) * 0.15 * (1.0 - mp_prog)
    vb_emerge = sine(freq_B, t) * 0.18 * mp_prog
    for h in [2, 3]:
        vb_emerge += (0.3 / h) * sine(freq_B * h, t) * 0.18 * mp_prog
    out[:, 0] += p3 * (va_fade * 0.6 + vb_emerge * 0.4)
    out[:, 1] += p3 * (va_fade * 0.4 + vb_emerge * 0.6)

    # Phase 4 (15-18s): B established, fading
    p4 = sb_mask & (local_b >= 15.0)
    impl_fade = np.clip(1.0 - (local_b - 16.5) / 1.5, 0, 1)
    vb_est = np.zeros(n)
    for h in range(1, 5):
        vb_est += (0.6 / h) * sine(freq_B * h, t)
    vb_est *= 0.18 * impl_fade
    out[:, 0] += p4 * vb_est * 0.4
    out[:, 1] += p4 * vb_est * 0.6

    # ── Section C: Disjunction (A ∨ B → C) 36-55s ──
    sc_mask = (t >= 36.0) & (t < 55.0)
    local_c = t - 36.0

    # Phase 1 (0-7s): Two paths glide toward C
    c1 = sc_mask & (local_c < 7.0)
    c1_prog = np.clip(local_c / 7.0, 0, 1)
    freq_left = freq_A + (freq_C - freq_A) * c1_prog
    freq_right = freq_B + (freq_C - freq_B) * c1_prog
    num_h_c = np.clip(c1_prog * 4 + 1, 1, 4).astype(int)

    vl = sine(freq_left, t) * 0.16
    vr = sine(freq_right, t) * 0.16
    for h in range(2, 5):
        h_on = num_h_c >= h
        vl += h_on * (0.4 / h) * sine(freq_left * h, t) * 0.16
        vr += h_on * (0.4 / h) * sine(freq_right * h, t) * 0.16

    out[:, 0] += c1 * vl
    out[:, 1] += c1 * vr

    # Phase 2 (7-12s): Convergence to C
    c2 = sc_mask & (local_c >= 7.0) & (local_c < 12.0)
    conv_prog = np.clip((local_c - 7.0) / 5.0, 0, 1)
    freq_conv_l = freq_C + (freq_A - freq_C) * (1.0 - conv_prog) * 0.2
    freq_conv_r = freq_C + (freq_B - freq_C) * (1.0 - conv_prog) * 0.2
    out[:, 0] += c2 * sine(freq_conv_l, t) * 0.17
    out[:, 1] += c2 * sine(freq_conv_r, t) * 0.17

    # Phase 3 (12-19s): Unified C chord + A∨B→C three-note coda
    c3 = sc_mask & (local_c >= 12.0)
    c3_fade = np.clip(1.0 - (local_c - 16.0) / 3.0, 0, 1)
    vc = np.zeros(n)
    for h in range(1, 7):
        vc += (0.7 / h) * sine(freq_C * h, t)
    vc *= 0.15 * c3_fade
    out[:, 0] += c3 * (vc + sine(freq_A, t) * 0.06 * c3_fade)
    out[:, 1] += c3 * (vc + sine(freq_B, t) * 0.06 * c3_fade)

    # Global 55Hz drone
    drone = sine(55.0, t) * 0.035 * envelope(n, attack=1.5, release=3.0)
    out[:, 0] += drone
    out[:, 1] += drone

    # Global envelope
    env_g = envelope(n, attack=0.3, release=2.5)
    out[:, 0] *= env_g
    out[:, 1] *= env_g

    return out


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────
if __name__ == '__main__':
    outdir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(outdir, exist_ok=True)

    print("Phase 17: Type Systems")
    print("=" * 50)

    print("\n1. Hindley-Milner Inference")
    data = hindley_milner()
    write_wav(os.path.join(outdir, 'type_1_hindley_milner.wav'), data)

    print("\n2. Subtype Lattice")
    data = subtype_lattice()
    write_wav(os.path.join(outdir, 'type_2_subtype_lattice.wav'), data)

    print("\n3. Curry-Howard Correspondence")
    data = curry_howard()
    write_wav(os.path.join(outdir, 'type_3_curry_howard.wav'), data)

    print("\nDone! 3 wav files in output/")
