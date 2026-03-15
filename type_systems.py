#!/usr/bin/env python3
"""Phase 17: Type Systems — inference, subtyping, dependent types.

Three pieces:
1. Hindley-Milner (55s, stereo) — Type inference as harmonic convergence.
   Polymorphic type variables are wavering tones (frequency uncertain),
   constraints narrow the spectrum, unification snaps to a clear chord.
2. Subtype Lattice (50s, stereo) — A hierarchy from Top (all harmonics)
   to Bottom (silence). Covariance ascends, contravariance descends.
   Subtypes share the harmonic DNA of their supertypes.
3. Curry-Howard (55s, stereo) — Propositions-as-types, proofs-as-programs.
   Logical connectives ∧∨→ become harmonic operators. A proof construction
   builds a chord; an incomplete proof leaves dissonance unresolved.
"""
import numpy as np
import os

SR = 44100


def write_wav(path, data):
    import wave, struct
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


def mix(a, b, alpha):
    return a * (1 - alpha) + b * alpha


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
# 1. Hindley-Milner Inference
# ─────────────────────────────────────────────────
def hindley_milner():
    """Type inference as harmonic convergence.

    5 type variables α, β, γ, δ, ε each start as a band of uncertain
    frequencies (chorus-like wavering). Constraints arrive one by one,
    each narrowing the band until the variable "snaps" to a definite pitch.

    The inference engine processes constraints in order:
      - α ~ Int       (snap to 220Hz A3)
      - β ~ α → γ     (β gets arrow-type FM timbre, γ partially constrained)
      - γ ~ Bool      (snap to 330Hz E4)
      - δ ~ List α    (snap to 440Hz A4, octave of α)
      - ε ~ β         (unify: ε takes β's identity)

    When all variables are ground, the principal type emerges as a major chord.
    """
    dur = 55.0
    n = int(dur * SR)
    out = np.zeros((n, 2))
    t = np.arange(n) / SR

    # Target frequencies for each type variable after inference
    targets = {
        'alpha': 220.0,   # Int → A3
        'beta':  277.18,  # Arrow type → C#4
        'gamma': 329.63,  # Bool → E4
        'delta': 440.0,   # List Int → A4
        'epsilon': 277.18 # Unifies with beta → C#4
    }

    # Constraint arrival times (seconds)
    snap_times = {
        'alpha': 8.0,
        'beta': 16.0,
        'gamma': 24.0,
        'delta': 32.0,
        'epsilon': 38.0,
    }

    # Stereo positions (left=-1, right=+1)
    pans = {
        'alpha': -0.7,
        'beta': -0.3,
        'gamma': 0.0,
        'delta': 0.3,
        'epsilon': 0.7,
    }

    # For each variable, generate a signal that starts as uncertain wavering
    # and snaps to a definite frequency at its constraint time
    for var_name, target_freq in targets.items():
        snap_t = snap_times[var_name]
        pan = pans[var_name]

        # Uncertainty band: ±30% of target initially, narrowing over time
        for i in range(n):
            tc = t[i]
            if tc < snap_t - 2.0:
                # Pre-constraint: wavering in uncertainty band
                uncertainty = 0.30
                # Multiple slightly detuned copies (chorus = uncertainty)
                waver_rate = 1.5 + 0.3 * hash(var_name) % 7
                center = target_freq * (1.0 + uncertainty * 0.5 * np.sin(2 * np.pi * waver_rate * tc))
                # 3 detuned voices
                v = 0.0
                for k, detune in enumerate([-0.12, 0.0, 0.12]):
                    freq = center * (1.0 + detune * uncertainty)
                    v += np.sin(2 * np.pi * freq * tc + k * 1.3) * 0.33
                amp = 0.12
            elif tc < snap_t:
                # Narrowing: uncertainty shrinks over 2 seconds
                progress = (tc - (snap_t - 2.0)) / 2.0
                uncertainty = 0.30 * (1.0 - progress)
                center = target_freq * (1.0 + uncertainty * 0.3 * np.sin(2 * np.pi * 2.0 * tc))
                v = 0.0
                for k, detune in enumerate([-0.12, 0.0, 0.12]):
                    freq = center * (1.0 + detune * uncertainty)
                    v += np.sin(2 * np.pi * freq * tc + k * 1.3) * 0.33
                amp = 0.12 + 0.06 * progress
            else:
                # Post-constraint: locked to target, pure tone with light vibrato
                vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * 5.0 * tc)
                freq = target_freq * vibrato
                # Richer: fundamental + 2nd harmonic
                v = 0.7 * np.sin(2 * np.pi * freq * tc) + 0.3 * np.sin(2 * np.pi * freq * 2 * tc)
                amp = 0.18

            # Snap click at constraint moment
            if abs(tc - snap_t) < 0.01:
                v += 0.8 * np.sin(2 * np.pi * 2000 * tc) * np.exp(-300 * abs(tc - snap_t))

            # Apply to stereo
            left_gain = 0.5 * (1.0 - pan)
            right_gain = 0.5 * (1.0 + pan)
            out[i, 0] += v * amp * left_gain
            out[i, 1] += v * amp * right_gain

    # Section E: Principal type chord (last 10 seconds)
    # All variables ground → A major chord (A3, C#4, E4, A4)
    chord_start = 44.0
    chord_dur = 11.0
    for i in range(n):
        tc = t[i]
        if tc >= chord_start:
            progress = (tc - chord_start) / chord_dur
            chord_amp = 0.06 * min(1.0, progress * 3.0)
            if progress > 0.7:
                chord_amp *= (1.0 - progress) / 0.3  # fade out
            for freq in [220.0, 277.18, 329.63, 440.0]:
                v = np.sin(2 * np.pi * freq * tc) * chord_amp
                out[i, 0] += v
                out[i, 1] += v

    # Drone: 55Hz sub
    drone = sine(55.0, t) * 0.04
    env_d = envelope(n, attack=2.0, release=3.0)
    out[:, 0] += drone * env_d
    out[:, 1] += drone * env_d

    # Global envelope
    env_g = envelope(n, attack=0.5, release=2.0)
    out[:, 0] *= env_g
    out[:, 1] *= env_g

    return out


# ─────────────────────────────────────────────────
# 2. Subtype Lattice
# ─────────────────────────────────────────────────
def subtype_lattice():
    """A hierarchy of types from Top to Bottom.

    The type lattice:
        Top (all harmonics, white-noise-ish)
         ├── Number (harmonics 1-8)
         │    ├── Int (odd harmonics: square-wave-like)
         │    └── Float (all harmonics, smooth: sawtooth-like)
         ├── Sequence (rhythmic pulse train)
         │    ├── List (regular pulse)
         │    └── Stream (irregular pulse, Poisson-like)
         └── Function (FM timbre)
              ├── Pure (clean FM)
              └── Effect (noisy FM)
        Bottom (silence)

    We traverse the lattice depth-first. At each node, the type sounds for
    ~4 seconds. Subtypes inherit their parent's harmonics (covariance) but add
    their own character. Traversal goes down (pitch descends) and back up
    (pitch ascends, contravariance).
    """
    dur = 50.0
    n = int(dur * SR)
    out = np.zeros((n, 2))
    t = np.arange(n) / SR

    # Lattice nodes: (name, base_freq, stereo_pan, start_time, duration, harmonic_recipe)
    # harmonic_recipe: list of (harmonic_number, amplitude) pairs
    nodes = [
        # Top: all harmonics (rich, broad)
        ('Top',      165.0, 0.0,  0.0,  4.0, [(k, 0.7/k) for k in range(1, 16)]),
        # Number: harmonics 1-8
        ('Number',   220.0, -0.4, 4.5,  3.5, [(k, 0.8/k) for k in range(1, 9)]),
        # Int: odd harmonics only (square-ish)
        ('Int',      261.6, -0.6, 8.5,  3.0, [(k, 0.9/k) for k in range(1, 12, 2)]),
        # Float: dense harmonics (sawtooth-ish)
        ('Float',    293.7, -0.3, 12.0, 3.0, [(k, 0.7/k) for k in range(1, 14)]),
        # Back to Number then Sequence
        ('Sequence', 220.0, 0.3,  15.5, 3.5, [(1, 0.6), (3, 0.4), (5, 0.25), (7, 0.15)]),
        # List: regular pulse overlay
        ('List',     329.6, 0.5,  19.5, 3.0, [(1, 0.7), (2, 0.5), (4, 0.3)]),
        # Stream: irregular
        ('Stream',   349.2, 0.6,  23.0, 3.0, [(1, 0.6), (3, 0.5), (6, 0.3), (11, 0.15)]),
        # Function: FM timbre
        ('Function', 196.0, 0.0,  26.5, 3.5, []),  # FM, not additive
        # Pure
        ('Pure',     247.0, -0.2, 30.5, 3.0, []),
        # Effect
        ('Effect',   277.2, 0.2,  34.0, 3.0, []),
        # Bottom: silence approach
        ('Bottom',   55.0,  0.0,  37.5, 3.0, [(1, 0.1)]),
        # Coda: lattice chord
        ('Chord',    0.0,   0.0,  41.0, 9.0, []),
    ]

    rng = np.random.RandomState(42)

    for name, base_freq, pan, start, node_dur, harmonics in nodes:
        i0 = int(start * SR)
        i1 = min(int((start + node_dur) * SR), n)
        seg_n = i1 - i0
        if seg_n <= 0:
            continue

        seg_t = t[i0:i1]
        seg_env = envelope(seg_n, attack=0.08, release=0.3)

        if name == 'Chord':
            # Final chord: all type frequencies together
            chord_freqs = [165.0, 220.0, 261.6, 329.6]
            chord_env = envelope(seg_n, attack=1.0, release=3.0)
            for cf in chord_freqs:
                tone = sine(cf, seg_t) * 0.08 * chord_env
                out[i0:i1, 0] += tone
                out[i0:i1, 1] += tone
            continue

        if name in ('Function', 'Pure', 'Effect'):
            # FM synthesis for function types
            if name == 'Function':
                mod_depth = 2.0
                mod_freq = base_freq * 1.5
            elif name == 'Pure':
                mod_depth = 1.5
                mod_freq = base_freq * 2.0
            else:  # Effect
                mod_depth = 4.0
                mod_freq = base_freq * 1.618  # golden ratio

            tone = fm_tone(base_freq, mod_freq, mod_depth, seg_t) * 0.14 * seg_env
            # Effect gets noise modulation
            if name == 'Effect':
                noise = rng.randn(seg_n) * 0.03 * seg_env
                tone += noise

            left_g = 0.5 * (1.0 - pan)
            right_g = 0.5 * (1.0 + pan)
            out[i0:i1, 0] += tone * left_g
            out[i0:i1, 1] += tone * right_g
            continue

        if name in ('List', 'Stream'):
            # Pulse train overlay
            if name == 'List':
                pulse_interval = 0.15  # regular
            else:
                pulse_interval = 0.0  # irregular, handled below

            for j in range(seg_n):
                tc = seg_t[j] - start
                if name == 'List':
                    if tc % pulse_interval < 0.02:
                        pulse_amp = 0.15 * np.exp(-50 * (tc % pulse_interval))
                    else:
                        pulse_amp = 0.0
                else:
                    # Poisson-like: random triggers
                    if rng.random() < 0.0005:
                        pulse_amp = 0.12
                    else:
                        pulse_amp = 0.0

                if pulse_amp > 0:
                    out[i0 + j, 0] += pulse_amp * np.sin(2 * np.pi * 2000 * seg_t[j])
                    out[i0 + j, 1] += pulse_amp * np.sin(2 * np.pi * 2000 * seg_t[j])

        # Additive synthesis from harmonic recipe
        signal = np.zeros(seg_n)
        for h_num, h_amp in harmonics:
            signal += h_amp * sine(base_freq * h_num, seg_t)

        signal *= 0.12 * seg_env
        left_g = 0.5 * (1.0 - pan)
        right_g = 0.5 * (1.0 + pan)
        out[i0:i1, 0] += signal * left_g
        out[i0:i1, 1] += signal * right_g

    # Transition clicks between nodes
    for i_node in range(1, len(nodes) - 1):
        click_t = nodes[i_node][3]
        ci = int(click_t * SR)
        if ci < n - 100:
            click = np.exp(-np.arange(100) / 10.0) * 0.15 * np.sin(2 * np.pi * 1500 * np.arange(100) / SR)
            out[ci:ci+100, 0] += click
            out[ci:ci+100, 1] += click

    # Sub drone
    drone = sine(55.0, t) * 0.03 * envelope(n, attack=1.0, release=2.0)
    out[:, 0] += drone
    out[:, 1] += drone

    env_g = envelope(n, attack=0.3, release=2.0)
    out[:, 0] *= env_g
    out[:, 1] *= env_g

    return out


# ─────────────────────────────────────────────────
# 3. Curry-Howard Correspondence
# ─────────────────────────────────────────────────
def curry_howard():
    """Propositions-as-types, proofs-as-programs.

    Three proof constructions, each building a harmonic structure:

    Section A (0-18s): Conjunction (A ∧ B)
      Both A (220Hz) and B (330Hz) must be proved.
      A's proof builds note by note (harmonics accumulate).
      B's proof builds similarly. When both complete, the conjunction
      rings as their combined spectrum — a perfect fifth chord.

    Section B (18-36s): Implication (A → B)
      A function type. Given proof of A, produce proof of B.
      A's frequency feeds through an FM transformation to produce B.
      The arrow (→) is the transformation itself — audible as modulation.
      Modus ponens: applying the function to A produces B directly.

    Section C (36-55s): Disjunction via case analysis (A ∨ B → C)
      Left channel: proof via A → C (A transforms to C=392Hz via ascending glide).
      Right channel: proof via B → C (B transforms to C via descending glide).
      Both arrive at the same C — the proof is complete regardless of which
      disjunct holds. Convergence = harmonic unification.
    """
    dur = 55.0
    n = int(dur * SR)
    out = np.zeros((n, 2))
    t = np.arange(n) / SR

    freq_A = 220.0   # A3
    freq_B = 329.63   # E4 (perfect fifth)
    freq_C = 392.0    # G4

    # ── Section A: Conjunction (A ∧ B) ──
    # Build proof of A (left-biased), then B (right-biased), then combine
    conj_start = 0.0
    conj_end = 18.0

    for i in range(int(conj_start * SR), min(int(conj_end * SR), n)):
        tc = t[i]
        phase_progress = tc / conj_end

        # Proof of A builds over 0-8s: adding harmonics one by one
        if tc < 8.0:
            num_harmonics_a = min(int(tc / 1.5) + 1, 6)
            va = 0.0
            for h in range(1, num_harmonics_a + 1):
                va += (0.7 / h) * sine(freq_A * h, np.array([tc]))[0]
            va *= 0.15 * min(1.0, tc / 0.5)
            vb = 0.0
        # Proof of B builds over 8-15s
        elif tc < 15.0:
            va = 0.0
            for h in range(1, 7):
                va += (0.7 / h) * sine(freq_A * h, np.array([tc]))[0]
            va *= 0.15

            b_progress = (tc - 8.0) / 7.0
            num_harmonics_b = min(int(b_progress * 6) + 1, 6)
            vb = 0.0
            for h in range(1, num_harmonics_b + 1):
                vb += (0.7 / h) * sine(freq_B * h, np.array([tc]))[0]
            vb *= 0.15 * min(1.0, (tc - 8.0) / 0.5)
        else:
            # Both proved: conjunction chord
            conj_fade = 1.0 - max(0, (tc - 17.0)) / 1.0
            conj_fade = max(0, conj_fade)
            va = 0.0
            for h in range(1, 7):
                va += (0.7 / h) * sine(freq_A * h, np.array([tc]))[0]
            va *= 0.18 * conj_fade
            vb = 0.0
            for h in range(1, 7):
                vb += (0.7 / h) * sine(freq_B * h, np.array([tc]))[0]
            vb *= 0.18 * conj_fade

        out[i, 0] += va * 0.7 + vb * 0.3  # A left-biased
        out[i, 1] += va * 0.3 + vb * 0.7  # B right-biased

    # ── Section B: Implication (A → B) ──
    # The arrow type as FM: A modulates to produce B
    impl_start = 18.0
    impl_end = 36.0

    for i in range(int(impl_start * SR), min(int(impl_end * SR), n)):
        tc = t[i]
        local_t = tc - impl_start
        section_dur = impl_end - impl_start

        if local_t < 6.0:
            # Present the hypothesis A (pure)
            progress = local_t / 6.0
            va = sine(freq_A, np.array([tc]))[0] * 0.18
            # Light vibrato
            va *= (1.0 + 0.1 * sine(5.0, np.array([tc]))[0])
            out[i, 0] += va * 0.6
            out[i, 1] += va * 0.4
        elif local_t < 12.0:
            # Arrow construction: A frequency modulates carrier toward B
            arrow_progress = (local_t - 6.0) / 6.0
            mod_depth = arrow_progress * 3.0
            carrier = freq_A + (freq_B - freq_A) * arrow_progress
            v = fm_tone(carrier, freq_A, mod_depth, np.array([tc]))[0] * 0.16
            out[i, 0] += v * 0.5
            out[i, 1] += v * 0.5
        elif local_t < 15.0:
            # Modus ponens: apply the arrow to A, get B
            mp_progress = (local_t - 12.0) / 3.0
            # A fades as it's consumed
            va = sine(freq_A, np.array([tc]))[0] * 0.15 * (1.0 - mp_progress)
            # B emerges
            vb = sine(freq_B, np.array([tc]))[0] * 0.18 * mp_progress
            for h in [2, 3]:
                vb += (0.3 / h) * sine(freq_B * h, np.array([tc]))[0] * 0.18 * mp_progress
            out[i, 0] += va * 0.6 + vb * 0.4
            out[i, 1] += va * 0.4 + vb * 0.6
        else:
            # B established
            fade = 1.0 - max(0, (local_t - 16.5)) / 1.5
            fade = max(0, fade)
            vb = 0.0
            for h in range(1, 5):
                vb += (0.6 / h) * sine(freq_B * h, np.array([tc]))[0]
            vb *= 0.18 * fade
            out[i, 0] += vb * 0.4
            out[i, 1] += vb * 0.6

    # ── Section C: Disjunction case analysis (A ∨ B → C) ──
    disj_start = 36.0
    disj_end = 55.0

    for i in range(int(disj_start * SR), min(int(disj_end * SR), n)):
        tc = t[i]
        local_t = tc - disj_start
        section_dur = disj_end - disj_start

        if local_t < 7.0:
            # Left case: A → C (ascending glide, left channel)
            progress = local_t / 7.0
            freq_left = freq_A + (freq_C - freq_A) * progress
            vl = sine(freq_left, np.array([tc]))[0] * 0.16
            # Add harmonics as proof builds
            num_h = min(int(progress * 4) + 1, 4)
            for h in range(2, num_h + 1):
                vl += (0.4 / h) * sine(freq_left * h, np.array([tc]))[0] * 0.16

            # Right case: B → C (descending-then-ascending, right channel)
            freq_right = freq_B + (freq_C - freq_B) * progress
            vr = sine(freq_right, np.array([tc]))[0] * 0.16
            for h in range(2, num_h + 1):
                vr += (0.4 / h) * sine(freq_right * h, np.array([tc]))[0] * 0.16

            out[i, 0] += vl
            out[i, 1] += vr
        elif local_t < 12.0:
            # Both converge to C — proof complete
            conv_progress = (local_t - 7.0) / 5.0
            # Both channels approach freq_C
            freq_l = freq_C + (freq_A - freq_C) * (1.0 - conv_progress) * 0.2
            freq_r = freq_C + (freq_B - freq_C) * (1.0 - conv_progress) * 0.2
            vl = sine(freq_l, np.array([tc]))[0] * 0.17
            vr = sine(freq_r, np.array([tc]))[0] * 0.17
            out[i, 0] += vl
            out[i, 1] += vr
        else:
            # Coda: unified C with full harmonic proof, stereo center
            fade = 1.0 - max(0, (local_t - 16.0)) / 3.0
            fade = max(0, fade)
            vc = 0.0
            for h in range(1, 7):
                vc += (0.7 / h) * sine(freq_C * h, np.array([tc]))[0]
            vc *= 0.15 * fade
            # Plus the complete type: A ∨ B → C as three-note chord
            va = sine(freq_A, np.array([tc]))[0] * 0.06 * fade
            vb = sine(freq_B, np.array([tc]))[0] * 0.06 * fade
            out[i, 0] += vc + va
            out[i, 1] += vc + vb

    # Global drone: 55Hz
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
