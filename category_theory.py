#!/usr/bin/env python3
"""Phase 18: Category Theory — functors, natural transformations, monads.

Three pieces:
1. Functor (55s, stereo) — Two categories as frequency spaces. Functor F
   maps objects and morphisms from C to D, preserving composition.
2. Natural Transformation (50s, stereo) — Two functors F,G from C to D.
   η: F→G morphs between them component by component. Naturality =
   path independence made audible.
3. Monad (55s, stereo) — The Maybe monad: unit wraps in uncertainty,
   bind chains uncertain computations, join collapses nesting.
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
    print(f"  -> {path} ({len(data)/SR:.1f}s)")


def sine(freq, t, phase=0.0):
    return np.sin(2 * np.pi * freq * t + phase)


def fm_tone(carrier, mod_freq, mod_depth, t):
    return np.sin(2 * np.pi * carrier * t + mod_depth * np.sin(2 * np.pi * mod_freq * t))


def envelope(n, attack=0.01, release=0.05):
    env = np.ones(n)
    att = int(attack * SR)
    rel = int(release * SR)
    if att > 0:
        env[:min(att, n)] = np.linspace(0, 1, min(att, n))
    if rel > 0 and rel < n:
        env[-rel:] = np.linspace(1, 0, rel)
    return env


def crossfade(n):
    return np.linspace(0, 1, n)


# Category C: pentatonic (C4, D4, E4, G4, A4)
CAT_C = np.array([261.63, 293.66, 329.63, 392.00, 440.00])
# Category D: up a fifth (G4, A4, B4, D5, E5)
CAT_D = np.array([392.00, 440.00, 493.88, 587.33, 659.26])
# Morphism order (cycle through objects)
MORPH_ORDER = [0, 1, 2, 3, 4, 3, 2, 1, 0, 4]


# -------------------------------------------------
# 1. Functor
# -------------------------------------------------
def functor():
    """Two categories, one structure-preserving map between them."""
    dur = 55.0
    n = int(dur * SR)
    out = np.zeros((n, 2))
    t = np.arange(n) / SR

    # Drone D1=55Hz throughout
    drone = 0.08 * sine(55.0, t) * envelope(n, attack=2.0, release=3.0)
    out[:, 0] += drone
    out[:, 1] += drone

    # --- Section A (0-15s): Category C melody, left-biased ---
    note_dur = 1.2
    note_n = int(note_dur * SR)
    for i, idx in enumerate(MORPH_ORDER):
        start = int(i * note_dur * SR)
        if start + note_n > n:
            break
        seg_t = np.arange(note_n) / SR
        freq = CAT_C[idx]
        # Pure sine + 2 harmonics
        sig = 0.3 * sine(freq, seg_t)
        sig += 0.15 * sine(freq * 2, seg_t)
        sig += 0.08 * sine(freq * 3, seg_t)
        sig *= envelope(note_n, attack=0.02, release=0.15)
        # Portamento glide from previous note
        if i > 0:
            prev_freq = CAT_C[MORPH_ORDER[i - 1]]
            glide_n = min(int(0.08 * SR), note_n)
            glide_f = np.linspace(prev_freq, freq, glide_n)
            glide_sig = 0.2 * np.sin(2 * np.pi * np.cumsum(glide_f) / SR)
            sig[:glide_n] += glide_sig * envelope(glide_n, attack=0.005, release=0.02)
        end = min(start + note_n, n)
        seg_len = end - start
        out[start:end, 0] += sig[:seg_len] * 0.8  # left bias
        out[start:end, 1] += sig[:seg_len] * 0.3

    # --- Section B (15-35s): Functor application ---
    sec_b_start = int(15.0 * SR)
    transform_dur = 3.5
    transform_n = int(transform_dur * SR)
    for i in range(5):
        start = sec_b_start + int(i * transform_dur * SR)
        if start + transform_n > n:
            break
        seg_t = np.arange(transform_n) / SR
        freq_c = CAT_C[i]
        freq_d = CAT_D[i]
        # FM transformation: carrier sweeps from C to D
        freq_sweep = np.linspace(freq_c, freq_d, transform_n)
        phase = 2 * np.pi * np.cumsum(freq_sweep) / SR
        mod_depth_ramp = np.linspace(0, 4.0, transform_n)
        sig = 0.25 * np.sin(phase + mod_depth_ramp * np.sin(2 * np.pi * freq_c * 0.5 * seg_t))
        # Add harmonics that morph
        h2_freq = np.linspace(freq_c * 2, freq_d * 2, transform_n)
        sig += 0.12 * np.sin(2 * np.pi * np.cumsum(h2_freq) / SR) * np.linspace(1, 0.5, transform_n)
        sig *= envelope(transform_n, attack=0.05, release=0.3)
        # Click at transformation midpoint
        mid = transform_n // 2
        click_n = int(0.005 * SR)
        if mid + click_n < transform_n:
            sig[mid:mid + click_n] += 0.3 * sine(2000, np.arange(click_n) / SR) * envelope(click_n, attack=0.001, release=0.002)
        end = min(start + transform_n, n)
        seg_len = end - start
        # Stereo: left to center
        pan = np.linspace(0.7, 0.5, seg_len)
        out[start:end, 0] += sig[:seg_len] * pan
        out[start:end, 1] += sig[:seg_len] * (1 - pan)

    # --- Section C (35-50s): Category D melody, right-biased ---
    sec_c_start = int(35.0 * SR)
    note_dur_d = 1.2
    note_n_d = int(note_dur_d * SR)
    for i, idx in enumerate(MORPH_ORDER):
        start = sec_c_start + int(i * note_dur_d * SR)
        if start + note_n_d > n:
            break
        seg_t = np.arange(note_n_d) / SR
        freq = CAT_D[idx]
        # FM synthesis (enriched, distinct from C)
        sig = 0.25 * fm_tone(freq, freq * 1.01, 1.5, seg_t)
        sig += 0.12 * fm_tone(freq * 2, freq * 0.99, 0.8, seg_t)
        sig *= envelope(note_n_d, attack=0.02, release=0.15)
        if i > 0:
            prev_freq = CAT_D[MORPH_ORDER[i - 1]]
            glide_n = min(int(0.08 * SR), note_n_d)
            glide_f = np.linspace(prev_freq, freq, glide_n)
            glide_sig = 0.15 * np.sin(2 * np.pi * np.cumsum(glide_f) / SR)
            sig[:glide_n] += glide_sig * envelope(glide_n, attack=0.005, release=0.02)
        end = min(start + note_n_d, n)
        seg_len = end - start
        out[start:end, 0] += sig[:seg_len] * 0.3  # right bias
        out[start:end, 1] += sig[:seg_len] * 0.8

    # --- Coda (50-55s): Both categories simultaneously ---
    coda_start = int(50.0 * SR)
    coda_n = n - coda_start
    if coda_n > 0:
        seg_t = np.arange(coda_n) / SR
        env_coda = envelope(coda_n, attack=0.3, release=1.5)
        for i in range(5):
            # C left
            out[coda_start:, 0] += 0.08 * sine(CAT_C[i], seg_t) * env_coda
            out[coda_start:, 0] += 0.04 * sine(CAT_C[i] * 2, seg_t) * env_coda
            # D right
            out[coda_start:, 1] += 0.08 * fm_tone(CAT_D[i], CAT_D[i] * 1.01, 1.0, seg_t) * env_coda

    out *= 0.7
    write_wav(os.path.join("output", "cat_1_functor.wav"), out)


# -------------------------------------------------
# 2. Natural Transformation
# -------------------------------------------------
def natural_transformation():
    """Two functors, a family of morphisms between them."""
    dur = 50.0
    n = int(dur * SR)
    out = np.zeros((n, 2))
    t = np.arange(n) / SR

    # Base objects in C
    base = np.array([220.0, 261.63, 293.66, 329.63, 392.00])
    # Functor F: perfect fifth (x1.5)
    F_img = base * 1.5
    # Functor G: major third (x1.25)
    G_img = base * 1.25

    # Drone
    drone = 0.07 * sine(55.0, t) * envelope(n, attack=1.5, release=2.5)
    out[:, 0] += drone
    out[:, 1] += drone

    # --- Section A (0-12s): Functor F — bright FM tones ---
    sec_a_dur = 2.0
    sec_a_n = int(sec_a_dur * SR)
    for i in range(5):
        start = int(i * sec_a_dur * SR)
        if start + sec_a_n > n:
            break
        seg_t = np.arange(sec_a_n) / SR
        freq = F_img[i]
        sig = 0.25 * fm_tone(freq, freq * 0.5, 3.0, seg_t)
        sig += 0.10 * fm_tone(freq * 2, freq, 1.5, seg_t)
        sig *= envelope(sec_a_n, attack=0.03, release=0.3)
        end = min(start + sec_a_n, n)
        seg_len = end - start
        pan = 0.3 + 0.08 * i  # spread slightly
        out[start:end, 0] += sig[:seg_len] * (1 - pan)
        out[start:end, 1] += sig[:seg_len] * pan

    # --- Section B (12-24s): Functor G — warm sine + even harmonics ---
    sec_b_base = int(12.0 * SR)
    for i in range(5):
        start = sec_b_base + int(i * sec_a_dur * SR)
        if start + sec_a_n > n:
            break
        seg_t = np.arange(sec_a_n) / SR
        freq = G_img[i]
        sig = 0.25 * sine(freq, seg_t)
        sig += 0.15 * sine(freq * 2, seg_t)  # even harmonic
        sig += 0.08 * sine(freq * 4, seg_t)  # even harmonic
        sig *= envelope(sec_a_n, attack=0.05, release=0.4)
        end = min(start + sec_a_n, n)
        seg_len = end - start
        pan = 0.3 + 0.08 * i
        out[start:end, 0] += sig[:seg_len] * (1 - pan)
        out[start:end, 1] += sig[:seg_len] * pan

    # --- Section C (24-40s): Natural transformation η ---
    # For each object: F(A) morphs to G(A), naturality square in L/R
    sec_c_base = int(24.0 * SR)
    morph_dur = 3.0
    morph_n = int(morph_dur * SR)
    for i in range(5):
        start = sec_c_base + int(i * morph_dur * SR)
        if start + morph_n > n:
            break
        seg_t = np.arange(morph_n) / SR

        f_freq = F_img[i]
        g_freq = G_img[i]

        # Frequency glide F(A) -> G(A)
        freq_glide = np.linspace(f_freq, g_freq, morph_n)
        phase = 2 * np.pi * np.cumsum(freq_glide) / SR

        # Timbre crossfade: FM (F) -> sine+even (G)
        cf = crossfade(morph_n)
        fm_part = fm_tone(f_freq, f_freq * 0.5, 3.0 * (1 - cf), seg_t)
        sine_part = sine(g_freq, seg_t) + 0.5 * sine(g_freq * 2, seg_t)
        sig = 0.2 * ((1 - cf) * fm_part + cf * sine_part * 0.7)

        # Add the gliding carrier
        sig += 0.1 * np.sin(phase)
        sig *= envelope(morph_n, attack=0.05, release=0.4)

        # Naturality square: path 1 (left) = map then morph
        # path 2 (right) = morph then map — they sound the same!
        end = min(start + morph_n, n)
        seg_len = end - start
        out[start:end, 0] += sig[:seg_len]
        out[start:end, 1] += sig[:seg_len]

        # Subtle 2kHz click at midpoint marking the η component
        mid = morph_n // 2
        click_n = int(0.004 * SR)
        if start + mid + click_n < n:
            click = 0.2 * sine(2000, np.arange(click_n) / SR) * envelope(click_n, attack=0.001, release=0.001)
            out[start + mid:start + mid + click_n, 0] += click
            out[start + mid:start + mid + click_n, 1] += click

    # --- Coda (40-50s): All η components as chord ---
    coda_start = int(40.0 * SR)
    coda_n = n - coda_start
    if coda_n > 0:
        seg_t = np.arange(coda_n) / SR
        env_c = envelope(coda_n, attack=1.0, release=3.0)
        for i in range(5):
            # Midpoint frequency between F and G images
            mid_freq = (F_img[i] + G_img[i]) / 2
            out[coda_start:, 0] += 0.06 * sine(mid_freq, seg_t) * env_c
            out[coda_start:, 1] += 0.06 * sine(mid_freq, seg_t) * env_c
            # Ghost of F and G
            out[coda_start:, 0] += 0.03 * fm_tone(F_img[i], F_img[i] * 0.5, 1.0, seg_t) * env_c
            out[coda_start:, 1] += 0.03 * sine(G_img[i], seg_t) * env_c

    out *= 0.7
    write_wav(os.path.join("output", "cat_2_natural_transformation.wav"), out)


# -------------------------------------------------
# 3. Monad
# -------------------------------------------------
def monad():
    """The Maybe monad: unit, bind, join — uncertainty as structure."""
    np.random.seed(42)
    dur = 55.0
    n = int(dur * SR)
    out = np.zeros((n, 2))
    t = np.arange(n) / SR

    # Pentatonic notes for pure values
    penta = np.array([220.0, 261.63, 329.63, 392.00, 440.00])

    # Drone
    drone = 0.07 * sine(55.0, t) * envelope(n, attack=1.5, release=3.0)
    out[:, 0] += drone
    out[:, 1] += drone

    def pure_tone(freq, dur_s):
        """Clean sine — a pure value."""
        nn = int(dur_s * SR)
        tt = np.arange(nn) / SR
        sig = 0.3 * sine(freq, tt)
        sig += 0.12 * sine(freq * 2, tt)
        return sig * envelope(nn, attack=0.02, release=0.1)

    def just_tone(freq, dur_s):
        """Sine + tremolo — a wrapped value (Just x)."""
        nn = int(dur_s * SR)
        tt = np.arange(nn) / SR
        tremolo = 0.7 + 0.3 * sine(4.0, tt)  # 4Hz AM
        sig = 0.25 * sine(freq, tt) * tremolo
        sig += 0.10 * sine(freq * 2, tt) * tremolo
        # Reverb tail: decaying copy
        tail_n = min(int(0.3 * SR), nn)
        if tail_n > 0:
            tail = 0.08 * sine(freq, tt[:tail_n]) * np.linspace(0.5, 0, tail_n)
            sig[-tail_n:] += tail
        return sig * envelope(nn, attack=0.03, release=0.2)

    def nothing_burst(dur_s):
        """Noise burst — Nothing (computation failed)."""
        nn = int(dur_s * SR)
        sig = 0.15 * np.random.randn(nn) * envelope(nn, attack=0.002, release=0.01)
        return sig

    def nested_tone(freq, dur_s):
        """Double-wrapped: tremolo on tremolo — unstable."""
        nn = int(dur_s * SR)
        tt = np.arange(nn) / SR
        trem1 = 0.6 + 0.4 * sine(4.0, tt)
        trem2 = 0.7 + 0.3 * sine(7.3, tt)  # different rate
        detune = freq * (1 + 0.008 * sine(1.5, tt))  # pitch waver
        phase = 2 * np.pi * np.cumsum(detune) / SR
        sig = 0.22 * np.sin(phase) * trem1 * trem2
        sig += 0.08 * np.sin(phase * 2.01) * trem1 * trem2
        return sig * envelope(nn, attack=0.03, release=0.2)

    # --- Section A (0-12s): Pure values ---
    note_dur = 2.0
    note_n = int(note_dur * SR)
    for i in range(5):
        start = int(i * note_dur * SR)
        if start + note_n > n:
            break
        sig = pure_tone(penta[i], note_dur)
        end = min(start + note_n, n)
        seg_len = end - start
        out[start:end, 0] += sig[:seg_len] * 0.6
        out[start:end, 1] += sig[:seg_len] * 0.6

    # --- Section B (12-25s): Unit (η) — wrapping into monad ---
    sec_b_start = int(12.0 * SR)
    wrap_dur = 2.4
    wrap_n = int(wrap_dur * SR)
    for i in range(5):
        start = sec_b_start + int(i * wrap_dur * SR)
        if start + wrap_n > n:
            break
        # First half: pure, second half: crossfade to Just
        half = wrap_n // 2
        sig = np.zeros(wrap_n)
        pure_sig = pure_tone(penta[i], wrap_dur)[:wrap_n]
        just_sig = just_tone(penta[i], wrap_dur)[:wrap_n]
        cf = np.zeros(wrap_n)
        cf[half:] = np.linspace(0, 1, wrap_n - half)
        sig = (1 - cf) * pure_sig + cf * just_sig
        end = min(start + wrap_n, n)
        seg_len = end - start
        out[start:end, 0] += sig[:seg_len] * 0.55
        out[start:end, 1] += sig[:seg_len] * 0.55

    # --- Section C (25-40s): Bind/Kleisli chain ---
    sec_c_start = int(25.0 * SR)
    chain_notes = [0, 2, 4, 3, 1, 0, 4, 2, 3, 1, 4, 0]  # indices into penta
    step_dur = 1.2
    step_n = int(step_dur * SR)
    alive = True
    chain_idx = 0
    for i, idx in enumerate(chain_notes):
        start = sec_c_start + int(i * step_dur * SR)
        if start + step_n > n or start >= int(40.0 * SR):
            break
        if alive:
            # Increasing uncertainty: more tremolo depth, more detuning
            uncertainty = min(i / len(chain_notes), 0.8)
            nn_step = min(step_n, n - start)
            tt = np.arange(nn_step) / SR
            trem_depth = 0.1 + 0.4 * uncertainty
            tremolo = (1 - trem_depth) + trem_depth * sine(4.0 + uncertainty * 3, tt)
            detune = penta[idx] * (1 + 0.003 * uncertainty * sine(1.8, tt))
            phase = 2 * np.pi * np.cumsum(detune) / SR
            sig = 0.22 * np.sin(phase) * tremolo
            sig += 0.08 * np.sin(phase * 2.01) * tremolo * (1 - 0.5 * uncertainty)
            sig *= envelope(nn_step, attack=0.02, release=0.15)

            # Pan spreads with uncertainty
            pan_l = 0.5 + 0.2 * uncertainty * (1 if i % 2 == 0 else -1)
            out[start:start + nn_step, 0] += sig * pan_l
            out[start:start + nn_step, 1] += sig * (1 - pan_l)

            # 30% chance of Nothing
            if np.random.random() < 0.3 and i > 1:
                # Nothing burst at end of note
                burst_n = min(int(0.08 * SR), nn_step)
                burst = nothing_burst(0.08)[:burst_n]
                out[start + nn_step - burst_n:start + nn_step, 0] += burst
                out[start + nn_step - burst_n:start + nn_step, 1] += burst
                alive = False
        else:
            # After Nothing: silence with very faint noise
            nn_step = min(step_n, n - start)
            faint = 0.01 * np.random.randn(nn_step) * envelope(nn_step, attack=0.1, release=0.1)
            out[start:start + nn_step, 0] += faint
            out[start:start + nn_step, 1] += faint

    # --- Section D (40-48s): Join (μ) — nested collapses to single ---
    sec_d_start = int(40.0 * SR)
    join_dur = 2.5
    join_n = int(join_dur * SR)
    join_notes = [0, 2, 4]
    for i, idx in enumerate(join_notes):
        start = sec_d_start + int(i * join_dur * SR)
        if start + join_n > n:
            break
        # First half: nested (unstable), second half: snap to Just (stable)
        half = join_n // 2
        nested_sig = nested_tone(penta[idx], join_dur)[:join_n]
        just_sig = just_tone(penta[idx], join_dur)[:join_n]
        cf = np.zeros(join_n)
        # Sharp transition at midpoint
        snap_width = int(0.05 * SR)
        snap_start = half - snap_width // 2
        snap_end = half + snap_width // 2
        cf[snap_end:] = 1.0
        cf[snap_start:snap_end] = np.linspace(0, 1, snap_end - snap_start)
        sig = (1 - cf) * nested_sig + cf * just_sig
        # Click at snap point
        click_n = int(0.003 * SR)
        if half + click_n < join_n:
            sig[half:half + click_n] += 0.25 * sine(2500, np.arange(click_n) / SR) * envelope(click_n, attack=0.001, release=0.001)
        end = min(start + join_n, n)
        seg_len = end - start
        out[start:end, 0] += sig[:seg_len] * 0.55
        out[start:end, 1] += sig[:seg_len] * 0.55

    # --- Coda (48-55s): Resolved A major chord ---
    coda_start = int(48.0 * SR)
    coda_n = n - coda_start
    if coda_n > 0:
        seg_t = np.arange(coda_n) / SR
        env_c = envelope(coda_n, attack=1.0, release=2.5)
        # A major: A3=220, C#4=277.18, E4=329.63
        chord = [220.0, 277.18, 329.63, 440.0]
        for freq in chord:
            out[coda_start:, 0] += 0.08 * sine(freq, seg_t) * env_c
            out[coda_start:, 1] += 0.08 * sine(freq, seg_t) * env_c
            out[coda_start:, 0] += 0.03 * sine(freq * 2, seg_t) * env_c
            out[coda_start:, 1] += 0.03 * sine(freq * 2, seg_t) * env_c

    out *= 0.7
    write_wav(os.path.join("output", "cat_3_monad.wav"), out)


def main():
    os.makedirs("output", exist_ok=True)
    print("Phase 18: Category Theory")
    print("=" * 40)
    print("\n1. Functor")
    functor()
    print("\n2. Natural Transformation")
    natural_transformation()
    print("\n3. Monad")
    monad()
    print("\nDone.")


if __name__ == "__main__":
    main()
