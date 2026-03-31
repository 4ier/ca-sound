#!/usr/bin/env python3
"""Phase 23: Topology -- Euler characteristic, knot invariants, Mobius strip.

Three pieces:
1. Euler Characteristic (55s, stereo) -- Polyhedra morph through a sequence of
   shapes (tetrahedron -> cube -> octahedron -> dodecahedron -> icosahedron),
   all sharing V-E+F=2. Vertices = bright FM pings, Edges = sustained mid tones,
   Faces = low warm chords. The invariant chi=2 manifests as a persistent
   perfect-fifth drone that never changes despite the morphing geometry.

2. Knot Invariants (50s, stereo) -- A trefoil knot's crossing number (3) drives
   a 3-beat rhythmic ostinato. Reidemeister moves (twist/poke/slide) morph the
   knot through isotopy-equivalent forms, each move creating a characteristic
   sound gesture. The unknot resolves to a pure sine -- zero crossings, zero
   complexity.

3. Mobius Strip (55s, stereo) -- A traversal around a Mobius strip. The walker
   starts at a point and returns to the same point but on the opposite side
   (orientation reversal). Left and right channels slowly swap roles over one
   full traversal, then swap back on the second lap. The twist point creates
   a brief moment of mono collapse.
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
    with wave.open(path, "w") as w:
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


def click(n=200, freq=2000):
    t_arr = np.arange(n) / SR
    return np.sin(2 * np.pi * freq * t_arr) * np.exp(-t_arr * 40)


# ── Piece 1: Euler Characteristic ───────────────────────────────────────────

def euler_characteristic():
    """Platonic solids morph: V-E+F=2 is the invariant drone."""
    print("Generating Euler Characteristic...")
    duration = 55.0
    N = int(duration * SR)
    t = np.arange(N) / SR
    out_L = np.zeros(N)
    out_R = np.zeros(N)

    # Platonic solids: (name, V, E, F)
    # All have chi = V - E + F = 2
    solids = [
        ("tetrahedron", 4, 6, 4),
        ("cube", 8, 12, 6),
        ("octahedron", 6, 12, 8),
        ("dodecahedron", 20, 30, 12),
        ("icosahedron", 12, 30, 20),
    ]

    # chi=2 invariant drone: perfect fifth (A1 + E2) = the topological constant
    inv_freq1 = 55.0   # A1
    inv_freq2 = 82.5    # E2 (perfect fifth)
    inv_drone = (sine(inv_freq1, t) * 0.12 + sine(inv_freq2, t) * 0.08)
    inv_drone *= envelope(N, attack=3.0, release=4.0)
    out_L += inv_drone
    out_R += inv_drone

    # Each solid gets ~10s
    solid_dur = 9.0
    gap = 1.0
    rng = np.random.RandomState(42)

    for si, (name, V, E, F) in enumerate(solids):
        t_start = si * (solid_dur + gap) + 1.0
        s_start = int(t_start * SR)

        # ── Vertices: bright FM pings, count = V ──
        # Spread V pings over first 3s of the solid's section
        v_base_freq = 440.0 + si * 60  # shift up per solid
        for vi in range(V):
            ping_t = t_start + (vi / max(V - 1, 1)) * 3.0
            s_ping = int(ping_t * SR)
            ping_dur = int(0.15 * SR)
            if s_ping + ping_dur >= N:
                break
            pt = np.arange(ping_dur) / SR
            # FM ping: bright attack
            ping = fm_tone(v_base_freq + rng.uniform(-20, 20),
                          v_base_freq * 2, 3.0, pt)
            ping *= np.exp(-pt * 15) * 0.18
            pan = rng.uniform(0.2, 0.8)
            out_L[s_ping:s_ping + ping_dur] += ping * (1 - pan)
            out_R[s_ping:s_ping + ping_dur] += ping * pan

        # ── Edges: sustained mid-frequency tones, count = E ──
        # Map edges to a cluster of tones over 4s
        e_base_freq = 220.0 + si * 30
        e_start_t = t_start + 2.0
        e_dur_samples = int(5.0 * SR)
        e_start_s = int(e_start_t * SR)
        e_end_s = min(e_start_s + e_dur_samples, N)
        e_n = e_end_s - e_start_s

        if e_n > 0:
            # Create E overlapping tones (limit to 12 for sanity, use density for more)
            n_tones = min(E, 12)
            for ei in range(n_tones):
                freq = e_base_freq * (1 + ei * 0.05)
                tone_start = e_start_s + int(ei / n_tones * e_n * 0.3)
                tone_len = min(e_n - int(ei / n_tones * e_n * 0.3), e_n)
                if tone_start + tone_len > N:
                    tone_len = N - tone_start
                if tone_len <= 0:
                    break
                tt = np.arange(tone_len) / SR
                tone = sine(freq, tt) * envelope(tone_len, attack=0.3, release=0.5)
                # Amplitude scales with total edge count
                amp = 0.06 * (E / 30.0)
                pan = ei / max(n_tones - 1, 1)
                out_L[tone_start:tone_start + tone_len] += tone * amp * (1 - pan)
                out_R[tone_start:tone_start + tone_len] += tone * amp * pan

        # ── Faces: low warm chords, count = F ──
        f_base_freq = 110.0 + si * 15
        f_start_t = t_start + 4.0
        f_dur_samples = int(5.0 * SR)
        f_start_s = int(f_start_t * SR)
        f_end_s = min(f_start_s + f_dur_samples, N)
        f_n = f_end_s - f_start_s

        if f_n > 0:
            n_face_tones = min(F, 8)
            for fi in range(n_face_tones):
                # Harmonic series based on face count
                freq = f_base_freq * (1 + fi * 0.08)
                ft = np.arange(f_n) / SR
                # Warm tone: fundamental + soft 2nd harmonic
                face = (sine(freq, ft) + 0.3 * sine(freq * 2, ft)) * 0.04
                face *= envelope(f_n, attack=0.5, release=1.0)
                pan = 0.3 + fi / max(n_face_tones - 1, 1) * 0.4
                out_L[f_start_s:f_end_s] += face * (1 - pan)
                out_R[f_start_s:f_end_s] += face * pan

        # ── Transition click between solids ──
        if si < len(solids) - 1:
            trans_s = int((t_start + solid_dur) * SR)
            c = click(400, 1800)
            if trans_s + len(c) < N:
                out_L[trans_s:trans_s + len(c)] += c * 0.12
                out_R[trans_s:trans_s + len(c)] += c * 0.12

    # Global envelope
    env = envelope(N, attack=1.5, release=4.0)
    out_L *= env
    out_R *= env

    out = np.stack([out_L, out_R], axis=1)
    peak = np.max(np.abs(out))
    if peak > 0:
        out *= 0.85 / peak
    return out


# ── Piece 2: Knot Invariants ────────────────────────────────────────────────

def knot_invariants():
    """Trefoil knot crossing number drives rhythm; Reidemeister moves morph topology."""
    print("Generating Knot Invariants...")
    duration = 50.0
    N = int(duration * SR)
    t = np.arange(N) / SR
    out_L = np.zeros(N)
    out_R = np.zeros(N)

    rng = np.random.RandomState(73)

    # ── Section A (0-18s): Trefoil knot ──
    # Crossing number = 3 → 3-beat rhythmic ostinato
    crossing_num = 3
    bpm = 90.0
    beat_s = int(60.0 / bpm * SR)

    # Base frequencies for crossings: each crossing = a voice
    cross_freqs = [220.0, 277.18, 329.63]  # A3, C#4, E4 (A major triad)

    # Over/under at each crossing: different timbres
    # Over = bright FM, Under = muted sine
    for beat_idx in range(int(18.0 * bpm / 60)):
        beat_time = beat_idx * 60.0 / bpm
        cross_idx = beat_idx % crossing_num
        s = int(beat_time * SR)
        note_len = int(0.3 * SR)

        if s + note_len >= N:
            break

        freq = cross_freqs[cross_idx]
        nt = np.arange(note_len) / SR

        # Alternate over/under each cycle
        is_over = (beat_idx // crossing_num) % 2 == 0
        if is_over:
            # Over-crossing: bright FM tone
            note = fm_tone(freq, freq * 1.5, 2.5, nt) * 0.25
        else:
            # Under-crossing: muted pure sine
            note = sine(freq, nt) * 0.15

        note *= envelope(note_len, attack=0.005, release=0.15)

        # Pan rotates with crossings (3 positions)
        pan = [0.25, 0.5, 0.75][cross_idx]
        out_L[s:s + note_len] += note * (1 - pan)
        out_R[s:s + note_len] += note * pan

    # ── Section B (18-35s): Reidemeister moves ──
    # Three types of moves that preserve knot type
    # R1 (twist): frequency glissando up then down
    # R2 (poke): two notes added then removed (pair creation/annihilation)
    # R3 (slide): three notes shift positions simultaneously

    moves = [
        (18.0, "R1"), (20.5, "R2"), (23.0, "R1"),
        (25.5, "R3"), (28.0, "R2"), (30.5, "R3"),
        (33.0, "R1"),
    ]

    for move_t, move_type in moves:
        ms = int(move_t * SR)
        move_dur = int(2.0 * SR)
        if ms + move_dur >= N:
            break
        mt = np.arange(move_dur) / SR

        if move_type == "R1":
            # Twist: frequency sweeps up then back
            freq_sweep = 220 + 200 * np.sin(np.pi * mt / (move_dur / SR))
            sig = np.sin(2 * np.pi * np.cumsum(freq_sweep) / SR) * 0.2
            sig *= envelope(move_dur, attack=0.05, release=0.3)
            out_L[ms:ms + move_dur] += sig * 0.6
            out_R[ms:ms + move_dur] += sig * 0.4

        elif move_type == "R2":
            # Poke: pair of notes appear then vanish
            half = move_dur // 2
            ht = np.arange(half) / SR
            n1 = sine(330, ht) * envelope(half, attack=0.02, release=0.1) * 0.2
            n2 = sine(440, ht) * envelope(half, attack=0.02, release=0.1) * 0.2
            # First half: appear (crescendo)
            n1[:half] *= np.linspace(0, 1, half)
            n2[:half] *= np.linspace(0, 1, half)
            out_L[ms:ms + half] += n1
            out_R[ms:ms + half] += n2
            # Second half: annihilate (reverse + cross)
            out_L[ms + half:ms + move_dur] += n2[:move_dur - half] * np.linspace(1, 0, move_dur - half)
            out_R[ms + half:ms + move_dur] += n1[:move_dur - half] * np.linspace(1, 0, move_dur - half)

        elif move_type == "R3":
            # Slide: three voices shift simultaneously
            third = move_dur // 3
            for vi, (freq, pan) in enumerate([(220, 0.2), (330, 0.5), (440, 0.8)]):
                # Each voice slides to the next position
                next_pan = [0.5, 0.8, 0.2][vi]
                vt = np.arange(move_dur) / SR
                voice = sine(freq, vt) * 0.12 * envelope(move_dur, attack=0.1, release=0.3)
                p = np.linspace(pan, next_pan, move_dur)
                out_L[ms:ms + move_dur] += voice * (1 - p)
                out_R[ms:ms + move_dur] += voice * p

        # Click at each Reidemeister move
        c = click(300, 2000)
        if ms + len(c) < N:
            out_L[ms:ms + len(c)] += c * 0.1
            out_R[ms:ms + len(c)] += c * 0.1

    # ── Section C (35-50s): Unknotting ──
    # The trefoil cannot be unknotted, but we can show the unknot resolving
    # Crossings fade one by one → pure sine (zero crossings)
    unknot_start = 35.0
    unknot_dur = 12.0

    # Fading trefoil rhythm (crossing voices disappear)
    for ci in range(3):
        fade_start = unknot_start + ci * 3.0
        fade_end = fade_start + 4.0
        fs = int(fade_start * SR)
        fe = min(int(fade_end * SR), N)
        seg_len = fe - fs
        if seg_len <= 0:
            break

        freq = cross_freqs[ci]
        st = np.arange(seg_len) / SR
        fading = sine(freq, st) * np.linspace(0.2, 0, seg_len) * 0.3
        fading *= envelope(seg_len, attack=0.1, release=1.0)
        out_L[fs:fe] += fading * 0.5
        out_R[fs:fe] += fading * 0.5

    # Pure sine emerges: the unknot = zero crossings = pure simplicity
    unknot_emerge = int(42.0 * SR)
    unknot_end_s = min(int(50.0 * SR), N)
    unknot_len = unknot_end_s - unknot_emerge
    if unknot_len > 0:
        ut = np.arange(unknot_len) / SR
        pure = sine(220.0, ut) * 0.35
        pure *= envelope(unknot_len, attack=2.0, release=3.0)
        out_L[unknot_emerge:unknot_end_s] += pure * 0.5
        out_R[unknot_emerge:unknot_end_s] += pure * 0.5

    # Drone: D2 = 73.4 Hz
    drone = sine(73.4, t) * 0.07 * envelope(N, attack=2.0, release=3.0)
    out_L += drone
    out_R += drone

    # Global envelope
    env = envelope(N, attack=1.0, release=3.0)
    out_L *= env
    out_R *= env

    out = np.stack([out_L, out_R], axis=1)
    peak = np.max(np.abs(out))
    if peak > 0:
        out *= 0.85 / peak
    return out


# ── Piece 3: Möbius Strip ───────────────────────────────────────────────────

def mobius_strip():
    """Traversal of a Mobius strip: orientation reversal as stereo swap."""
    print("Generating Mobius Strip...")
    duration = 55.0
    N = int(duration * SR)
    t = np.arange(N) / SR
    out_L = np.zeros(N)
    out_R = np.zeros(N)

    rng = np.random.RandomState(55)

    # The Mobius strip has one surface and one edge.
    # A walker traversing it returns to start after one lap but on the "other side"
    # (orientation reversed). After two laps, they're back to original orientation.

    # Musical mapping:
    # - Position along strip → frequency (pentatonic melody)
    # - Orientation → stereo balance (L/R swap over one lap)
    # - The twist → brief mono collapse + dissonance

    # Two full laps: 0-25s (lap 1) + 25-50s (lap 2) + 50-55s (coda)
    lap_dur = 24.0
    coda_start = 49.0

    # Pentatonic scale in D: D3 E3 F#3 A3 B3 D4 E4 F#4 A4 B4
    penta = [146.83, 164.81, 185.00, 220.00, 246.94,
             293.66, 329.63, 369.99, 440.00, 493.88]

    # ── Walking melody: position along strip drives note selection ──
    step_dur = 0.6  # seconds per step
    n_steps_per_lap = int(lap_dur / step_dur)

    for lap in range(2):
        lap_start = lap * lap_dur + 0.5

        for step in range(n_steps_per_lap):
            step_time = lap_start + step * step_dur
            s = int(step_time * SR)
            note_len = int(0.45 * SR)
            if s + note_len >= N:
                break

            # Position along strip: 0 to 1
            pos = step / n_steps_per_lap
            # Note selection: smoothly traverse the pentatonic scale
            note_idx = int(pos * (len(penta) - 1))
            freq = penta[note_idx]

            nt = np.arange(note_len) / SR

            # Tone: sine + gentle 2nd harmonic, slight vibrato
            vib = 0.003 * freq * np.sin(2 * np.pi * 5.5 * nt)
            note = sine(freq + vib, nt) * 0.25
            note += sine(freq * 2 + vib * 2, nt) * 0.08
            note *= envelope(note_len, attack=0.02, release=0.2)

            # ── Stereo: orientation reversal ──
            # Lap 1: starts L-dominant (pan=0.2), ends R-dominant (pan=0.8)
            # Lap 2: starts R-dominant (pan=0.8), ends L-dominant (pan=0.2)
            if lap == 0:
                pan = 0.2 + pos * 0.6
            else:
                pan = 0.8 - pos * 0.6

            out_L[s:s + note_len] += note * (1 - pan)
            out_R[s:s + note_len] += note * pan

            # ── The twist point: halfway through each lap ──
            # At pos ≈ 0.5, orientation flips. Mark with dissonance + mono collapse
            if abs(pos - 0.5) < 0.03:
                # Brief tritone (most dissonant interval) + both channels equal
                twist_len = int(0.3 * SR)
                if s + twist_len < N:
                    tt = np.arange(twist_len) / SR
                    tritone = (sine(freq, tt) + sine(freq * np.sqrt(2), tt)) * 0.15
                    tritone *= envelope(twist_len, attack=0.01, release=0.15)
                    # Mono collapse: equal in both channels
                    out_L[s:s + twist_len] += tritone * 0.5
                    out_R[s:s + twist_len] += tritone * 0.5

    # ── Edge tone: the Mobius strip has ONE edge ──
    # A continuous tone that never breaks, representing the single edge
    edge_freq = 110.0  # A2
    edge = sine(edge_freq, t) * 0.08
    edge += sine(edge_freq * 3, t) * 0.025  # subtle 3rd harmonic
    edge *= envelope(N, attack=3.0, release=4.0)
    out_L += edge * 0.6
    out_R += edge * 0.4

    # ── Coda (49-55s): Orientation resolved ──
    # After two laps, back to original orientation: symmetric resolution
    coda_s = int(coda_start * SR)
    coda_len = N - coda_s
    if coda_len > 0:
        ct = np.arange(coda_len) / SR
        # D major chord resolving symmetrically (equal L/R)
        chord = (sine(146.83, ct) * 0.15 +   # D3
                 sine(185.00, ct) * 0.12 +     # F#3
                 sine(220.00, ct) * 0.12 +     # A3
                 sine(293.66, ct) * 0.08)      # D4
        chord *= envelope(coda_len, attack=1.5, release=3.0)
        out_L[coda_s:] += chord * 0.5
        out_R[coda_s:] += chord * 0.5

    # Drone: D1 = 36.7 Hz (very low, felt more than heard)
    drone = sine(36.7, t) * 0.06 * envelope(N, attack=3.0, release=4.0)
    out_L += drone
    out_R += drone

    # Global envelope
    env = envelope(N, attack=1.5, release=4.0)
    out_L *= env
    out_R *= env

    out = np.stack([out_L, out_R], axis=1)
    peak = np.max(np.abs(out))
    if peak > 0:
        out *= 0.85 / peak
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    pieces = [
        ("topo_1_euler_characteristic", euler_characteristic),
        ("topo_2_knot_invariants", knot_invariants),
        ("topo_3_mobius_strip", mobius_strip),
    ]
    for name, fn in pieces:
        data = fn()
        write_wav(f"output/{name}.wav", data)
    print("\nDone! All topology pieces generated.")
