#!/usr/bin/env python3
"""Phase 24: Database Internals -- B-Tree traversal, transaction isolation, MVCC.

Three pieces:
1. B-Tree Search (55s, stereo) -- A B-Tree of order 5 with 4 levels. A search
   key descends through internal nodes (each node comparison = FM click at that
   node's frequency), finally reaching a leaf (warm sine resolution). Root = high
   pitch, leaves = low pitch. Multiple searches overlap as polyphonic melodies.
   55Hz drone represents the persistent storage layer.

2. Transaction Isolation (50s, stereo) -- Three concurrent transactions:
   T1 reads (pure sines, left), T2 writes (FM tones, right), T3 does both (center).
   Read = gentle sine at row's frequency. Write = FM burst + commit chord or
   rollback dissonance. Dirty read = tritone clash. Serializable = clean separation.

3. MVCC (55s, stereo) -- Multi-Version Concurrency Control. Each row has versions
   stacking as overtone layers. Old versions = dim lower harmonics, current = bright
   fundamental. Snapshot reads see frozen harmonic state. Garbage collection prunes
   old overtones one by one. Vacuum = silence then fresh fundamental.
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

# ── Piece 1: B-Tree Search ──────────────────────────────────────────────────

def btree_search():
    """B-Tree traversal: root-to-leaf descent as pitch descent."""
    print("Generating B-Tree Search...")
    duration = 55.0
    N = int(duration * SR)
    t = np.arange(N) / SR
    out_L = np.zeros(N)
    out_R = np.zeros(N)
    rng = np.random.default_rng(42)

    # B-Tree structure: 4 levels, order 5
    # Level 0 (root): 1 node, freq ~880Hz
    # Level 1: 5 nodes, freq ~440Hz
    # Level 2: 25 nodes, freq ~220Hz
    # Level 3 (leaves): 125 nodes, freq ~110Hz
    level_freqs = [880.0, 440.0, 220.0, 110.0]
    level_harmonics = [6, 4, 3, 2]  # root = bright, leaf = warm

    # Persistent storage drone
    drone_freq = 55.0
    drone = sine(drone_freq, t) * 0.06 * envelope(N, attack=2.0, release=2.0)
    out_L += drone
    out_R += drone

    # Generate 18 search queries, each descends 4 levels
    n_searches = 18
    search_starts = np.linspace(1.0, 48.0, n_searches)
    level_duration = 0.8  # time per level comparison

    for i, start in enumerate(search_starts):
        # Each search picks a path through the tree
        pan = 0.3 + 0.4 * rng.random()  # stereo position
        key_offset = rng.integers(0, 5)

        for level in range(4):
            t0 = start + level * level_duration
            if t0 >= duration - 1.0:
                break

            s0 = int(t0 * SR)
            freq = level_freqs[level] * (1.0 + 0.05 * key_offset)
            n_harm = level_harmonics[level]

            # Comparison sound: FM click + harmonics
            note_len = int(0.5 * SR)
            if s0 + note_len > N:
                note_len = N - s0
            if note_len <= 0:
                break

            t_note = np.arange(note_len) / SR
            tone = np.zeros(note_len)

            # Harmonics for this level
            for h in range(1, n_harm + 1):
                amp = 0.3 / h
                tone += amp * sine(freq * h, t_note)

            # FM comparison click at start
            click_len = min(int(0.05 * SR), note_len)
            t_click = np.arange(click_len) / SR
            comparison = fm_tone(freq * 2, freq * 4, 3.0, t_click) * np.exp(-t_click * 30)
            tone[:click_len] += comparison * 0.4

            # Envelope
            env = envelope(note_len, attack=0.01, release=0.2)
            tone *= env * 0.15

            # Leaf = warmer, longer sustain
            if level == 3:
                leaf_env = envelope(note_len, attack=0.05, release=0.3)
                leaf_tone = sine(freq, t_note) * leaf_env * 0.12
                # Add perfect fifth for resolution
                leaf_tone += sine(freq * 1.5, t_note) * leaf_env * 0.06
                tone += leaf_tone

            end = min(s0 + note_len, N)
            actual_len = end - s0
            out_L[s0:end] += tone[:actual_len] * (1 - pan)
            out_R[s0:end] += tone[:actual_len] * pan

    # Final chord: all level frequencies
    coda_start = int(50.0 * SR)
    coda_len = int(4.5 * SR)
    if coda_start + coda_len <= N:
        t_coda = np.arange(coda_len) / SR
        coda_env = envelope(coda_len, attack=0.5, release=2.0)
        coda = np.zeros(coda_len)
        for freq in level_freqs:
            coda += sine(freq, t_coda) * 0.08
            coda += sine(freq * 1.5, t_coda) * 0.04  # fifths
        coda *= coda_env
        out_L[coda_start:coda_start + coda_len] += coda * 0.7
        out_R[coda_start:coda_start + coda_len] += coda * 0.7

    mix = np.stack([out_L, out_R], axis=1)
    mix /= max(np.max(np.abs(mix)), 1e-6) / 0.85
    return mix

# ── Piece 2: Transaction Isolation ───────────────────────────────────────────

def transaction_isolation():
    """Concurrent transactions: reads, writes, conflicts, commits."""
    print("Generating Transaction Isolation...")
    duration = 50.0
    N = int(duration * SR)
    t = np.arange(N) / SR
    out_L = np.zeros(N)
    out_R = np.zeros(N)
    rng = np.random.default_rng(77)

    # 55Hz base drone = database engine
    drone = sine(55.0, t) * 0.05 * envelope(N, attack=1.5, release=1.5)
    out_L += drone
    out_R += drone

    # Row frequencies (pentatonic: D3, E3, G3, A3, B3, D4, E4, G4)
    row_freqs = [146.83, 164.81, 196.0, 220.0, 246.94, 293.66, 329.63, 392.0]

    # Transaction events: (time, type, row_idx, tx_id)
    # tx_id: 0=T1(reader,left), 1=T2(writer,right), 2=T3(read-write,center)
    events = []

    # T1: sequential reads across rows (left channel)
    for i in range(12):
        t_ev = 2.0 + i * 3.2
        row = rng.integers(0, len(row_freqs))
        events.append((t_ev, 'read', row, 0))

    # T2: writes with commits (right channel)
    for i in range(8):
        t_ev = 3.5 + i * 4.5
        row = rng.integers(0, len(row_freqs))
        events.append((t_ev, 'write', row, 1))
        # Commit 0.8s later
        events.append((t_ev + 0.8, 'commit', row, 1))

    # T3: mixed read-write (center)
    for i in range(6):
        t_ev = 5.0 + i * 6.0
        row = rng.integers(0, len(row_freqs))
        events.append((t_ev, 'read', row, 2))
        events.append((t_ev + 1.0, 'write', row, 2))
        if rng.random() > 0.3:
            events.append((t_ev + 1.8, 'commit', row, 2))
        else:
            events.append((t_ev + 1.8, 'rollback', row, 2))

    # Dirty reads: T1 reads while T2 is mid-write (conflict)
    events.append((12.0, 'dirty_read', 3, 0))
    events.append((25.0, 'dirty_read', 5, 0))

    events.sort(key=lambda x: x[0])

    tx_pans = [0.2, 0.8, 0.5]  # L, R, center

    for ev_time, ev_type, row, tx_id in events:
        if ev_time >= duration - 1.5:
            break
        s0 = int(ev_time * SR)
        freq = row_freqs[row]
        pan = tx_pans[tx_id]

        if ev_type == 'read':
            # Gentle sine + 2nd harmonic
            note_len = int(0.6 * SR)
            if s0 + note_len > N:
                note_len = N - s0
            t_n = np.arange(note_len) / SR
            tone = sine(freq, t_n) * 0.15 + sine(freq * 2, t_n) * 0.05
            tone *= envelope(note_len, attack=0.03, release=0.2)
            end = min(s0 + note_len, N)
            out_L[s0:end] += tone[:end - s0] * (1 - pan)
            out_R[s0:end] += tone[:end - s0] * pan

        elif ev_type == 'write':
            # FM burst = write operation
            note_len = int(0.4 * SR)
            if s0 + note_len > N:
                note_len = N - s0
            t_n = np.arange(note_len) / SR
            tone = fm_tone(freq, freq * 1.5, 2.5, t_n) * 0.18
            tone *= envelope(note_len, attack=0.005, release=0.15)
            end = min(s0 + note_len, N)
            out_L[s0:end] += tone[:end - s0] * (1 - pan)
            out_R[s0:end] += tone[:end - s0] * pan

        elif ev_type == 'commit':
            # Perfect fifth chord = successful commit
            note_len = int(0.7 * SR)
            if s0 + note_len > N:
                note_len = N - s0
            t_n = np.arange(note_len) / SR
            tone = (sine(freq, t_n) * 0.1 +
                    sine(freq * 1.5, t_n) * 0.07 +
                    sine(freq * 2, t_n) * 0.04)
            tone *= envelope(note_len, attack=0.02, release=0.3)
            # Commit click
            cl = click(300, 1800)
            cl_len = min(len(cl), note_len)
            tone[:cl_len] += cl[:cl_len] * 0.15
            end = min(s0 + note_len, N)
            out_L[s0:end] += tone[:end - s0] * (1 - pan)
            out_R[s0:end] += tone[:end - s0] * pan

        elif ev_type == 'rollback':
            # Descending FM sweep = rollback
            note_len = int(0.5 * SR)
            if s0 + note_len > N:
                note_len = N - s0
            t_n = np.arange(note_len) / SR
            sweep_freq = freq * (1.0 - 0.5 * t_n / (note_len / SR))
            tone = np.sin(2 * np.pi * sweep_freq * t_n) * 0.15
            tone *= envelope(note_len, attack=0.005, release=0.15)
            end = min(s0 + note_len, N)
            out_L[s0:end] += tone[:end - s0] * (1 - pan)
            out_R[s0:end] += tone[:end - s0] * pan

        elif ev_type == 'dirty_read':
            # Tritone clash = dirty read conflict
            note_len = int(0.8 * SR)
            if s0 + note_len > N:
                note_len = N - s0
            t_n = np.arange(note_len) / SR
            tritone = freq * np.sqrt(2)
            tone = (sine(freq, t_n) * 0.12 +
                    sine(tritone, t_n) * 0.12 +
                    fm_tone(freq, freq * 0.97, 1.5, t_n) * 0.08)
            tone *= envelope(note_len, attack=0.005, release=0.3)
            end = min(s0 + note_len, N)
            # Dirty reads spread across both channels
            out_L[s0:end] += tone[:end - s0] * 0.6
            out_R[s0:end] += tone[:end - s0] * 0.6

    # Coda: serializable isolation = clean A major chord
    coda_start = int(45.0 * SR)
    coda_len = int(4.5 * SR)
    if coda_start + coda_len <= N:
        t_c = np.arange(coda_len) / SR
        coda_env = envelope(coda_len, attack=0.5, release=2.0)
        # A major: A3(220) C#4(277.18) E4(329.63)
        coda = (sine(220, t_c) * 0.1 + sine(277.18, t_c) * 0.08 +
                sine(329.63, t_c) * 0.07 + sine(440, t_c) * 0.05)
        coda *= coda_env
        out_L[coda_start:coda_start + coda_len] += coda
        out_R[coda_start:coda_start + coda_len] += coda

    mix = np.stack([out_L, out_R], axis=1)
    mix /= max(np.max(np.abs(mix)), 1e-6) / 0.85
    return mix

# ── Piece 3: MVCC ───────────────────────────────────────────────────────────

def mvcc():
    """Multi-Version Concurrency Control: version layers as overtone stacking."""
    print("Generating MVCC...")
    duration = 55.0
    N = int(duration * SR)
    t = np.arange(N) / SR
    out_L = np.zeros(N)
    out_R = np.zeros(N)
    rng = np.random.default_rng(99)

    # C1 drone = storage engine
    drone = sine(32.7, t) * 0.05 * envelope(N, attack=2.0, release=2.0)
    out_L += drone
    out_R += drone

    # 6 rows, each with a base frequency (D minor pentatonic)
    row_base = [146.83, 174.61, 196.0, 220.0, 261.63, 293.66]
    n_rows = len(row_base)

    # Version history per row: list of (create_time, expire_time)
    # Newer versions add higher harmonics
    max_versions = 5

    # Phase 1 (0-18s): Versions accumulate — writes create new versions
    # Each write adds a harmonic layer
    write_events = []
    for i in range(20):
        t_ev = 1.0 + i * 0.85
        row = rng.integers(0, n_rows)
        write_events.append((t_ev, row))

    for t_ev, row in write_events:
        if t_ev >= duration - 2.0:
            break
        s0 = int(t_ev * SR)
        freq = row_base[row]

        # Write = FM burst creating new version
        note_len = int(0.4 * SR)
        if s0 + note_len > N:
            continue
        t_n = np.arange(note_len) / SR
        tone = fm_tone(freq, freq * 1.5, 2.0, t_n) * 0.12
        tone *= envelope(note_len, attack=0.005, release=0.15)
        pan = 0.2 + 0.6 * (row / n_rows)
        out_L[s0:s0 + note_len] += tone * (1 - pan)
        out_R[s0:s0 + note_len] += tone * pan

        # Version layer: sustained harmonic that persists
        version_harm = 1 + rng.integers(1, max_versions)
        layer_len = int(3.0 * SR)
        if s0 + layer_len > N:
            layer_len = N - s0
        t_l = np.arange(layer_len) / SR
        layer = sine(freq * version_harm, t_l) * (0.04 / version_harm)
        layer *= envelope(layer_len, attack=0.1, release=1.5)
        out_L[s0:s0 + layer_len] += layer * (1 - pan)
        out_R[s0:s0 + layer_len] += layer * pan

    # Phase 2 (18-35s): Snapshot reads — frozen harmonic state
    snapshot_events = []
    for i in range(10):
        t_ev = 18.0 + i * 1.7
        row = rng.integers(0, n_rows)
        snapshot_events.append((t_ev, row))

    for t_ev, row in snapshot_events:
        if t_ev >= duration - 2.0:
            break
        s0 = int(t_ev * SR)
        freq = row_base[row]

        # Snapshot read = pure sine + frozen harmonics (2nd and 3rd)
        note_len = int(1.0 * SR)
        if s0 + note_len > N:
            continue
        t_n = np.arange(note_len) / SR
        tone = sine(freq, t_n) * 0.12
        tone += sine(freq * 2, t_n) * 0.04  # visible version
        tone += sine(freq * 3, t_n) * 0.02  # older version (dimmer)
        tone *= envelope(note_len, attack=0.03, release=0.4)

        # Snapshot click
        cl = click(200, 1500)
        cl_len = min(len(cl), note_len)
        tone[:cl_len] += cl[:cl_len] * 0.1

        pan = 0.2 + 0.6 * (row / n_rows)
        out_L[s0:s0 + note_len] += tone * (1 - pan)
        out_R[s0:s0 + note_len] += tone * pan

    # Phase 3 (35-50s): Garbage collection — old versions pruned
    gc_events = []
    for i in range(12):
        t_ev = 35.0 + i * 1.2
        row = rng.integers(0, n_rows)
        harm = rng.integers(3, max_versions + 1)  # prune high harmonics first
        gc_events.append((t_ev, row, harm))

    for t_ev, row, harm in gc_events:
        if t_ev >= duration - 2.0:
            break
        s0 = int(t_ev * SR)
        freq = row_base[row]

        # GC = harmonic fadeout (the old version disappearing)
        gc_len = int(0.8 * SR)
        if s0 + gc_len > N:
            continue
        t_g = np.arange(gc_len) / SR
        # Dying harmonic
        dying = sine(freq * harm, t_g) * (0.06 / harm)
        dying *= np.exp(-t_g * 4)  # rapid decay
        # Small click marking the prune
        cl = click(150, 2200)
        cl_len = min(len(cl), gc_len)
        dying[:cl_len] += cl[:cl_len] * 0.08

        pan = 0.2 + 0.6 * (row / n_rows)
        out_L[s0:s0 + gc_len] += dying * (1 - pan)
        out_R[s0:s0 + gc_len] += dying * pan

    # Vacuum moment at ~42s: brief silence then fresh fundamentals
    vacuum_start = int(42.0 * SR)
    vacuum_silence = int(0.5 * SR)
    # Dip the drone
    dip_start = max(0, vacuum_start - int(0.3 * SR))
    dip_end = min(N, vacuum_start + vacuum_silence)
    dip_len = dip_end - dip_start
    dip_env = np.ones(dip_len)
    mid = dip_len // 2
    dip_env[:mid] = np.linspace(1, 0.2, mid)
    dip_env[mid:] = np.linspace(0.2, 1, dip_len - mid)
    out_L[dip_start:dip_end] *= dip_env
    out_R[dip_start:dip_end] *= dip_env

    # Fresh fundamentals emerge after vacuum
    fresh_start = int(43.0 * SR)
    fresh_len = int(3.0 * SR)
    if fresh_start + fresh_len <= N:
        t_f = np.arange(fresh_len) / SR
        fresh_env = envelope(fresh_len, attack=0.8, release=1.0)
        fresh = np.zeros(fresh_len)
        for freq in row_base:
            fresh += sine(freq, t_f) * 0.06
        fresh *= fresh_env
        out_L[fresh_start:fresh_start + fresh_len] += fresh * 0.5
        out_R[fresh_start:fresh_start + fresh_len] += fresh * 0.5

    # Coda: D minor chord (all rows in harmony)
    coda_start = int(49.0 * SR)
    coda_len = int(5.5 * SR)
    if coda_start + coda_len <= N:
        t_c = np.arange(coda_len) / SR
        coda_env = envelope(coda_len, attack=0.5, release=2.5)
        # D minor: D3(146.83) F3(174.61) A3(220)
        coda = (sine(146.83, t_c) * 0.1 + sine(174.61, t_c) * 0.08 +
                sine(220.0, t_c) * 0.07 + sine(293.66, t_c) * 0.05)
        coda *= coda_env
        out_L[coda_start:coda_start + coda_len] += coda
        out_R[coda_start:coda_start + coda_len] += coda

    mix = np.stack([out_L, out_R], axis=1)
    mix /= max(np.max(np.abs(mix)), 1e-6) / 0.85
    return mix


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    wav = btree_search()
    write_wav("output/db_1_btree_search.wav", wav)

    wav = transaction_isolation()
    write_wav("output/db_2_transaction_isolation.wav", wav)

    wav = mvcc()
    write_wav("output/db_3_mvcc.wav", wav)

    print("\nDone! 3 tracks generated.")
