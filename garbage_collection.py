#!/usr/bin/env python3
"""Phase 27: Garbage Collection -- mark-sweep, reference counting, generational.

Three pieces:
1. Mark-Sweep (55s, stereo) -- Object graph with roots on the left. Mark phase
   sweeps through reachable objects (bright FM pings propagating along edges),
   unreachable objects dim. Sweep phase: unreachable objects emit descending
   death tones and silence. Three GC cycles with increasing fragmentation
   (gaps in the frequency spectrum). 55Hz runtime drone.

2. Reference Counting (50s, stereo) -- Each object has a ref count mapped to
   harmonic richness (count=1 pure sine, count=5 rich harmonics). Increments
   add harmonics with rising click, decrements remove with falling click.
   Ref count reaching zero triggers immediate reclamation (quick descending
   sweep). Cycle detection failure: two objects with count=1 form a tritone
   beating pair that never resolves -- memory leak as dissonance.

3. Generational GC (55s, stereo) -- Three generations: nursery (high pitch,
   fast allocation rate, frequent minor GC), survivor (mid pitch, less
   frequent), tenured (low pitch, rare major GC). Objects promoted between
   generations with ascending frequency glide. Minor GC = quick high sweep,
   major GC = dramatic full-spectrum pause (stop-the-world silence then burst).
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
        env[-min(rel, n):] = np.linspace(1, 0, min(rel, n))
    return env


def mix_to(buf, signal, start, pan=0.5):
    n = len(signal)
    end = min(start + n, len(buf))
    seg = signal[:end - start]
    buf[start:end, 0] += seg * (1 - pan)
    buf[start:end, 1] += seg * pan


def click(freq=2000, dur=0.008):
    t = np.linspace(0, dur, int(SR * dur), False)
    return sine(freq, t) * envelope(len(t), 0.001, dur * 0.6) * 0.3

def generate_mark_sweep():
    """Mark-Sweep GC: object graph mark phase then sweep phase."""
    duration = 55.0
    n = int(SR * duration)
    buf = np.zeros((n, 2))
    t_full = np.linspace(0, duration, n, False)

    # 55Hz runtime drone
    drone = sine(55, t_full) * 0.06 * envelope(n, 2.0, 3.0)
    buf[:, 0] += drone
    buf[:, 1] += drone

    np.random.seed(42)

    # Object graph: 20 objects with random connections
    num_objects = 20
    # Frequencies: log-spaced 110-880Hz
    freqs = np.geomspace(110, 880, num_objects)
    # Random edges
    edges = []
    for i in range(num_objects):
        num_edges = np.random.randint(0, 3)
        for _ in range(num_edges):
            j = np.random.randint(0, num_objects)
            if j != i:
                edges.append((i, j))

    # Three GC cycles
    cycle_dur = duration / 3.3
    for cycle in range(3):
        cycle_start = cycle * cycle_dur + 0.5
        # Roots: first 3-5 objects reachable
        num_roots = 5 - cycle  # fewer roots each cycle = more garbage
        reachable = set()
        stack = list(range(num_roots))
        while stack:
            obj = stack.pop()
            if obj in reachable:
                continue
            reachable.add(obj)
            for (a, b) in edges:
                if a == obj and b not in reachable:
                    stack.append(b)

        # Mark phase: bright FM pings for reachable objects
        mark_dur = cycle_dur * 0.45
        for idx, obj in enumerate(sorted(reachable)):
            t_mark = cycle_start + (idx / max(len(reachable), 1)) * mark_dur
            s = int(t_mark * SR)
            freq = freqs[obj]
            note_len = int(0.15 * SR)
            if s + note_len > n:
                break
            t_note = np.linspace(0, 0.15, note_len, False)
            tone = fm_tone(freq, freq * 0.5, 2.0, t_note) * envelope(note_len, 0.005, 0.08) * 0.18
            pan = obj / num_objects
            mix_to(buf, tone, s, pan)

        # Sweep phase: unreachable objects die
        sweep_start = cycle_start + mark_dur + 0.2
        unreachable = sorted(set(range(num_objects)) - reachable)
        sweep_dur = cycle_dur * 0.35
        for idx, obj in enumerate(unreachable):
            t_sweep = sweep_start + (idx / max(len(unreachable), 1)) * sweep_dur
            s = int(t_sweep * SR)
            freq = freqs[obj]
            # Death tone: descending sweep
            death_len = int(0.2 * SR)
            if s + death_len > n:
                break
            t_death = np.linspace(0, 0.2, death_len, False)
            sweep_freq = freq * (1 - t_death / 0.2 * 0.7)
            tone = sine(sweep_freq, t_death) * envelope(death_len, 0.002, 0.15) * 0.12
            pan = obj / num_objects
            mix_to(buf, tone, s, pan)
            # Click at death
            ck = click(1500, 0.005)
            if s + len(ck) <= n:
                mix_to(buf, ck, s, pan)

        # Fragmentation: gaps marked by silence + subtle noise
        if cycle > 0:
            frag_start = int((sweep_start + sweep_dur * 0.8) * SR)
            frag_len = int(0.3 * SR)
            if frag_start + frag_len <= n:
                noise = np.random.randn(frag_len) * 0.02 * (cycle / 3)
                buf[frag_start:frag_start + frag_len, 0] += noise
                buf[frag_start:frag_start + frag_len, 1] += noise

    # Coda: all surviving objects as chord
    coda_start = int((duration - 4) * SR)
    coda_len = int(3.5 * SR)
    if coda_start + coda_len <= n:
        t_coda = np.linspace(0, 3.5, coda_len, False)
        chord = np.zeros(coda_len)
        final_reachable = set()
        stack = list(range(2))
        while stack:
            obj = stack.pop()
            if obj in final_reachable:
                continue
            final_reachable.add(obj)
            for (a, b) in edges:
                if a == obj and b not in final_reachable:
                    stack.append(b)
        for obj in sorted(final_reachable)[:8]:
            chord += sine(freqs[obj], t_coda) * (0.08 / max(len(final_reachable), 1))
        chord *= envelope(coda_len, 0.5, 2.0)
        buf[coda_start:coda_start + coda_len, 0] += chord
        buf[coda_start:coda_start + coda_len, 1] += chord

    return buf

def generate_ref_counting():
    """Reference Counting: harmonic richness = ref count, cycle leak = tritone."""
    duration = 50.0
    n = int(SR * duration)
    buf = np.zeros((n, 2))
    t_full = np.linspace(0, duration, n, False)

    np.random.seed(77)

    # 12 objects, each with a base frequency (D pentatonic spread)
    penta = [146.83, 164.81, 196.0, 220.0, 261.63, 293.66, 329.63, 369.99, 440.0, 493.88, 523.25, 587.33]
    num_obj = 12
    ref_counts = np.ones(num_obj, dtype=int)  # all start at 1

    # 55Hz drone
    drone = sine(55, t_full) * 0.05 * envelope(n, 1.5, 2.5)
    buf[:, 0] += drone
    buf[:, 1] += drone

    # Phase 1 (0-18s): normal operations - inc/dec ref counts
    events = []
    t_pos = 0.5
    for _ in range(60):
        obj = np.random.randint(0, num_obj)
        action = np.random.choice(['inc', 'dec', 'dec'], p=[0.45, 0.35, 0.20])
        events.append((t_pos, obj, action))
        t_pos += np.random.uniform(0.2, 0.4)
        if t_pos > 18:
            break

    for t_ev, obj, action in events:
        s = int(t_ev * SR)
        freq = penta[obj]
        if action == 'inc':
            ref_counts[obj] = min(ref_counts[obj] + 1, 8)
            # Add harmonic - rising click + tone with N harmonics
            note_len = int(0.2 * SR)
            if s + note_len > n:
                continue
            t_note = np.linspace(0, 0.2, note_len, False)
            tone = np.zeros(note_len)
            for h in range(1, ref_counts[obj] + 1):
                tone += sine(freq * h, t_note) * (0.12 / h)
            tone *= envelope(note_len, 0.005, 0.1)
            pan = obj / num_obj
            mix_to(buf, tone, s, pan)
            ck = click(2500, 0.004)
            if s + len(ck) <= n:
                mix_to(buf, ck, s, pan)
        else:
            ref_counts[obj] = max(ref_counts[obj] - 1, 0)
            if ref_counts[obj] == 0:
                # Immediate reclamation: quick descending sweep
                death_len = int(0.25 * SR)
                if s + death_len > n:
                    continue
                t_death = np.linspace(0, 0.25, death_len, False)
                sweep_f = freq * np.exp(-t_death * 8)
                tone = sine(sweep_f, t_death) * envelope(death_len, 0.002, 0.2) * 0.15
                pan = obj / num_obj
                mix_to(buf, tone, s, pan)
                ref_counts[obj] = 1  # respawn for more events
            else:
                note_len = int(0.15 * SR)
                if s + note_len > n:
                    continue
                t_note = np.linspace(0, 0.15, note_len, False)
                tone = np.zeros(note_len)
                for h in range(1, ref_counts[obj] + 1):
                    tone += sine(freq * h, t_note) * (0.1 / h)
                tone *= envelope(note_len, 0.005, 0.08)
                pan = obj / num_obj
                mix_to(buf, tone, s, pan)
                ck = click(1200, 0.004)
                if s + len(ck) <= n:
                    mix_to(buf, ck, s, pan)

    # Phase 2 (18-35s): cycle formation - two objects form tritone beating pair
    cycle_start = int(18 * SR)
    cycle_end = int(35 * SR)
    cycle_len = cycle_end - cycle_start
    t_cycle = np.linspace(0, 17, cycle_len, False)
    # Object A at 220Hz, Object B at 220*sqrt(2) = tritone
    freq_a, freq_b = 220.0, 220.0 * np.sqrt(2)
    tone_a = sine(freq_a, t_cycle) * 0.12 * envelope(cycle_len, 1.0, 2.0)
    tone_b = sine(freq_b, t_cycle) * 0.12 * envelope(cycle_len, 1.0, 2.0)
    # Beating intensifies
    beat_mod = 1 + 0.3 * np.sin(2 * np.pi * 2 * t_cycle) * np.minimum(t_cycle / 10, 1)
    buf[cycle_start:cycle_end, 0] += tone_a * beat_mod
    buf[cycle_start:cycle_end, 1] += tone_b * beat_mod

    # Phase 3 (35-47s): more normal operations around the stuck pair
    t_pos = 35.5
    for _ in range(30):
        obj = np.random.randint(0, num_obj)
        s = int(t_pos * SR)
        freq = penta[obj]
        note_len = int(0.12 * SR)
        if s + note_len <= n:
            t_note = np.linspace(0, 0.12, note_len, False)
            tone = sine(freq, t_note) * envelope(note_len, 0.005, 0.06) * 0.1
            mix_to(buf, tone, s, obj / num_obj)
        t_pos += np.random.uniform(0.25, 0.5)
        if t_pos > 47:
            break

    # Coda (47-50s): tritone leak fades, leaving D major resolution
    coda_start = int(47 * SR)
    coda_len = int(3 * SR)
    if coda_start + coda_len <= n:
        t_coda = np.linspace(0, 3, coda_len, False)
        # D major: D4 293.66, F#4 369.99, A4 440
        chord = (sine(293.66, t_coda) + sine(369.99, t_coda) * 0.8 + sine(440, t_coda) * 0.7) * 0.08
        chord *= envelope(coda_len, 0.3, 2.0)
        buf[coda_start:coda_start + coda_len, 0] += chord
        buf[coda_start:coda_start + coda_len, 1] += chord

    return buf

def generate_generational():
    """Generational GC: nursery/survivor/tenured with promotion and stop-the-world."""
    duration = 55.0
    n = int(SR * duration)
    buf = np.zeros((n, 2))
    t_full = np.linspace(0, duration, n, False)

    np.random.seed(99)

    # 55Hz drone
    drone = sine(55, t_full) * 0.05 * envelope(n, 1.5, 3.0)
    buf[:, 0] += drone
    buf[:, 1] += drone

    # Three generations:
    # Nursery: 660-880Hz (high, bright), frequent minor GC
    # Survivor: 330-440Hz (mid), less frequent
    # Tenured: 110-220Hz (low, warm), rare major GC
    gens = [
        {'name': 'nursery', 'freq_lo': 660, 'freq_hi': 880, 'pan': 0.3, 'gc_interval': 2.5, 'harmonics': 2},
        {'name': 'survivor', 'freq_lo': 330, 'freq_hi': 440, 'pan': 0.5, 'gc_interval': 7.0, 'harmonics': 4},
        {'name': 'tenured', 'freq_lo': 110, 'freq_hi': 220, 'pan': 0.7, 'gc_interval': 18.0, 'harmonics': 6},
    ]

    # Allocation: continuous stream of nursery objects
    alloc_rate = 8  # objects per second
    total_allocs = int(duration * alloc_rate * 0.8)

    for i in range(total_allocs):
        t_alloc = (i / alloc_rate) + 0.3
        if t_alloc >= duration - 3:
            break
        s = int(t_alloc * SR)
        freq = np.random.uniform(gens[0]['freq_lo'], gens[0]['freq_hi'])
        note_len = int(0.04 * SR)
        if s + note_len > n:
            continue
        t_note = np.linspace(0, 0.04, note_len, False)
        tone = sine(freq, t_note) * envelope(note_len, 0.002, 0.02) * 0.06
        mix_to(buf, tone, s, gens[0]['pan'] + np.random.uniform(-0.1, 0.1))

    # Minor GC events (nursery sweep)
    minor_gc_times = np.arange(2.5, duration - 5, 2.5)
    for t_gc in minor_gc_times:
        s = int(t_gc * SR)
        # Quick high-frequency sweep
        sweep_len = int(0.15 * SR)
        if s + sweep_len > n:
            continue
        t_sweep = np.linspace(0, 0.15, sweep_len, False)
        sweep_f = np.linspace(880, 660, sweep_len)
        tone = sine(sweep_f, t_sweep) * envelope(sweep_len, 0.002, 0.1) * 0.15
        mix_to(buf, tone, s, 0.3)
        # Click
        ck = click(2000, 0.006)
        if s + len(ck) <= n:
            mix_to(buf, ck, s, 0.3)

    # Promotions: some objects promoted nursery -> survivor -> tenured
    promotion_times = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    for idx, t_promo in enumerate(promotion_times):
        s = int(t_promo * SR)
        # Ascending frequency glide
        from_gen = 0 if idx % 3 != 2 else 1
        to_gen = 1 if from_gen == 0 else 2
        glide_len = int(0.3 * SR)
        if s + glide_len > n:
            continue
        t_glide = np.linspace(0, 0.3, glide_len, False)
        f_start = np.mean([gens[from_gen]['freq_lo'], gens[from_gen]['freq_hi']])
        f_end = np.mean([gens[to_gen]['freq_lo'], gens[to_gen]['freq_hi']])
        # Glide downward (promotion = moving to lower freq gen)
        glide_freq = f_start + (f_end - f_start) * (t_glide / 0.3)
        num_h = gens[to_gen]['harmonics']
        tone = np.zeros(glide_len)
        for h in range(1, num_h + 1):
            tone += sine(glide_freq * h, t_glide) * (0.08 / h)
        tone *= envelope(glide_len, 0.01, 0.15)
        pan = gens[from_gen]['pan'] + (gens[to_gen]['pan'] - gens[from_gen]['pan']) * (t_glide / 0.3)
        # Mix with varying pan
        for k in range(glide_len):
            if s + k < n:
                p = pan[k] if hasattr(pan, '__len__') else pan
                buf[s + k, 0] += tone[k] * (1 - p)
                buf[s + k, 1] += tone[k] * p

    # Major GC: stop-the-world pause at t=22 and t=42
    for t_major in [22.0, 42.0]:
        pause_start = int(t_major * SR)
        pause_len = int(0.4 * SR)
        # Silence gap (stop-the-world)
        fade_len = int(0.05 * SR)
        if pause_start + pause_len + fade_len > n:
            continue
        # Fade out before pause
        for k in range(fade_len):
            factor = 1 - k / fade_len
            buf[pause_start + k] *= factor
        # Silence
        buf[pause_start + fade_len:pause_start + pause_len] *= 0.05
        # Burst after pause: full-spectrum sweep
        burst_start = pause_start + pause_len
        burst_len = int(0.5 * SR)
        if burst_start + burst_len > n:
            continue
        t_burst = np.linspace(0, 0.5, burst_len, False)
        burst = np.zeros(burst_len)
        for gen in gens:
            f = np.mean([gen['freq_lo'], gen['freq_hi']])
            for h in range(1, gen['harmonics'] + 1):
                burst += sine(f * h, t_burst) * (0.04 / h)
        burst *= envelope(burst_len, 0.01, 0.3)
        buf[burst_start:burst_start + burst_len, 0] += burst
        buf[burst_start:burst_start + burst_len, 1] += burst
        # Major GC click
        ck = click(1000, 0.01)
        if burst_start + len(ck) <= n:
            mix_to(buf, ck, burst_start, 0.5)

    # Survivor generation sustained tones (background)
    surv_start = int(5 * SR)
    surv_len = int(40 * SR)
    if surv_start + surv_len <= n:
        t_surv = np.linspace(0, 40, surv_len, False)
        surv_tone = sine(385, t_surv) * 0.03 * envelope(surv_len, 3.0, 5.0)
        surv_tone += sine(385 * 2, t_surv) * 0.015 * envelope(surv_len, 3.0, 5.0)
        buf[surv_start:surv_start + surv_len, 0] += surv_tone * 0.6
        buf[surv_start:surv_start + surv_len, 1] += surv_tone * 0.4

    # Tenured generation drone (background, very low)
    ten_start = int(15 * SR)
    ten_len = int(35 * SR)
    if ten_start + ten_len <= n:
        t_ten = np.linspace(0, 35, ten_len, False)
        ten_tone = sine(165, t_ten) * 0.04 * envelope(ten_len, 5.0, 8.0)
        buf[ten_start:ten_start + ten_len, 0] += ten_tone * 0.4
        buf[ten_start:ten_start + ten_len, 1] += ten_tone * 0.6

    # Coda: all three generation tones converge
    coda_start = int(51 * SR)
    coda_len = int(3.5 * SR)
    if coda_start + coda_len <= n:
        t_coda = np.linspace(0, 3.5, coda_len, False)
        chord = np.zeros(coda_len)
        # D minor: D3(146.83) + F3(174.61) + A3(220) + D4(293.66)
        for f in [146.83, 174.61, 220.0, 293.66]:
            chord += sine(f, t_coda) * 0.06
        chord *= envelope(coda_len, 0.3, 2.5)
        buf[coda_start:coda_start + coda_len, 0] += chord
        buf[coda_start:coda_start + coda_len, 1] += chord

    return buf

def main():
    os.makedirs("output", exist_ok=True)
    print("Phase 27: Garbage Collection")
    print("=" * 40)

    print("\n1. Mark-Sweep")
    buf = generate_mark_sweep()
    write_wav("output/gc_1_mark_sweep.wav", buf)

    print("\n2. Reference Counting")
    buf = generate_ref_counting()
    write_wav("output/gc_2_ref_counting.wav", buf)

    print("\n3. Generational GC")
    buf = generate_generational()
    write_wav("output/gc_3_generational.wav", buf)

    print("\nDone!")


if __name__ == "__main__":
    main()
