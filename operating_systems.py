#!/usr/bin/env python3
"""Phase 25: Operating Systems -- scheduler, virtual memory, pipes.

Three pieces:
1. Process Scheduler (55s, stereo) -- Round-robin scheduling of 6 processes.
   Each process = unique voice (frequency + timbre). Time quantum = musical phrase.
   Context switch = 2kHz click + brief silence. Priority inversion: low-priority
   process holds resource needed by high-priority → high goes silent while low
   plays its rich chord. Starvation: one process gradually fades. 55Hz kernel drone.

2. Page Replacement (50s, stereo) -- LRU page replacement in 8-frame memory.
   Each page = frequency. Page hit = warm sine sustain. Page fault = dissonant FM
   burst + new page loads. Thrashing section: working set exceeds frames, faults
   cascade into chaos. Steady state: working set fits, harmonious choir.

3. Unix Pipes (55s, stereo) -- Data flows through a 4-stage pipeline:
   cat|grep|sort|wc. Each stage transforms the signal: stage 1 raw broadband,
   stage 2 filters (bandpass), stage 3 orders (ascending arpeggio), stage 4
   compresses to single summary tone. Backpressure: upstream density builds when
   downstream is slow. EOF propagates as silence wave left→right.
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


def click_sound(n=200, freq=2000):
    t_arr = np.arange(n) / SR
    return np.sin(2 * np.pi * freq * t_arr) * np.exp(-t_arr * 40)


# Pentatonic scale frequencies
PENTA_D = [146.83, 164.81, 196.0, 220.0, 261.63, 293.66, 329.63, 392.0, 440.0, 523.25]

# ── Piece 1: Process Scheduler ──────────────────────────────────────────────

def process_scheduler():
    """Round-robin scheduling: 6 processes compete for CPU time."""
    print("Generating Process Scheduler...")
    duration = 55.0
    N = int(duration * SR)
    t = np.arange(N) / SR
    out_L = np.zeros(N)
    out_R = np.zeros(N)
    rng = np.random.default_rng(42)

    # 6 processes with different voices
    processes = [
        {"freq": 220.0, "harmonics": 3, "pan": 0.1, "priority": 5, "name": "init"},
        {"freq": 277.18, "harmonics": 4, "pan": 0.3, "priority": 3, "name": "shell"},
        {"freq": 329.63, "harmonics": 2, "pan": 0.5, "priority": 4, "name": "daemon"},
        {"freq": 392.0, "harmonics": 5, "pan": 0.7, "priority": 2, "name": "worker"},
        {"freq": 440.0, "harmonics": 3, "pan": 0.9, "priority": 1, "name": "cron"},
        {"freq": 523.25, "harmonics": 4, "pan": 0.5, "priority": 6, "name": "user"},
    ]
    n_procs = len(processes)

    # Kernel drone at 55Hz
    drone = sine(55, t) * 0.06 * envelope(N, attack=2.0, release=2.0)
    out_L += drone
    out_R += drone

    # Schedule: time quanta of 0.8-1.5s each
    cursor = 0.0
    quantum_base = 0.9
    schedule_events = []

    # Phase 1 (0-20s): Normal round-robin
    proc_idx = 0
    while cursor < 20.0:
        quantum = quantum_base + rng.uniform(-0.1, 0.3)
        schedule_events.append((cursor, proc_idx % n_procs, quantum, 1.0))
        cursor += quantum + 0.05  # 50ms context switch gap
        proc_idx += 1

    # Phase 2 (20-35s): Priority inversion — process 4 (low priority cron) holds
    # a resource needed by process 0 (high priority init)
    while cursor < 35.0:
        idx = proc_idx % n_procs
        if idx == 0:  # init blocked — silent or very quiet
            quantum = quantum_base * 0.5
            schedule_events.append((cursor, idx, quantum, 0.08))
        elif idx == 4:  # cron holding resource — plays rich
            quantum = quantum_base * 1.5
            schedule_events.append((cursor, idx, quantum, 1.2))
        else:
            quantum = quantum_base + rng.uniform(-0.1, 0.2)
            schedule_events.append((cursor, idx, quantum, 0.7))
        cursor += quantum + 0.05
        proc_idx += 1

    # Phase 3 (35-50s): Starvation — process 2 (daemon) gradually fades
    fade_start = 35.0
    while cursor < 50.0:
        idx = proc_idx % n_procs
        quantum = quantum_base + rng.uniform(-0.1, 0.3)
        vol = 1.0
        if idx == 2:
            progress = (cursor - fade_start) / (50.0 - fade_start)
            vol = max(0.02, 1.0 - progress * 0.95)
        schedule_events.append((cursor, idx, quantum, vol))
        cursor += quantum + 0.05
        proc_idx += 1

    # Phase 4 (50-55s): All processes converge to shared chord
    while cursor < 54.0:
        quantum = 0.6
        for i in range(n_procs):
            if cursor < 54.0:
                schedule_events.append((cursor, i, quantum, 0.5))
                cursor += 0.15

    # Render schedule
    for (start, pidx, quantum, vol) in schedule_events:
        p = processes[pidx]
        s = int(start * SR)
        dur = int(quantum * SR)
        if s >= N or s + dur > N:
            continue
        t_local = np.arange(dur) / SR

        # Process voice: fundamental + harmonics with process-specific timbre
        voice = np.zeros(dur)
        for h in range(1, p["harmonics"] + 1):
            amp = vol * 0.15 / h
            if h % 2 == 0:
                voice += amp * sine(p["freq"] * h, t_local)
            else:
                voice += amp * fm_tone(p["freq"] * h, p["freq"] * 0.5, 0.3, t_local)
        voice *= envelope(dur, attack=0.015, release=0.04)

        pan = p["pan"]
        out_L[s:s+dur] += voice * (1 - pan)
        out_R[s:s+dur] += voice * pan

        # Context switch click at end of quantum
        click_pos = s + dur - 200
        if 0 < click_pos < N - 200:
            c = click_sound(200, 2000) * 0.12
            out_L[click_pos:click_pos+200] += c
            out_R[click_pos:click_pos+200] += c

    # Normalize
    peak = max(np.max(np.abs(out_L)), np.max(np.abs(out_R)), 1e-6)
    if peak > 0.95:
        out_L *= 0.9 / peak
        out_R *= 0.9 / peak

    return np.stack([out_L, out_R], axis=1)

# ── Piece 2: Page Replacement ────────────────────────────────────────────────

def page_replacement():
    """LRU page replacement: hits sing, faults scream."""
    print("Generating Page Replacement...")
    duration = 50.0
    N = int(duration * SR)
    t = np.arange(N) / SR
    out_L = np.zeros(N)
    out_R = np.zeros(N)
    rng = np.random.default_rng(73)

    n_frames = 8
    n_pages = 20  # virtual pages
    # Each page has a unique frequency
    page_freqs = np.array([110 * (2 ** (i / 12)) for i in range(n_pages)])

    # Memory drone: 55Hz continuous
    drone = sine(55, t) * 0.05 * envelope(N, attack=1.5, release=2.0)
    out_L += drone
    out_R += drone

    # Generate access patterns for three phases
    accesses = []

    # Phase 1 (0-18s): Good locality — working set of 5 pages
    ws1 = [0, 1, 2, 3, 4]
    for i in range(60):
        page = rng.choice(ws1)
        accesses.append((0.3 * i, page))

    # Phase 2 (18-35s): Thrashing — working set of 16 pages in 8 frames
    ws2 = list(range(16))
    for i in range(80):
        page = rng.choice(ws2)
        accesses.append((18.0 + 0.21 * i, page))

    # Phase 3 (35-48s): Recovery — back to small working set
    ws3 = [5, 6, 7, 8, 9]
    for i in range(50):
        page = rng.choice(ws3)
        accesses.append((35.0 + 0.26 * i, page))

    # Simulate LRU
    frames = []  # list of (page, load_time)
    lru_order = []  # most recent at end

    for (access_time, page) in accesses:
        s = int(access_time * SR)
        if s >= N:
            continue
        freq = page_freqs[page % n_pages]
        is_hit = page in [f[0] for f in frames]

        if is_hit:
            # Page hit: warm sustained sine
            dur = int(0.2 * SR)
            if s + dur > N:
                dur = N - s
            t_local = np.arange(dur) / SR
            tone = sine(freq, t_local) * 0.12
            # Add gentle harmonics for warmth
            tone += sine(freq * 2, t_local) * 0.04
            tone += sine(freq * 3, t_local) * 0.02
            tone *= envelope(dur, attack=0.01, release=0.08)
            pan = (page % n_pages) / n_pages
            out_L[s:s+dur] += tone * (1 - pan)
            out_R[s:s+dur] += tone * pan
            # Update LRU
            lru_order = [p for p in lru_order if p != page]
            lru_order.append(page)
        else:
            # Page fault: FM burst + load new page
            dur = int(0.3 * SR)
            if s + dur > N:
                dur = N - s
            t_local = np.arange(dur) / SR

            # Dissonant FM fault burst
            fault_tone = fm_tone(freq * 1.414, freq * 2.7, 3.0, t_local) * 0.18
            fault_tone *= np.exp(-t_local * 12)
            # Click
            click_len = min(300, dur)
            fault_tone[:click_len] += click_sound(click_len, 1500) * 0.15

            out_L[s:s+dur] += fault_tone * 0.6
            out_R[s:s+dur] += fault_tone * 0.6

            # Evict LRU if full
            if len(frames) >= n_frames:
                if lru_order:
                    evict_page = lru_order.pop(0)
                    frames = [(p, lt) for (p, lt) in frames if p != evict_page]
                    # Eviction: brief downward sweep
                    sweep_dur = min(int(0.08 * SR), N - s)
                    if sweep_dur > 0:
                        t_sw = np.arange(sweep_dur) / SR
                        evict_freq = page_freqs[evict_page % n_pages]
                        sweep = sine(evict_freq * (1 - t_sw / (sweep_dur/SR) * 0.5), t_sw) * 0.06
                        sweep *= envelope(sweep_dur, attack=0.0, release=0.05)
                        out_L[s:s+sweep_dur] += sweep * 0.5
                        out_R[s:s+sweep_dur] += sweep * 0.5

            frames.append((page, access_time))
            lru_order = [p for p in lru_order if p != page]
            lru_order.append(page)

            # New page loads: ascending tone
            load_start = s + int(0.1 * SR)
            load_dur = int(0.15 * SR)
            if load_start + load_dur < N:
                t_ld = np.arange(load_dur) / SR
                load_tone = sine(freq, t_ld) * 0.10 * envelope(load_dur, attack=0.02, release=0.06)
                load_tone += sine(freq * 1.5, t_ld) * 0.04 * envelope(load_dur, attack=0.02, release=0.06)
                pan = (page % n_pages) / n_pages
                out_L[load_start:load_start+load_dur] += load_tone * (1 - pan)
                out_R[load_start:load_start+load_dur] += load_tone * pan

    # Coda (48-50s): All frames sound together as chord
    coda_start = int(48.0 * SR)
    coda_dur = int(2.0 * SR)
    if coda_start + coda_dur <= N:
        t_coda = np.arange(coda_dur) / SR
        coda_env = envelope(coda_dur, attack=0.3, release=0.8)
        for (page, _) in frames:
            freq = page_freqs[page % n_pages]
            tone = sine(freq, t_coda) * 0.06 * coda_env
            tone += sine(freq * 2, t_coda) * 0.02 * coda_env
            pan = (page % n_pages) / n_pages
            out_L[coda_start:coda_start+coda_dur] += tone * (1 - pan)
            out_R[coda_start:coda_start+coda_dur] += tone * pan

    peak = max(np.max(np.abs(out_L)), np.max(np.abs(out_R)), 1e-6)
    if peak > 0.95:
        out_L *= 0.9 / peak
        out_R *= 0.9 / peak

    return np.stack([out_L, out_R], axis=1)

# ── Piece 3: Unix Pipes ─────────────────────────────────────────────────────

def unix_pipes():
    """cat | grep | sort | wc — data transformation pipeline."""
    print("Generating Unix Pipes...")
    duration = 55.0
    N = int(duration * SR)
    t = np.arange(N) / SR
    out_L = np.zeros(N)
    out_R = np.zeros(N)
    rng = np.random.default_rng(99)

    # 4 stages, each at a different stereo position (left→right = pipeline flow)
    stages = [
        {"name": "cat", "pan": 0.1, "freq_base": 165.0, "harmonics": 8},   # raw broadband
        {"name": "grep", "pan": 0.35, "freq_base": 220.0, "harmonics": 4},  # filtered
        {"name": "sort", "pan": 0.65, "freq_base": 293.66, "harmonics": 3}, # ordered
        {"name": "wc", "pan": 0.9, "freq_base": 440.0, "harmonics": 1},     # summary
    ]

    # Kernel pipe drone: 55Hz
    drone = sine(55, t) * 0.05 * envelope(N, attack=1.5, release=2.0)
    out_L += drone
    out_R += drone

    # Generate data packets flowing through pipeline
    # Each packet starts at stage 0, then propagates through stages with delay
    n_packets = 60
    packet_times = np.cumsum(rng.uniform(0.3, 0.9, n_packets))
    # Scale to fit in 0-42s
    packet_times = packet_times / packet_times[-1] * 42.0

    stage_delay = 1.2  # seconds between stages
    packet_freqs = rng.choice(PENTA_D[:8], size=n_packets)
    # Some packets are "filtered out" by grep (30% rejection)
    grep_pass = rng.random(n_packets) > 0.3

    for pi in range(n_packets):
        base_time = packet_times[pi]
        freq = packet_freqs[pi]

        for si, stage in enumerate(stages):
            arrival = base_time + si * stage_delay
            s = int(arrival * SR)
            if s >= N or s < 0:
                continue

            # grep stage filters some packets
            if si >= 1 and not grep_pass[pi]:
                # Rejected packet: brief dissonant blip at grep stage
                if si == 1:
                    blip_dur = int(0.06 * SR)
                    if s + blip_dur < N:
                        t_blip = np.arange(blip_dur) / SR
                        blip = fm_tone(freq * 1.414, freq * 3, 2.0, t_blip) * 0.08
                        blip *= np.exp(-t_blip * 30)
                        pan = stage["pan"]
                        out_L[s:s+blip_dur] += blip * (1 - pan)
                        out_R[s:s+blip_dur] += blip * pan
                continue  # don't propagate further

            # Stage-specific sound
            dur = int(0.25 * SR)
            if s + dur > N:
                dur = N - s
            if dur <= 0:
                continue
            t_local = np.arange(dur) / SR

            if si == 0:
                # cat: raw broadband — many harmonics, noisy
                tone = np.zeros(dur)
                for h in range(1, stage["harmonics"] + 1):
                    amp = 0.10 / h
                    tone += amp * sine(freq * h, t_local, rng.uniform(0, 2*np.pi))
                # Add noise texture
                tone += rng.normal(0, 0.015, dur)
                tone *= envelope(dur, attack=0.005, release=0.05)

            elif si == 1:
                # grep: bandpass filtered — fewer harmonics, cleaner
                tone = np.zeros(dur)
                for h in range(1, stage["harmonics"] + 1):
                    amp = 0.12 / (h * 0.7)
                    tone += amp * sine(freq * h, t_local)
                tone *= envelope(dur, attack=0.01, release=0.06)

            elif si == 2:
                # sort: ordered ascending arpeggio
                arp_notes = sorted([freq, freq * 1.25, freq * 1.5])
                note_dur = dur // 3
                tone = np.zeros(dur)
                for ni, nf in enumerate(arp_notes):
                    ns = ni * note_dur
                    ne = min(ns + note_dur, dur)
                    if ne > ns:
                        t_n = np.arange(ne - ns) / SR
                        tone[ns:ne] = sine(nf, t_n) * 0.11
                        tone[ns:ne] += sine(nf * 2, t_n) * 0.03
                tone *= envelope(dur, attack=0.01, release=0.08)

            elif si == 3:
                # wc: single pure summary tone
                tone = sine(stage["freq_base"], t_local) * 0.13
                tone *= envelope(dur, attack=0.02, release=0.1)

            pan = stage["pan"]
            out_L[s:s+dur] += tone * (1 - pan)
            out_R[s:s+dur] += tone * pan

        # Pipe transfer click between stages
        for si in range(3):
            if si >= 1 and not grep_pass[pi]:
                break
            xfer_time = base_time + (si + 0.5) * stage_delay
            xs = int(xfer_time * SR)
            if 0 < xs < N - 150:
                c = click_sound(150, 1800) * 0.06
                mid_pan = (stages[si]["pan"] + stages[si+1]["pan"]) / 2
                out_L[xs:xs+150] += c * (1 - mid_pan)
                out_R[xs:xs+150] += c * mid_pan

    # Backpressure section (42-48s): upstream density builds
    bp_start = 42.0
    bp_dur_total = 6.0
    for i in range(40):
        progress = i / 40
        bp_time = bp_start + progress * bp_dur_total
        s = int(bp_time * SR)
        dur = int(0.15 * SR)
        if s + dur >= N:
            continue
        t_local = np.arange(dur) / SR
        # Upstream stages get denser and louder
        for si in range(2):  # cat and grep back up
            stage = stages[si]
            density_amp = 0.08 * (1 + progress * 2)  # crescendo
            tone = sine(stage["freq_base"] * (1 + rng.uniform(-0.02, 0.02)), t_local) * density_amp
            for h in range(2, 5):
                tone += sine(stage["freq_base"] * h, t_local) * density_amp * 0.3 / h
            tone *= envelope(dur, attack=0.005, release=0.03)
            pan = stage["pan"]
            out_L[s:s+dur] += tone * (1 - pan)
            out_R[s:s+dur] += tone * pan

    # EOF propagation (48-54s): silence wave left→right
    eof_start = 48.0
    eof_dur_per_stage = 1.2
    for si, stage in enumerate(stages):
        # Each stage goes silent with a final dying tone
        eof_time = eof_start + si * eof_dur_per_stage
        es = int(eof_time * SR)
        dur = int(1.0 * SR)
        if es + dur > N:
            dur = N - es
        if dur <= 0:
            continue
        t_local = np.arange(dur) / SR
        # Descending glissando
        freq_start = stage["freq_base"]
        freq_end = stage["freq_base"] * 0.5
        freqs = np.linspace(freq_start, freq_end, dur)
        phase = np.cumsum(2 * np.pi * freqs / SR)
        eof_tone = np.sin(phase) * 0.10 * np.exp(-t_local * 2.5)
        pan = stage["pan"]
        out_L[es:es+dur] += eof_tone * (1 - pan)
        out_R[es:es+dur] += eof_tone * pan

    peak = max(np.max(np.abs(out_L)), np.max(np.abs(out_R)), 1e-6)
    if peak > 0.95:
        out_L *= 0.9 / peak
        out_R *= 0.9 / peak

    return np.stack([out_L, out_R], axis=1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("output", exist_ok=True)
    os.makedirs("docs/audio", exist_ok=True)

    pieces = [
        ("os_1_process_scheduler", process_scheduler),
        ("os_2_page_replacement", page_replacement),
        ("os_3_unix_pipes", unix_pipes),
    ]

    for name, fn in pieces:
        audio = fn()
        write_wav(f"output/{name}.wav", audio)

    print("\nPhase 25: Operating Systems — done.")


if __name__ == "__main__":
    main()
