#!/usr/bin/env python3
"""Phase 12: Concurrency — 并发原语的声化

三首：
1. Dining Philosophers (60s, stereo) — 5 位哲学家争夺 5 把叉子，
   死锁 = 冻结的和声，进食 = 满足的旋律，饥饿 = 不安的颤音。
2. Race Condition (50s, stereo) — 两个线程竞争修改共享变量，
   正确交替 = 协和音程，冲突 = 失谐碰撞，最终值漂移 = pitch drift。
3. Mutex Heartbeats (55s, stereo) — N 个进程通过互斥锁序列化访问，
   等待 = 沉默的张力，获锁 = 心跳脉冲，临界区 = 独奏时刻。
"""

import numpy as np
import os

SR = 44100

def write_wav(path: str, data: np.ndarray):
    """Write float64 stereo/mono array to 16-bit WAV."""
    import struct
    if data.ndim == 1:
        data = np.column_stack([data, data])
    data = np.clip(data, -1, 1)
    samples = (data * 32767).astype(np.int16)
    n_frames = samples.shape[0]
    with open(path, 'wb') as f:
        f.write(b'RIFF')
        data_size = n_frames * 4
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, 2, SR, SR * 4, 4, 16))
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(samples.tobytes())

def fade(n: int, fade_in: int = 0, fade_out: int = 0) -> np.ndarray:
    env = np.ones(n)
    if fade_in > 0:
        env[:fade_in] *= np.linspace(0, 1, fade_in)
    if fade_out > 0:
        env[-fade_out:] *= np.linspace(1, 0, fade_out)
    return env

# ─── 1. Dining Philosophers ─────────────────────────────────────────────

def dining_philosophers(duration=60.0):
    """5 philosophers around a table, each needs two forks to eat.
    Simulate with randomized timing; deadlock episodes freeze the soundscape."""
    n = int(SR * duration)
    out_l = np.zeros(n)
    out_r = np.zeros(n)

    n_phil = 5
    # Each philosopher has a base frequency (pentatonic on D)
    base_freqs = [146.83, 164.81, 196.0, 220.0, 261.63]  # D3, E3, G3, A3, C4
    # Stereo positions: equally spaced around the table
    pans = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Simulate discrete time steps
    step_dur = 0.15  # seconds per simulation step
    n_steps = int(duration / step_dur)

    # State: 0=thinking, 1=hungry (waiting), 2=eating
    rng = np.random.default_rng(42)

    forks = [False] * n_phil  # True = taken
    state = [0] * n_phil
    eating_time = [0] * n_phil
    thinking_time = [0] * n_phil
    hunger_time = [0] * n_phil

    for step in range(n_steps):
        t_start = int(step * step_dur * SR)
        t_end = min(int((step + 1) * step_dur * SR), n)
        seg_len = t_end - t_start
        if seg_len <= 0:
            continue
        t = np.linspace(step * step_dur, (step + 1) * step_dur, seg_len, endpoint=False)

        # Update states
        for p in range(n_phil):
            left = p
            right = (p + 1) % n_phil

            if state[p] == 0:  # thinking
                thinking_time[p] += 1
                if thinking_time[p] > rng.integers(5, 20):
                    state[p] = 1  # become hungry
                    thinking_time[p] = 0
                    hunger_time[p] = 0

            elif state[p] == 1:  # hungry, try to pick up forks
                hunger_time[p] += 1
                if not forks[left] and not forks[right]:
                    forks[left] = True
                    forks[right] = True
                    state[p] = 2
                    eating_time[p] = 0
                # Deadlock detection: if hungry too long, force drop (livelock avoidance)
                elif hunger_time[p] > 40:
                    # One philosopher yields (asymmetry breaks deadlock)
                    if p == 0:
                        state[p] = 0
                        thinking_time[p] = 0
                        hunger_time[p] = 0

            elif state[p] == 2:  # eating
                eating_time[p] += 1
                if eating_time[p] > rng.integers(8, 25):
                    forks[left] = False
                    forks[right] = False
                    state[p] = 0
                    thinking_time[p] = 0

        # Sonify current states
        for p in range(n_phil):
            f0 = base_freqs[p]
            pan_l = 1.0 - pans[p]
            pan_r = pans[p]

            if state[p] == 0:  # thinking: gentle sine, low amplitude
                amp = 0.04
                sig = amp * np.sin(2 * np.pi * f0 * t)

            elif state[p] == 1:  # hungry: tremolo, increasingly agitated
                agitation = min(hunger_time[p] / 30.0, 1.0)
                amp = 0.06 + 0.08 * agitation
                trem_rate = 4 + 12 * agitation
                sig = amp * np.sin(2 * np.pi * f0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * trem_rate * t))
                # Add dissonant overtone when very hungry
                if agitation > 0.5:
                    sig += amp * 0.3 * agitation * np.sin(2 * np.pi * f0 * 1.414 * t)

            elif state[p] == 2:  # eating: rich harmonics, warm
                amp = 0.12
                sig = amp * (
                    0.6 * np.sin(2 * np.pi * f0 * t) +
                    0.25 * np.sin(2 * np.pi * f0 * 2 * t) +
                    0.1 * np.sin(2 * np.pi * f0 * 3 * t) +
                    0.05 * np.sin(2 * np.pi * f0 * 4 * t)
                )

            out_l[t_start:t_end] += sig * pan_l
            out_r[t_start:t_end] += sig * pan_r

        # Check for deadlock: all hungry, no one eating
        all_hungry = all(s == 1 for s in state)
        if all_hungry:
            # Deadlock drone: low dissonant cluster
            amp = 0.15
            for p in range(n_phil):
                f0 = base_freqs[p]
                # Slightly detune for beating
                detune = 1.0 + 0.003 * (p - 2)
                sig = amp * np.sin(2 * np.pi * f0 * detune * t) * 0.5
                out_l[t_start:t_end] += sig * 0.5
                out_r[t_start:t_end] += sig * 0.5

    # Bass drone throughout: D1
    t_full = np.linspace(0, duration, n, endpoint=False)
    drone = 0.06 * np.sin(2 * np.pi * 36.71 * t_full)
    out_l += drone
    out_r += drone

    # Master fade
    env = fade(n, int(0.5 * SR), int(2.0 * SR))
    out_l *= env
    out_r *= env

    mx = max(np.max(np.abs(out_l)), np.max(np.abs(out_r)), 1e-6)
    out_l /= mx * 1.1
    out_r /= mx * 1.1

    return np.column_stack([out_l, out_r])

# ─── 2. Race Condition ──────────────────────────────────────────────────

def race_condition(duration=50.0):
    """Two threads increment a shared counter. When they collide (read-modify-write
    overlap), the result is wrong — sonified as pitch drift and dissonance."""
    n = int(SR * duration)
    out_l = np.zeros(n)  # Thread A
    out_r = np.zeros(n)  # Thread B

    rng = np.random.default_rng(137)

    # Simulate time slices
    slice_dur = 0.08
    n_slices = int(duration / slice_dur)

    expected_val = 0
    actual_val = 0
    base_freq = 220.0  # A3

    thread_a_active = False
    thread_b_active = False
    a_read_val = 0
    b_read_val = 0
    a_phase = 0  # 0=idle, 1=read, 2=modify, 3=write
    b_phase = 0

    drift_history = []

    for s in range(n_slices):
        t_start = int(s * slice_dur * SR)
        t_end = min(int((s + 1) * slice_dur * SR), n)
        seg_len = t_end - t_start
        if seg_len <= 0:
            continue
        t = np.linspace(s * slice_dur, (s + 1) * slice_dur, seg_len, endpoint=False)

        # Thread scheduling (random interleaving)
        a_runs = rng.random() < 0.6
        b_runs = rng.random() < 0.6

        collision = False

        if a_runs:
            if a_phase == 0:
                a_read_val = actual_val
                a_phase = 1
            elif a_phase == 1:
                a_phase = 2  # modify (increment)
            elif a_phase == 2:
                actual_val = a_read_val + 1
                a_phase = 0
                expected_val += 1

        if b_runs:
            if b_phase == 0:
                b_read_val = actual_val
                b_phase = 1
            elif b_phase == 1:
                b_phase = 2
            elif b_phase == 2:
                actual_val = b_read_val + 1
                b_phase = 0
                expected_val += 1

        # Detect race: both read before either writes
        if a_phase > 0 and b_phase > 0:
            collision = True

        drift = expected_val - actual_val
        drift_history.append(drift)
        drift_ratio = min(abs(drift) / 20.0, 1.0)

        # Thread A sound (left): pentatonic melody based on actual_val
        a_freq = base_freq * (1.0 + (actual_val % 12) * 0.05)
        a_amp = 0.15 if a_runs else 0.03
        sig_a = a_amp * np.sin(2 * np.pi * a_freq * t)
        if collision:
            # Dissonant tritone overlay
            sig_a += a_amp * 0.4 * np.sin(2 * np.pi * a_freq * 1.414 * t)

        # Thread B sound (right): slightly different base
        b_freq = base_freq * 1.5 * (1.0 + (actual_val % 12) * 0.05)
        b_amp = 0.15 if b_runs else 0.03
        sig_b = b_amp * np.sin(2 * np.pi * b_freq * t)
        if collision:
            sig_b += b_amp * 0.4 * np.sin(2 * np.pi * b_freq * 1.414 * t)

        # Drift drone: gets louder and more detuned as drift grows
        if drift > 0:
            drift_freq = 55.0 * (1.0 + drift_ratio * 0.1)
            drift_amp = 0.05 + 0.15 * drift_ratio
            drift_sig = drift_amp * np.sin(2 * np.pi * drift_freq * t)
            sig_a += drift_sig * 0.5
            sig_b += drift_sig * 0.5

        out_l[t_start:t_end] += sig_a
        out_r[t_start:t_end] += sig_b

    env = fade(n, int(0.5 * SR), int(2.0 * SR))
    out_l *= env
    out_r *= env

    mx = max(np.max(np.abs(out_l)), np.max(np.abs(out_r)), 1e-6)
    out_l /= mx * 1.1
    out_r /= mx * 1.1

    return np.column_stack([out_l, out_r])

# ─── 3. Mutex Heartbeats ────────────────────────────────────────────────

def mutex_heartbeats(duration=55.0):
    """N processes share a mutex. Only one can enter the critical section at a time.
    Waiting = tense silence with subtle pulse. Holding = solo voice.
    Handoff = percussive click."""
    n = int(SR * duration)
    out_l = np.zeros(n)
    out_r = np.zeros(n)

    n_procs = 7
    # Each process has a unique frequency and pan
    freqs = [130.81, 146.83, 164.81, 196.0, 220.0, 261.63, 293.66]  # C3 to D4
    pans = np.linspace(0.1, 0.9, n_procs)

    rng = np.random.default_rng(271)

    # Simulate: at each moment, one process holds the lock
    # Others wait with varying patience
    events = []  # (start_time, end_time, process_id)

    t_cur = 0.5  # start after brief silence
    while t_cur < duration - 2.0:
        proc = rng.integers(0, n_procs)
        hold_time = rng.uniform(0.8, 3.0)
        events.append((t_cur, min(t_cur + hold_time, duration - 2.0), proc))
        t_cur += hold_time + rng.uniform(0.05, 0.15)  # tiny gap = handoff

    t_full = np.linspace(0, duration, n, endpoint=False)

    # Background: all processes emit quiet heartbeat pulses when waiting
    for p in range(n_procs):
        # Heartbeat: periodic low-amplitude pulse
        pulse_rate = 1.2 + p * 0.15  # slightly different rates
        pulse = 0.02 * np.maximum(0, np.sin(2 * np.pi * pulse_rate * t_full)) ** 8
        carrier = np.sin(2 * np.pi * freqs[p] * 0.5 * t_full)
        sig = pulse * carrier
        out_l += sig * (1.0 - pans[p])
        out_r += sig * pans[p]

    # Critical section: solo voice with rich harmonics
    for start, end, proc in events:
        s_start = int(start * SR)
        s_end = min(int(end * SR), n)
        seg_len = s_end - s_start
        if seg_len <= 0:
            continue

        t = np.linspace(start, end, seg_len, endpoint=False)
        f0 = freqs[proc]
        pan_l = 1.0 - pans[proc]
        pan_r = pans[proc]

        # Rich solo tone with vibrato
        vib = 1.0 + 0.003 * np.sin(2 * np.pi * 5.5 * t)
        sig = (
            0.20 * np.sin(2 * np.pi * f0 * vib * t) +
            0.10 * np.sin(2 * np.pi * f0 * 2 * vib * t) +
            0.05 * np.sin(2 * np.pi * f0 * 3 * vib * t) +
            0.03 * np.sin(2 * np.pi * f0 * 4 * vib * t)
        )

        # Envelope for this segment
        seg_env = fade(seg_len, min(int(0.02 * SR), seg_len // 4), min(int(0.05 * SR), seg_len // 4))
        sig *= seg_env

        out_l[s_start:s_end] += sig * pan_l
        out_r[s_start:s_end] += sig * pan_r

        # Lock acquire click at start
        click_len = min(int(0.01 * SR), seg_len)
        click = 0.3 * np.sin(2 * np.pi * 2000 * np.linspace(0, 0.01, click_len)) * np.linspace(1, 0, click_len)
        out_l[s_start:s_start + click_len] += click * 0.5
        out_r[s_start:s_start + click_len] += click * 0.5

    # Low drone: C1
    drone = 0.05 * np.sin(2 * np.pi * 32.7 * t_full)
    out_l += drone
    out_r += drone

    env = fade(n, int(0.5 * SR), int(2.0 * SR))
    out_l *= env
    out_r *= env

    mx = max(np.max(np.abs(out_l)), np.max(np.abs(out_r)), 1e-6)
    out_l /= mx * 1.1
    out_r /= mx * 1.1

    return np.column_stack([out_l, out_r])


# ─── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    print("Generating Dining Philosophers...")
    write_wav("output/conc_1_dining_philosophers.wav", dining_philosophers())
    print("  → output/conc_1_dining_philosophers.wav")

    print("Generating Race Condition...")
    write_wav("output/conc_2_race_condition.wav", race_condition())
    print("  → output/conc_2_race_condition.wav")

    print("Generating Mutex Heartbeats...")
    write_wav("output/conc_3_mutex_heartbeats.wav", mutex_heartbeats())
    print("  → output/conc_3_mutex_heartbeats.wav")

    print("Done!")
