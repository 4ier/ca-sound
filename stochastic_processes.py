#!/usr/bin/env python3
"""Phase 29: Stochastic Processes -- Markov chains, Brownian motion, Monte Carlo.

Three pieces:
1. Markov Chain (55s, stereo) -- A 6-state ergodic Markov chain. Each state maps
   to a pitch in D minor pentatonic. Transition = portamento glide + click.
   Stationary distribution emerges audibly: frequently visited states accumulate
   as persistent drone harmonics. Left channel = current state melody, right =
   accumulated stationary distribution chord growing richer over time.

2. Brownian Motion (50s, stereo) -- 12 independent Brownian particles start at
   middle frequency (330Hz). Random walks in log-frequency space create
   diverging pitch trajectories. Position -> frequency, velocity -> brightness.
   Particles that drift far from origin get louder (outlier amplification).
   Stereo position = particle index spread. The RMS displacement grows as sqrt(t)
   -- you hear the ensemble spread from unison to wide cluster.

3. Monte Carlo Pi (55s, stereo) -- Estimating pi by throwing darts at unit square.
   Hit (inside circle) = consonant major chord ping (left-biased). Miss (outside) =
   dissonant minor second blip (right-biased). Running estimate of pi modulates
   a continuous drone frequency (pi*100 = 314.159Hz target). Early wild swings
   in estimate = pitch instability, gradual convergence = pitch settling.
   Click marks each 100-dart batch. Final chord at pi frequency.
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
    end = min(start + len(buf) - start if start < len(buf) else 0, n)
    end_buf = min(start + n, len(buf))
    seg = signal[:end_buf - start]
    buf[start:end_buf, 0] += seg * (1 - pan)
    buf[start:end_buf, 1] += seg * pan


def click(freq=2000, dur=0.008):
    t = np.linspace(0, dur, int(SR * dur), False)
    return sine(freq, t) * envelope(len(t), 0.001, dur * 0.6) * 0.3


def generate_markov():
    """Markov Chain: ergodic 6-state chain, stationary distribution emerges."""
    duration = 55.0
    n = int(SR * duration)
    buf = np.zeros((n, 2))
    t_full = np.linspace(0, duration, n, False)
    rng = np.random.default_rng(42)

    # 55Hz drone
    drone = sine(55, t_full) * 0.04 * envelope(n, 2.0, 3.0)
    buf[:, 0] += drone
    buf[:, 1] += drone

    # D minor pentatonic: D4, F4, G4, A4, C5, D5
    pitches = [293.66, 349.23, 392.00, 440.00, 523.25, 587.33]
    n_states = 6

    # Transition matrix (ergodic, slightly favoring neighbors)
    P = np.array([
        [0.15, 0.30, 0.15, 0.15, 0.10, 0.15],
        [0.25, 0.10, 0.30, 0.10, 0.15, 0.10],
        [0.10, 0.25, 0.10, 0.30, 0.10, 0.15],
        [0.15, 0.10, 0.25, 0.10, 0.30, 0.10],
        [0.10, 0.15, 0.10, 0.25, 0.10, 0.30],
        [0.25, 0.10, 0.15, 0.10, 0.25, 0.15],
    ])

    # Compute stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stat_dist = np.real(eigenvectors[:, idx])
    stat_dist = stat_dist / stat_dist.sum()

    # Simulate chain
    steps = 400
    step_dur = duration / steps
    state = 0
    visit_counts = np.zeros(n_states)

    for step in range(steps):
        visit_counts[state] += 1
        s = int(step * step_dur * SR)
        grain_len = int(step_dur * SR * 0.85)
        if s + grain_len > n:
            break

        freq = pitches[state]
        t_grain = np.linspace(0, grain_len / SR, grain_len, False)

        # Left channel: current state melody (clear tone + vibrato)
        vibrato = 1 + 0.003 * sine(5.5, t_grain)
        melody = sine(freq * vibrato, t_grain) * 0.12
        melody += sine(freq * 2 * vibrato, t_grain) * 0.04  # octave harmonic
        melody *= envelope(grain_len, 0.005, step_dur * 0.3)
        mix_to(buf, melody, s, 0.3)

        # Transition click
        click_s = s + grain_len - int(0.01 * SR)
        if click_s > 0 and click_s + int(0.008 * SR) < n:
            mix_to(buf, click(1500, 0.008), click_s, 0.5)

        # Right channel: accumulated stationary distribution chord
        if step > 20:  # let it build up
            current_dist = visit_counts / visit_counts.sum()
            chord_len = int(step_dur * SR * 0.6)
            t_chord = np.linspace(0, chord_len / SR, chord_len, False)
            chord = np.zeros(chord_len)
            for si in range(n_states):
                weight = current_dist[si]
                if weight > 0.05:
                    chord += sine(pitches[si], t_chord) * weight * 0.08
            chord *= envelope(chord_len, 0.01, step_dur * 0.3)
            mix_to(buf, chord, s, 0.75)

        # Next state
        state = rng.choice(n_states, p=P[state])

    # Coda: full stationary distribution chord
    coda_start = int(51 * SR)
    coda_len = int(3.5 * SR)
    if coda_start + coda_len <= n:
        t_coda = np.linspace(0, 3.5, coda_len, False)
        coda_chord = np.zeros(coda_len)
        for si in range(n_states):
            coda_chord += sine(pitches[si], t_coda) * stat_dist[si] * 0.12
            coda_chord += sine(pitches[si] * 2, t_coda) * stat_dist[si] * 0.04
        coda_chord *= envelope(coda_len, 0.5, 2.5)
        buf[coda_start:coda_start + coda_len, 0] += coda_chord
        buf[coda_start:coda_start + coda_len, 1] += coda_chord

    return buf


def generate_brownian():
    """Brownian Motion: 12 particles diverging from unison."""
    duration = 50.0
    n = int(SR * duration)
    buf = np.zeros((n, 2))
    t_full = np.linspace(0, duration, n, False)
    rng = np.random.default_rng(73)

    # Low drone
    drone = sine(55, t_full) * 0.03 * envelope(n, 1.5, 3.0)
    buf[:, 0] += drone
    buf[:, 1] += drone

    n_particles = 12
    steps = 500
    step_dur = duration / steps

    # Brownian motion in log-frequency space, centered at ln(330)
    log_center = np.log(330)
    sigma = 0.03  # step size in log-frequency
    positions = np.full(n_particles, log_center)  # start at unison
    velocities = np.zeros(n_particles)

    for step in range(steps):
        s = int(step * step_dur * SR)
        grain_len = int(step_dur * SR * 0.9)
        if s + grain_len > n:
            break

        t_grain = np.linspace(0, grain_len / SR, grain_len, False)

        # Update positions (Brownian increments)
        increments = rng.normal(0, sigma, n_particles)
        velocities = increments / (step_dur + 1e-6)
        positions += increments

        # Clamp to audible range
        positions = np.clip(positions, np.log(80), np.log(2000))

        for p in range(n_particles):
            freq = np.exp(positions[p])
            # Distance from center -> amplitude (outliers louder)
            dist = abs(positions[p] - log_center)
            amp = 0.04 + 0.04 * min(dist / 1.5, 1.0)
            # Velocity -> brightness (more harmonics)
            speed = abs(velocities[p])
            n_harmonics = max(1, min(4, int(speed / 0.5) + 1))
            # Pan: spread by particle index
            pan = 0.1 + 0.8 * (p / (n_particles - 1))

            grain = np.zeros(grain_len)
            for h in range(1, n_harmonics + 1):
                grain += sine(freq * h, t_grain) * (amp / h)
            grain *= envelope(grain_len, 0.003, step_dur * 0.2)
            mix_to(buf, grain, s, pan)

    # Coda: particles freeze, slow fade
    coda_start = int(46 * SR)
    coda_len = int(3.5 * SR)
    if coda_start + coda_len <= n:
        t_coda = np.linspace(0, 3.5, coda_len, False)
        coda = np.zeros(coda_len)
        for p in range(n_particles):
            freq = np.exp(positions[p])
            coda += sine(freq, t_coda) * 0.03
        coda *= envelope(coda_len, 0.2, 3.0)
        buf[coda_start:coda_start + coda_len, 0] += coda * 0.5
        buf[coda_start:coda_start + coda_len, 1] += coda * 0.5

    return buf


def generate_monte_carlo():
    """Monte Carlo Pi: dart throwing converges to pi."""
    duration = 55.0
    n = int(SR * duration)
    buf = np.zeros((n, 2))
    t_full = np.linspace(0, duration, n, False)
    rng = np.random.default_rng(314)

    # Target frequency: pi * 100 = 314.159 Hz
    pi_freq = np.pi * 100

    # Total darts
    total_darts = 2000
    dart_dur = duration / total_darts
    hits = 0

    # Continuous drone: frequency tracks running pi estimate
    # We'll build it in segments
    for d in range(total_darts):
        x, y = rng.uniform(-1, 1, 2)
        inside = (x*x + y*y) <= 1.0
        if inside:
            hits += 1

        s = int(d * dart_dur * SR)
        grain_len = int(dart_dur * SR * 0.8)
        if s + grain_len > n:
            break

        t_grain = np.linspace(0, grain_len / SR, grain_len, False)

        # Running estimate
        estimate = 4.0 * hits / (d + 1) if d > 0 else 4.0
        est_freq = estimate * 100  # target: 314.159

        # Drone at estimate frequency (continuous)
        drone_amp = 0.06
        drone = sine(est_freq, t_grain) * drone_amp
        drone += sine(est_freq * 1.5, t_grain) * drone_amp * 0.3  # fifth
        drone *= envelope(grain_len, 0.002, 0.005)
        mix_to(buf, drone, s, 0.5)

        # Dart sound
        if inside:
            # Hit: consonant ping (C major triad fragment), left-biased
            ping_freq = 523.25  # C5
            ping = sine(ping_freq, t_grain) * 0.06
            ping += sine(ping_freq * 1.25, t_grain) * 0.03  # major third
            ping *= envelope(grain_len, 0.001, dart_dur * 0.5)
            mix_to(buf, ping, s, 0.3)
        else:
            # Miss: dissonant blip, right-biased
            blip_freq = 466.16  # Bb4 (minor second against A4)
            blip = fm_tone(blip_freq, 7, 2.0, t_grain) * 0.04
            blip *= envelope(grain_len, 0.001, dart_dur * 0.3)
            mix_to(buf, blip, s, 0.7)

        # Batch click every 100 darts
        if (d + 1) % 100 == 0 and s + int(0.01 * SR) < n:
            mix_to(buf, click(2000, 0.01), s, 0.5)

    # 55Hz drone throughout
    base_drone = sine(55, t_full) * 0.03 * envelope(n, 2.0, 3.0)
    buf[:, 0] += base_drone
    buf[:, 1] += base_drone

    # Coda: resolve to pi frequency chord
    coda_start = int(51 * SR)
    coda_len = int(3.5 * SR)
    if coda_start + coda_len <= n:
        t_coda = np.linspace(0, 3.5, coda_len, False)
        chord = (sine(pi_freq, t_coda) * 0.10 +
                 sine(pi_freq * 1.5, t_coda) * 0.06 +  # fifth
                 sine(pi_freq * 2, t_coda) * 0.04)     # octave
        chord *= envelope(coda_len, 0.3, 2.5)
        buf[coda_start:coda_start + coda_len, 0] += chord
        buf[coda_start:coda_start + coda_len, 1] += chord

    return buf


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    print("Phase 29: Stochastic Processes")

    print("\n1. Markov Chain")
    wav = generate_markov()
    write_wav("output/stoch_1_markov_chain.wav", wav)

    print("\n2. Brownian Motion")
    wav = generate_brownian()
    write_wav("output/stoch_2_brownian_motion.wav", wav)

    print("\n3. Monte Carlo Pi")
    wav = generate_monte_carlo()
    write_wav("output/stoch_3_monte_carlo_pi.wav", wav)

    print("\nDone!")
