#!/usr/bin/env python3
"""Phase 28: Chaos Theory -- Lorenz attractor, double pendulum, bifurcation.

Three pieces:
1. Lorenz Attractor (55s, stereo) -- The iconic butterfly. x/y/z coordinates of
   the Lorenz system map to frequency, stereo position, and harmonic richness.
   Two trajectories start epsilon apart (sensitive dependence on initial
   conditions): initially in unison, they diverge into independent melodies.
   The two lobes of the attractor create an alternating pattern between two
   pitch centers. 55Hz drone represents the strange attractor basin.

2. Double Pendulum (50s, stereo) -- Two coupled oscillators. The first pendulum
   (left channel, low frequency) drives the second (right channel, high
   frequency). Phase 1: small angles, nearly periodic motion, consonant
   intervals. Phase 2: energy increases, onset of chaos, intervals become
   dissonant and unpredictable. Phase 3: fully chaotic, both channels wildly
   independent. Angular velocity maps to brightness, angle to pitch.

3. Bifurcation Cascade (55s, stereo) -- The logistic map r parameter sweeps from
   2.5 to 4.0 over the full duration. Period-1 = single sustained tone.
   Period-2 = two alternating pitches (perfect fifth). Period-4 = four-note
   pattern. At r~3.57 onset of chaos = dense frequency cluster. Periodic
   windows emerge briefly as consonant islands. Final r=4.0 = full noise.
   Left channel = current x value as pitch, right = delayed x (phase portrait).
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

def generate_lorenz():
    """Lorenz Attractor: two diverging trajectories on the butterfly."""
    duration = 55.0
    n = int(SR * duration)
    buf = np.zeros((n, 2))
    t_full = np.linspace(0, duration, n, False)

    # 55Hz drone
    drone = sine(55, t_full) * 0.05 * envelope(n, 2.0, 3.0)
    buf[:, 0] += drone
    buf[:, 1] += drone

    # Lorenz system parameters (classic)
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    dt_sim = 0.005
    steps = int(duration * 200)  # 200 simulation steps per second of audio

    # Two trajectories, epsilon apart
    state_a = np.array([1.0, 1.0, 1.0])
    state_b = np.array([1.0, 1.0, 1.0001])  # tiny perturbation

    traj_a = np.zeros((steps, 3))
    traj_b = np.zeros((steps, 3))

    for i in range(steps):
        traj_a[i] = state_a
        traj_b[i] = state_b
        # RK4 for trajectory A
        def lorenz_deriv(s):
            return np.array([sigma * (s[1] - s[0]),
                             s[0] * (rho - s[2]) - s[1],
                             s[0] * s[1] - beta * s[2]])
        for state in [state_a, state_b]:
            k1 = lorenz_deriv(state)
            k2 = lorenz_deriv(state + 0.5 * dt_sim * k1)
            k3 = lorenz_deriv(state + 0.5 * dt_sim * k2)
            k4 = lorenz_deriv(state + dt_sim * k3)
            delta = (dt_sim / 6) * (k1 + 2*k2 + 2*k3 + k4)
            if state is state_a:
                state_a = state_a + delta
            else:
                state_b = state_b + delta

    # Map trajectories to audio
    # x range approx [-20, 20] -> frequency 110-880Hz (log)
    # y range -> stereo pan
    # z range [0, 50] -> harmonic richness 1-6

    samples_per_step = n / steps
    grain_len = int(0.02 * SR)  # 20ms grains

    for traj, channel_bias in [(traj_a, 0.3), (traj_b, 0.7)]:
        for i in range(0, steps, 2):  # every other step for performance
            s = int(i * samples_per_step)
            if s + grain_len > n:
                break
            x, y, z = traj[i]
            # x -> freq (log map from [-20,20] to [110,880])
            x_norm = np.clip((x + 20) / 40, 0, 1)
            freq = 110 * (880/110) ** x_norm
            # y -> pan
            y_norm = np.clip((y + 30) / 60, 0, 1)
            pan = 0.2 + 0.6 * y_norm  # stay within 0.2-0.8
            # z -> harmonics
            z_norm = np.clip(z / 50, 0, 1)
            num_h = max(1, int(z_norm * 5) + 1)

            t_grain = np.linspace(0, 0.02, grain_len, False)
            grain = np.zeros(grain_len)
            for h in range(1, num_h + 1):
                grain += sine(freq * h, t_grain) * (0.06 / h)
            grain *= envelope(grain_len, 0.002, 0.01)
            mix_to(buf, grain, s, pan * channel_bias / 0.5)

    # Divergence indicator: when trajectories diverge, add FM burst
    for i in range(0, steps, 100):
        dist = np.linalg.norm(traj_a[i] - traj_b[i])
        if dist > 5:
            s = int(i * samples_per_step)
            burst_len = int(0.05 * SR)
            if s + burst_len > n:
                break
            t_burst = np.linspace(0, 0.05, burst_len, False)
            burst = fm_tone(440, 5, dist * 0.3, t_burst) * envelope(burst_len, 0.002, 0.03) * 0.04
            mix_to(buf, burst, s, 0.5)

    # Coda: both trajectories fade, drone resolves to D major
    coda_start = int(51 * SR)
    coda_len = int(3.5 * SR)
    if coda_start + coda_len <= n:
        t_coda = np.linspace(0, 3.5, coda_len, False)
        chord = (sine(146.83, t_coda) + sine(185.0, t_coda) * 0.7 +
                 sine(220.0, t_coda) * 0.8) * 0.07
        chord *= envelope(coda_len, 0.3, 2.5)
        buf[coda_start:coda_start + coda_len, 0] += chord
        buf[coda_start:coda_start + coda_len, 1] += chord

    return buf

def generate_double_pendulum():
    """Double Pendulum: order to chaos transition."""
    duration = 50.0
    n = int(SR * duration)
    buf = np.zeros((n, 2))
    t_full = np.linspace(0, duration, n, False)

    # 55Hz drone
    drone = sine(55, t_full) * 0.05 * envelope(n, 1.5, 2.5)
    buf[:, 0] += drone
    buf[:, 1] += drone

    # Double pendulum simulation
    # theta1, omega1, theta2, omega2
    g = 9.81
    L1, L2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0
    dt_sim = 0.002
    steps = int(duration * 500)

    # Start with small angles (nearly periodic)
    # Energy increases over time by adding small kicks
    theta1, omega1 = 0.3, 0.0
    theta2, omega2 = 0.3, 0.0

    traj = np.zeros((steps, 4))  # theta1, omega1, theta2, omega2

    for i in range(steps):
        traj[i] = [theta1, omega1, theta2, omega2]

        # Equations of motion (simplified)
        delta = theta1 - theta2
        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) * np.cos(delta)
        den2 = (L2 / L1) * den1

        alpha1 = (m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta) +
                  m2 * g * np.sin(theta2) * np.cos(delta) +
                  m2 * L2 * omega2**2 * np.sin(delta) -
                  (m1 + m2) * g * np.sin(theta1)) / den1

        alpha2 = (-m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta) +
                  (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
                  (m1 + m2) * L1 * omega1**2 * np.sin(delta) -
                  (m1 + m2) * g * np.sin(theta2)) / den2

        omega1 += alpha1 * dt_sim
        omega2 += alpha2 * dt_sim
        theta1 += omega1 * dt_sim
        theta2 += omega2 * dt_sim

        # Energy injection at specific times to push into chaos
        sim_time = i * dt_sim
        if abs(sim_time - 15.0) < dt_sim:
            omega1 += 2.0
        if abs(sim_time - 25.0) < dt_sim:
            omega2 += 3.0

    # Map to audio: grains every ~4ms
    samples_per_step = n / steps
    grain_len = int(0.015 * SR)

    for i in range(0, steps, 4):
        s = int(i * samples_per_step)
        if s + grain_len > n:
            break
        th1, w1, th2, w2 = traj[i]

        # Pendulum 1 (left channel): angle -> pitch
        # theta1 range roughly [-pi, pi] -> 110-440Hz
        freq1 = 220 + 110 * np.sin(th1)
        # Angular velocity -> brightness (harmonics)
        bright1 = min(int(abs(w1) * 0.5) + 1, 6)

        t_grain = np.linspace(0, 0.015, grain_len, False)
        grain1 = np.zeros(grain_len)
        for h in range(1, bright1 + 1):
            grain1 += sine(freq1 * h, t_grain) * (0.08 / h)
        grain1 *= envelope(grain_len, 0.001, 0.008)

        # Pendulum 2 (right channel): higher octave
        freq2 = 440 + 220 * np.sin(th2)
        bright2 = min(int(abs(w2) * 0.5) + 1, 6)

        grain2 = np.zeros(grain_len)
        for h in range(1, bright2 + 1):
            grain2 += sine(freq2 * h, t_grain) * (0.08 / h)
        grain2 *= envelope(grain_len, 0.001, 0.008)

        mix_to(buf, grain1, s, 0.25)
        mix_to(buf, grain2, s, 0.75)

    # Coda: both pendulums settle (friction implied), D minor chord
    coda_start = int(46.5 * SR)
    coda_len = int(3 * SR)
    if coda_start + coda_len <= n:
        t_coda = np.linspace(0, 3, coda_len, False)
        chord = (sine(293.66, t_coda) + sine(349.23, t_coda) * 0.8 +
                 sine(440.0, t_coda) * 0.7) * 0.07
        chord *= envelope(coda_len, 0.3, 2.0)
        buf[coda_start:coda_start + coda_len, 0] += chord
        buf[coda_start:coda_start + coda_len, 1] += chord

    return buf

def generate_bifurcation():
    """Bifurcation Cascade: logistic map r=2.5 to 4.0, period doubling to chaos."""
    duration = 55.0
    n = int(SR * duration)
    buf = np.zeros((n, 2))
    t_full = np.linspace(0, duration, n, False)

    # 55Hz drone
    drone = sine(55, t_full) * 0.04 * envelope(n, 1.5, 3.0)
    buf[:, 0] += drone
    buf[:, 1] += drone

    # Sweep r from 2.5 to 4.0 over duration
    num_segments = 500
    seg_dur = duration / num_segments
    seg_samples = int(seg_dur * SR)

    x = 0.4  # initial condition
    x_prev = x

    for seg in range(num_segments):
        r = 2.5 + (4.0 - 2.5) * (seg / num_segments)
        t_seg = seg * seg_dur
        s = int(t_seg * SR)

        # Iterate logistic map several times to settle, then use values
        # Transient removal
        for _ in range(50):
            x = r * x * (1 - x)

        # Collect period values
        period_vals = []
        for _ in range(32):
            x = r * x * (1 - x)
            period_vals.append(x)

        # Map x values to frequencies: x in [0,1] -> 110-880Hz
        freqs = [110 * (880/110) ** v for v in period_vals]

        # Unique frequencies (within tolerance) determine the "period"
        unique_f = []
        for f in freqs:
            if not any(abs(f - uf) < 5 for uf in unique_f):
                unique_f.append(f)

        # Generate audio for this segment
        grain_len = min(seg_samples, int(0.1 * SR))
        if s + grain_len > n:
            break
        t_grain = np.linspace(0, grain_len / SR, grain_len, False)

        grain_l = np.zeros(grain_len)
        grain_r = np.zeros(grain_len)

        # Left channel: current x as pitch
        for idx, f in enumerate(unique_f[:8]):
            amp = 0.08 / max(len(unique_f), 1)
            # More unique freqs = more dissonant texture
            if len(unique_f) <= 2:
                # Period 1-2: clean sine
                grain_l += sine(f, t_grain) * amp
            elif len(unique_f) <= 4:
                # Period 4: add 2nd harmonic
                grain_l += (sine(f, t_grain) + sine(f * 2, t_grain) * 0.3) * amp
            else:
                # Chaos: FM synthesis for density
                grain_l += fm_tone(f, f * 0.5, 1.5, t_grain) * amp

        # Right channel: delayed x (one iteration behind)
        x_delayed = period_vals[-2] if len(period_vals) > 1 else period_vals[0]
        freq_r = 110 * (880/110) ** x_delayed
        if len(unique_f) <= 2:
            grain_r += sine(freq_r, t_grain) * 0.1
        else:
            grain_r += fm_tone(freq_r, freq_r * 0.3, len(unique_f) * 0.2, t_grain) * 0.08

        env = envelope(grain_len, 0.005, 0.02)
        grain_l *= env
        grain_r *= env

        end = min(s + grain_len, n)
        buf[s:end, 0] += grain_l[:end - s]
        buf[s:end, 1] += grain_r[:end - s]

        # Period-doubling bifurcation points: mark with click
        # Approximate r values: 3.0, 3.449, 3.544, 3.564...
        for r_bif in [3.0, 3.449, 3.544, 3.564]:
            if abs(r - r_bif) < 0.005:
                ck = click(1500, 0.01)
                if s + len(ck) <= n:
                    mix_to(buf, ck, s, 0.5)

    # Periodic windows: brief consonance in chaos (r~3.83 period-3 window)
    # Already naturally emerges from the simulation

    # Coda: r=4 noise fades, resolve to A major
    coda_start = int(51.5 * SR)
    coda_len = int(3 * SR)
    if coda_start + coda_len <= n:
        t_coda = np.linspace(0, 3, coda_len, False)
        chord = (sine(220, t_coda) + sine(277.18, t_coda) * 0.8 +
                 sine(329.63, t_coda) * 0.7) * 0.07
        chord *= envelope(coda_len, 0.5, 2.0)
        buf[coda_start:coda_start + coda_len, 0] += chord
        buf[coda_start:coda_start + coda_len, 1] += chord

    return buf

def main():
    os.makedirs("output", exist_ok=True)
    print("Phase 28: Chaos Theory")
    print("=" * 40)

    print("\n1. Lorenz Attractor")
    buf = generate_lorenz()
    write_wav("output/chaos_1_lorenz_attractor.wav", buf)

    print("\n2. Double Pendulum")
    buf = generate_double_pendulum()
    write_wav("output/chaos_2_double_pendulum.wav", buf)

    print("\n3. Bifurcation Cascade")
    buf = generate_bifurcation()
    write_wav("output/chaos_3_bifurcation_cascade.wav", buf)

    print("\nDone!")


if __name__ == "__main__":
    main()
