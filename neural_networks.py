#!/usr/bin/env python3
"""Phase 16: Neural Networks — gradient flow, activation landscapes, weight space.

Three pieces:
1. Backpropagation (55s, stereo) — Forward pass builds harmonics layer by layer,
   backward pass sends gradient echoes in reverse. Loss decreasing = dissonance resolving.
2. Activation Functions (50s, stereo) — The same input signal transformed by sigmoid,
   ReLU, tanh, softmax. Each activation is a different timbral filter.
3. Weight Space (55s, stereo) — Training trajectory through a loss landscape.
   Random init = noise cloud, saddle points = unstable harmonics, convergence = crystal chord.
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
    """Crossfade between a and b by alpha (0=a, 1=b)."""
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
# 1. Backpropagation
# ─────────────────────────────────────────────────
def backpropagation():
    """Forward pass builds harmonics layer by layer (4 layers, low→high),
    backward pass sends gradient echoes in reverse (high→low).
    Loss decreasing = dissonance interval narrowing to consonance."""
    dur = 55.0
    n = int(dur * SR)
    t = np.linspace(0, dur, n, endpoint=False)
    out_l = np.zeros(n)
    out_r = np.zeros(n)

    base_freq = 110.0  # A2
    layers = [
        (base_freq,       1.0),   # Input layer — fundamental
        (base_freq * 2,   0.7),   # Hidden 1 — octave
        (base_freq * 3,   0.5),   # Hidden 2 — fifth above octave
        (base_freq * 4,   0.35),  # Output layer — double octave
    ]

    # Phase 1: Forward pass (0-20s) — layers activate sequentially
    # Each layer adds its harmonic, building up timbre
    for i, (freq, amp) in enumerate(layers):
        onset = i * 4.5  # staggered entry
        fade_in = 2.0
        for j in range(n):
            time = j / SR
            if time < onset:
                continue
            progress = min(1.0, (time - onset) / fade_in)
            # Forward activation: smooth sigmoid-like ramp
            activation = 1.0 / (1.0 + np.exp(-8 * (progress - 0.5)))
            sig = sine(freq, time) * amp * activation
            # Slight stereo spread: lower layers left, higher right
            pan = i / (len(layers) - 1)  # 0=left, 1=right
            out_l[j] += sig * (1 - pan * 0.6)
            out_r[j] += sig * (pan * 0.6 + 0.4)

    # Phase 2: Loss signal (15-40s) — dissonance that gradually resolves
    # Start with tritone beating, converge to perfect fifth
    loss_start = 15.0
    loss_end = 40.0
    for j in range(n):
        time = j / SR
        if time < loss_start or time > loss_end:
            continue
        progress = (time - loss_start) / (loss_end - loss_start)
        # Tritone interval (diminished 5th) narrowing to perfect 5th
        # Start: 110 vs 155.56 (tritone), end: 110 vs 165 (perfect 5th)
        target_ratio = 1.5  # perfect fifth
        start_ratio = np.sqrt(2)  # tritone
        ratio = start_ratio + (target_ratio - start_ratio) * progress ** 1.5
        f1, f2 = 220.0, 220.0 * ratio
        # Loss amplitude decreases as it resolves
        loss_amp = 0.15 * (1 - 0.7 * progress)
        sig1 = sine(f1, time) * loss_amp
        sig2 = sine(f2, time) * loss_amp
        out_l[j] += sig1
        out_r[j] += sig2

    # Phase 3: Backward pass (25-50s) — gradient echoes in reverse order
    # Each layer gets a brief FM burst (gradient signal), high layers first
    for i in range(len(layers) - 1, -1, -1):
        layer_idx = len(layers) - 1 - i  # reverse order
        onset = 25.0 + layer_idx * 5.5
        freq, amp = layers[i]
        grad_dur = 4.0
        for j in range(n):
            time = j / SR
            if time < onset or time > onset + grad_dur:
                continue
            local_t = (time - onset) / grad_dur
            # Gradient as FM modulated echo — deeper modulation for early layers
            mod_d = 3.0 * (1 - local_t) * (i + 1) / len(layers)
            env_val = np.sin(np.pi * local_t) ** 0.5  # smooth bell
            grad_sig = fm_tone(freq * 0.5, freq * 0.25, mod_d, time) * amp * 0.3 * env_val
            # Gradient flows opposite direction in stereo
            pan = 1.0 - i / (len(layers) - 1)
            out_l[j] += grad_sig * (1 - pan * 0.6)
            out_r[j] += grad_sig * (pan * 0.6 + 0.4)

    # Phase 4: Convergence (45-55s) — all layers in tune, rich A major chord
    conv_start, conv_end = 45.0, dur
    chord_freqs = [110, 220, 277.18, 330, 440]  # A major with octave spread
    for j in range(n):
        time = j / SR
        if time < conv_start:
            continue
        progress = min(1.0, (time - conv_start) / 3.0)
        fade_out = max(0, 1 - (time - (conv_end - 4.0)) / 4.0) if time > conv_end - 4.0 else 1.0
        for k, cf in enumerate(chord_freqs):
            amp = 0.12 * progress * fade_out / (1 + k * 0.3)
            sig = sine(cf, time) * amp
            pan = k / (len(chord_freqs) - 1)
            out_l[j] += sig * (1 - pan * 0.5)
            out_r[j] += sig * (0.5 + pan * 0.5)

    # Drone: continuous low A1 (55 Hz)
    drone = sine(55, t) * 0.06
    out_l += drone
    out_r += drone

    out = np.stack([out_l, out_r], axis=1)
    env = envelope(n, attack=0.5, release=2.0)
    out[:, 0] *= env
    out[:, 1] *= env
    out /= max(np.max(np.abs(out)), 1e-6) * 1.15
    return out


# ─────────────────────────────────────────────────
# 2. Activation Functions
# ─────────────────────────────────────────────────
def activation_functions():
    """Same input signal through four activation functions:
    sigmoid (warm compression), ReLU (harsh clipping),
    tanh (saturated), softmax (probabilistic blend).
    Each section ~12s with crossfade transitions."""
    dur = 50.0
    n = int(dur * SR)
    t = np.linspace(0, dur, n, endpoint=False)
    out_l = np.zeros(n)
    out_r = np.zeros(n)

    # Base input: rich signal with multiple harmonics
    base_freq = 165.0  # E3
    def input_signal(time):
        s = 0.0
        for h in range(1, 8):
            s += sine(base_freq * h, time, phase=h * 0.3) / h
        # Add slow LFO modulation
        s *= (1 + 0.3 * sine(0.8, time))
        return s

    sections = [
        ("sigmoid", 0.0, 12.0),
        ("relu",    11.0, 24.0),
        ("tanh",    23.0, 37.0),
        ("softmax", 36.0, 50.0),
    ]

    for name, start, end in sections:
        for j in range(n):
            time = j / SR
            if time < start or time > end:
                continue
            # Fade in/out for crossfade
            fade_in = min(1.0, (time - start) / 1.5) if time < start + 1.5 else 1.0
            fade_out = min(1.0, (end - time) / 1.5) if time > end - 1.5 else 1.0
            fade = fade_in * fade_out

            sig = input_signal(time)

            if name == "sigmoid":
                # Warm compression — squashed into [0,1], mapped to [-1,1]
                activated = 2.0 / (1.0 + np.exp(-3 * sig)) - 1.0
                # Sigmoid adds warmth: slight even harmonics
                activated += 0.1 * sine(base_freq * 2, time) * fade
                pan_l, pan_r = 0.7, 0.5  # slightly left

            elif name == "relu":
                # Hard clipping — zeroes negative, keeps positive
                activated = max(0, sig)
                # ReLU creates sharp edges: add high-freq buzz on positive
                if sig > 0:
                    activated += 0.08 * sine(base_freq * 7, time) * sig
                pan_l, pan_r = 0.5, 0.7  # slightly right

            elif name == "tanh":
                # Saturated S-curve — natural compression
                activated = np.tanh(2 * sig)
                # Tanh saturates richly: emphasize odd harmonics
                activated += 0.06 * sine(base_freq * 3, time) * np.tanh(sig)
                activated += 0.04 * sine(base_freq * 5, time) * np.tanh(sig)
                pan_l, pan_r = 0.6, 0.6  # center

            else:  # softmax
                # Probabilistic: multiple competing frequencies
                freqs = [base_freq, base_freq * 1.5, base_freq * 2, base_freq * 2.5]
                logits = [np.sin(2 * np.pi * f * time + f * 0.01) for f in freqs]
                exp_l = [np.exp(3 * l) for l in logits]
                total = sum(exp_l) + 1e-8
                probs = [e / total for e in exp_l]
                activated = sum(p * sine(f, time) for p, f in zip(probs, freqs))
                pan_l, pan_r = 0.55, 0.55

            out_l[j] += activated * 0.4 * fade * pan_l
            out_r[j] += activated * 0.4 * fade * pan_r

    # Subtle drone
    drone = sine(82.41, t) * 0.04  # E2
    out_l += drone
    out_r += drone

    out = np.stack([out_l, out_r], axis=1)
    env = envelope(n, attack=0.3, release=1.5)
    out[:, 0] *= env
    out[:, 1] *= env
    out /= max(np.max(np.abs(out)), 1e-6) * 1.15
    return out


# ─────────────────────────────────────────────────
# 3. Weight Space
# ─────────────────────────────────────────────────
def weight_space():
    """Training trajectory through a loss landscape.
    Random init (noise), gradient steps (wandering pitch),
    saddle points (unstable beating), local minima (sustained dissonant chords),
    convergence (crystalline D minor chord)."""
    dur = 55.0
    n = int(dur * SR)
    t = np.linspace(0, dur, n, endpoint=False)
    out_l = np.zeros(n)
    out_r = np.zeros(n)
    rng = np.random.default_rng(42)

    # Simulate 2D loss landscape traversal
    # 200 training steps, each mapped to ~0.25s of audio
    n_steps = 200
    step_dur = dur / n_steps

    # Generate a random walk that eventually converges
    # Start far from optimum, end at D minor target
    target = np.array([293.66, 349.23, 440.0])  # D4, F4, A4 (D minor)

    # Initial random frequencies
    freqs = rng.uniform(200, 600, size=(n_steps + 1, 3))

    # Training trajectory: noisy at first, converging later
    for step in range(1, n_steps + 1):
        progress = step / n_steps
        # Learning rate decreases
        lr = 0.15 * (1 - progress) ** 0.7
        # Gradient toward target + noise
        noise_scale = 80 * (1 - progress) ** 1.5
        grad = (target - freqs[step - 1]) * lr
        noise = rng.normal(0, noise_scale, 3)
        freqs[step] = freqs[step - 1] + grad + noise

        # Saddle point around step 60-80: frequencies get stuck oscillating
        if 55 < step < 85:
            freqs[step] += 15 * np.sin(step * 0.5) * np.array([1, -1, 0.5])

        # Local minimum around step 110-130: stuck on wrong chord
        if 105 < step < 135:
            wrong_target = np.array([261.63, 329.63, 392.0])  # C major (wrong)
            pull = (wrong_target - freqs[step]) * 0.3
            freqs[step] += pull

    # Loss values (distance to target)
    losses = np.array([np.linalg.norm(f - target) for f in freqs])
    max_loss = np.max(losses)

    # Render audio
    for step in range(n_steps):
        sample_start = int(step * step_dur * SR)
        sample_end = int((step + 1) * step_dur * SR)
        if sample_end > n:
            sample_end = n
        step_n = sample_end - sample_start
        if step_n <= 0:
            continue

        progress = step / n_steps
        f = freqs[step]
        loss = losses[step] / max(max_loss, 1e-6)

        step_t = np.linspace(0, step_dur, step_n, endpoint=False)

        # Three voices for three weights
        for wi in range(3):
            # Harmonic richness proportional to loss (noisy when far, pure when close)
            n_harmonics = max(1, int(1 + 6 * loss))
            sig = np.zeros(step_n)
            for h in range(1, n_harmonics + 1):
                harm_amp = 1.0 / (h * (1 + loss * 2))
                sig += sine(f[wi] * h, step_t + step * step_dur) * harm_amp

            # FM noise proportional to loss
            if loss > 0.3:
                noise_mod = loss * 2.5
                sig += fm_tone(f[wi] * 0.5, f[wi] * 0.37, noise_mod, step_t + step * step_dur) * 0.15 * loss

            # Amplitude
            amp = 0.2 / (1 + wi * 0.2)
            sig *= amp

            # Stereo: weight 0 left, weight 1 center, weight 2 right
            pan = wi / 2.0
            out_l[sample_start:sample_end] += sig * (1 - pan * 0.7)
            out_r[sample_start:sample_end] += sig * (0.3 + pan * 0.7)

        # Beat/click on large gradient steps
        grad_mag = np.linalg.norm(freqs[min(step + 1, n_steps)] - f)
        if grad_mag > 30 and step_n > 100:
            click_n = min(int(0.008 * SR), step_n)
            click = rng.normal(0, 0.08, click_n) * np.linspace(1, 0, click_n) ** 2
            out_l[sample_start:sample_start + click_n] += click
            out_r[sample_start:sample_start + click_n] += click

    # Convergence coda (last 8s): crystalline D minor holds
    coda_start = dur - 8.0
    for j in range(n):
        time = j / SR
        if time < coda_start:
            continue
        progress = (time - coda_start) / 8.0
        fade = 1.0 - max(0, (progress - 0.6) / 0.4) ** 2  # fade out last 40%
        for k, cf in enumerate(target):
            amp = 0.15 * fade / (1 + k * 0.15)
            sig = sine(cf, time) * amp
            # Add gentle vibrato
            sig *= (1 + 0.02 * sine(4.5, time + k))
            pan = k / 2.0
            out_l[j] += sig * (1 - pan * 0.5) * 0.5  # blend with trajectory
            out_r[j] += sig * (0.5 + pan * 0.5) * 0.5

    # D2 drone throughout
    drone = sine(73.42, t) * 0.05
    out_l += drone
    out_r += drone

    out = np.stack([out_l, out_r], axis=1)
    env = envelope(n, attack=0.5, release=2.5)
    out[:, 0] *= env
    out[:, 1] *= env
    out /= max(np.max(np.abs(out)), 1e-6) * 1.15
    return out


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    pieces = [
        ("nn_1_backpropagation", backpropagation),
        ("nn_2_activation_functions", activation_functions),
        ("nn_3_weight_space", weight_space),
    ]
    for name, fn in pieces:
        print(f"Generating {name}...")
        audio = fn()
        write_wav(f"output/{name}.wav", audio)
    print("Done — Neural Networks triptych complete.")
