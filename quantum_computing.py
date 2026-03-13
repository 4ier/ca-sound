#!/usr/bin/env python3
"""Phase 15: Quantum Computing — superposition, entanglement, and interference as sound.

Three pieces:
1. Superposition (55s, stereo) — qubit states as overlapping frequencies,
   gates rotate the Bloch sphere, measurement collapses harmony to monotone
2. Entanglement (50s, stereo) — Bell pairs in correlated stereo channels,
   measuring one instantly constrains the other
3. Quantum Walk (55s, stereo) — quantum vs classical random walk on a line,
   interference patterns create unexpected probability peaks
"""

import numpy as np
import os

SR = 44100

def normalize(x, headroom=0.85):
    peak = np.max(np.abs(x))
    return x * headroom / peak if peak > 0 else x

def write_wav(name, data, sr=SR):
    os.makedirs("output", exist_ok=True)
    path = f"output/{name}.wav"
    import wave
    d = normalize(data)
    with wave.open(path, 'w') as w:
        ch = 2 if d.ndim == 2 else 1
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        if ch == 2:
            interleaved = np.empty(d.shape[1] * 2, dtype=np.float64)
            interleaved[0::2] = d[0]
            interleaved[1::2] = d[1]
            raw = np.clip(interleaved * 32767, -32768, 32767).astype(np.int16)
        else:
            raw = np.clip(d * 32767, -32768, 32767).astype(np.int16)
        w.writeframes(raw.tobytes())
    print(f"  → {path} ({len(data[0] if data.ndim == 2 else data) / sr:.1f}s)")

def fade(n, fade_in=0, fade_out=0, sr=SR):
    env = np.ones(n)
    if fade_in > 0:
        fi = int(fade_in * sr)
        env[:fi] *= np.linspace(0, 1, fi)
    if fade_out > 0:
        fo = int(fade_out * sr)
        env[-fo:] *= np.linspace(1, 0, fo)
    return env


# ── Piece 1: Superposition ─────────────────────────────────────────────────

def superposition():
    """A single qubit's journey through quantum gates.
    
    |0⟩ = 220 Hz (A3), |1⟩ = 330 Hz (E4) — a perfect fifth.
    Superposition = both frequencies, amplitudes = |α|² and |β|².
    
    Sections:
    A (0-12s): Pure |0⟩ — 220 Hz, clean
    B (12-24s): Hadamard → equal superposition, rich beating
    C (24-36s): Phase rotation — timbral evolution, Bloch sphere longitude
    D (36-45s): Gate sequence — amplitudes dance
    E (45-55s): Measurement collapse — noise burst, one frequency survives
    """
    print("Generating Superposition...")
    dur = 55.0
    n = int(dur * SR)
    t = np.arange(n) / SR
    
    f0, f1, drone_f = 220.0, 330.0, 55.0
    
    # Build theta/phi envelopes for Bloch sphere
    theta = np.zeros(n)
    phi = np.zeros(n)
    
    # Section boundaries
    s_a = int(12.0 * SR)
    s_b = int(24.0 * SR)
    s_c = int(36.0 * SR)
    s_d = int(45.0 * SR)
    
    # A: |0⟩
    # theta = 0 (default)
    
    # B: Hadamard gate (2s transition then hold)
    gate_dur = int(2.0 * SR)
    idx_b = np.arange(s_a, s_b)
    progress_b = np.minimum(1.0, (idx_b - s_a) / gate_dur)
    progress_b = 0.5 - 0.5 * np.cos(np.pi * progress_b)
    theta[s_a:s_b] = np.pi / 2 * progress_b
    
    # C: Phase rotation (θ stays π/2, φ sweeps 0→4π)
    theta[s_c:s_d] = np.pi / 2  # also for section C
    theta[s_b:s_c] = np.pi / 2
    idx_c = np.arange(s_b, s_c)
    phi[s_b:s_c] = 4 * np.pi * (idx_c - s_b) / (s_c - s_b)
    
    # D: Gate sequence with smooth interpolation
    gates = [(0.0, np.pi, 0), (2.5, np.pi/2, np.pi/4),
             (5.0, np.pi/4, np.pi/2), (7.0, np.pi/2, np.pi),
             (9.0, np.pi/3, 3*np.pi/4)]
    idx_d = np.arange(s_c, s_d)
    tt_d = (idx_d - s_c) / SR
    for i in range(len(gates) - 1):
        t0, th0, ph0 = gates[i]
        t1, th1, ph1 = gates[i+1]
        mask = (tt_d >= t0) & (tt_d < t1)
        frac = (tt_d[mask] - t0) / (t1 - t0)
        frac = 0.5 - 0.5 * np.cos(np.pi * np.minimum(1.0, frac))
        theta[s_c:s_d][mask] = th0 + (th1 - th0) * frac
        phi[s_c:s_d][mask] = ph0 + (ph1 - ph0) * frac
    # After last gate
    last_mask = tt_d >= gates[-1][0]
    theta[s_c:s_d][last_mask] = gates[-1][1]
    phi[s_c:s_d][last_mask] = gates[-1][2]
    
    # E: Measurement
    final_theta = theta[s_d - 1]
    prob_0 = np.cos(final_theta / 2) ** 2
    np.random.seed(42)
    outcome = 0 if np.random.random() < prob_0 else 1
    
    idx_e = np.arange(s_d, n)
    sec_e = (idx_e - s_d) / SR
    # Hold theta/phi but we'll override amplitudes below
    theta[s_d:] = 0.0 if outcome == 0 else np.pi
    phi[s_d:] = 0.0
    
    # Compute amplitudes
    amp_0 = np.cos(theta / 2)
    amp_1 = np.sin(theta / 2)
    
    # Override for collapse section
    noise_end = s_d + int(0.8 * SR)
    collapse_progress = np.clip((idx_e - s_d) / (0.8 * SR), 0, 1)
    if outcome == 0:
        amp_1[s_d:] = amp_1[s_d - 1] * np.maximum(0, 1.0 - collapse_progress)
        amp_0[s_d:noise_end] = np.linspace(amp_0[s_d-1], 1.0, noise_end - s_d)
        amp_0[noise_end:] = 1.0
    else:
        amp_0[s_d:] = amp_0[s_d - 1] * np.maximum(0, 1.0 - collapse_progress)
        amp_1[s_d:noise_end] = np.linspace(amp_1[s_d-1], 1.0, noise_end - s_d)
        amp_1[noise_end:] = 1.0
    
    # Superposition degree for harmonic richness
    superpos = 2 * amp_0 * amp_1
    
    # Phase accumulations
    phase_0 = np.cumsum(np.full(n, f0 / SR))
    phase_1 = np.cumsum(np.full(n, f1 / SR))
    phase_d = np.cumsum(np.full(n, drone_f / SR))
    
    # Synthesis
    sig_0 = amp_0 * np.sin(2 * np.pi * phase_0)
    sig_1 = amp_1 * np.sin(2 * np.pi * phase_1 + phi)
    harm_0 = amp_0 * 0.3 * superpos * np.sin(4 * np.pi * phase_0)
    harm_1 = amp_1 * 0.2 * superpos * np.sin(4 * np.pi * phase_1 + phi)
    drone = 0.15 * np.sin(2 * np.pi * phase_d)
    
    # Noise burst during collapse
    noise = np.zeros(n)
    if s_d < n:
        noise_env = np.zeros(n)
        noise_samples = min(noise_end, n) - s_d
        noise_env[s_d:s_d + noise_samples] = np.linspace(0.4, 0, noise_samples)
        np.random.seed(123)
        noise = np.random.normal(0, 1, n) * noise_env
    
    signal = 0.5 * (sig_0 + sig_1 + harm_0 + harm_1) + drone + noise * 0.15
    
    # Stereo panning
    pan_l = amp_0 * 0.65 + amp_1 * 0.35 + 0.3
    pan_r = amp_0 * 0.35 + amp_1 * 0.65 + 0.3
    L = signal * pan_l
    R = signal * pan_r
    
    env = fade(n, fade_in=0.5, fade_out=3.0)
    L *= env
    R *= env
    
    write_wav("qc_1_superposition", np.array([L, R]))


# ── Piece 2: Entanglement ──────────────────────────────────────────────────

def entanglement():
    """Bell state: two qubits, perfectly correlated across stereo channels.
    
    L = qubit A, R = qubit B.
    A: Independent → B: CNOT entangles → C: Correlated evolution →
    D: Measurement collapse → Coda: Classical, identical
    """
    print("Generating Entanglement...")
    dur = 50.0
    n = int(dur * SR)
    t = np.arange(n) / SR
    
    f_a, f_b = 220.0, 277.18  # independent frequencies
    f_bell = 220.0             # collapsed Bell state
    drone_f = 55.0
    
    # Shared modulation
    mod = np.sin(2 * np.pi * 0.7 * t)
    mod2 = np.sin(2 * np.pi * 1.9 * t) * 0.3
    
    # Build per-sample parameters via sections
    freq_a = np.full(n, f_a)
    freq_b = np.full(n, f_b)
    env_a = np.full(n, 0.6)
    env_b = np.full(n, 0.6)
    harm_a = np.full(n, 0.15)
    harm_b = np.full(n, 0.15)
    
    s1, s2, s3, s4 = int(10*SR), int(22*SR), int(34*SR), int(42*SR)
    
    # Section A (0-10s): Independent
    idx = np.arange(0, s1)
    env_a[:s1] = 0.6 + 0.4 * np.sin(2 * np.pi * 1.3 * t[:s1])
    env_b[:s1] = 0.6 + 0.4 * np.sin(2 * np.pi * 0.8 * t[:s1] + 1.7)
    
    # Section B (10-22s): Entangling
    idx = np.arange(s1, s2)
    ease = 0.5 - 0.5 * np.cos(np.pi * (idx - s1) / (s2 - s1))
    bell_freq = f_bell + 110 * (0.5 + 0.5 * mod[s1:s2])
    freq_a[s1:s2] = f_a * (1 - ease) + bell_freq * ease
    freq_b[s1:s2] = f_b * (1 - ease) + bell_freq * ease
    ind_a = 0.6 + 0.4 * np.sin(2 * np.pi * 1.3 * t[s1:s2])
    ind_b = 0.6 + 0.4 * np.sin(2 * np.pi * 0.8 * t[s1:s2] + 1.7)
    shared = 0.6 + 0.3 * mod[s1:s2] + 0.1 * mod2[s1:s2]
    env_a[s1:s2] = ind_a * (1 - ease) + shared * ease
    env_b[s1:s2] = ind_b * (1 - ease) + shared * ease
    harm_a[s1:s2] = 0.15 + 0.25 * ease
    harm_b[s1:s2] = 0.15 + 0.25 * ease
    
    # Section C (22-34s): Entangled evolution
    bell_freq_c = f_bell + 110 * (0.5 + 0.5 * mod[s2:s3])
    freq_a[s2:s3] = bell_freq_c
    freq_b[s2:s3] = bell_freq_c
    shared_c = 0.6 + 0.3 * mod[s2:s3] + 0.1 * mod2[s2:s3]
    perturb = 0.15 * np.sin(2 * np.pi * 2.1 * t[s2:s3])
    env_a[s2:s3] = shared_c + perturb
    env_b[s2:s3] = shared_c + perturb
    harm_a[s2:s3] = 0.4
    harm_b[s2:s3] = 0.4
    
    # Section D (34-42s): Measurement collapse
    idx = np.arange(s3, s4)
    progress = (idx - s3) / (s4 - s3)
    # Pre-collapse tension (0-15%), burst (15-25%), post-collapse (25-100%)
    pre = progress < 0.15
    burst = (progress >= 0.15) & (progress < 0.25)
    post = progress >= 0.25
    
    bell_freq_d = f_bell + 110 * (0.5 + 0.5 * mod[s3:s4])
    freq_a[s3:s4] = np.where(pre, bell_freq_d, f_bell)
    freq_b[s3:s4] = np.where(pre, bell_freq_d, f_bell)
    env_a[s3:s4] = np.where(pre, 0.8, 0.7)
    env_b[s3:s4] = np.where(pre, 0.8, 0.7)
    harm_a[s3:s4] = np.where(pre, 0.4 + 0.3 * progress / 0.15,
                     np.where(burst, 0.3, 0.15))
    harm_b[s3:s4] = harm_a[s3:s4]
    
    # Coda (42-50s)
    freq_a[s4:] = f_bell
    freq_b[s4:] = f_bell
    env_a[s4:] = 0.6
    env_b[s4:] = 0.6
    harm_a[s4:] = 0.1
    harm_b[s4:] = 0.1
    
    # Phase accumulation (variable frequency)
    phase_a = np.cumsum(freq_a / SR)
    phase_b = np.cumsum(freq_b / SR)
    phase_d = np.cumsum(np.full(n, drone_f / SR))
    
    # Synthesis
    sig_a = env_a * (np.sin(2*np.pi*phase_a) + 
                     harm_a * np.sin(4*np.pi*phase_a) +
                     harm_a * 0.5 * np.sin(6*np.pi*phase_a))
    sig_b = env_b * (np.sin(2*np.pi*phase_b) +
                     harm_b * np.sin(4*np.pi*phase_b) +
                     harm_b * 0.5 * np.sin(6*np.pi*phase_b))
    drone = 0.12 * np.sin(2 * np.pi * phase_d)
    
    # Noise burst during collapse
    noise = np.zeros(n)
    burst_start = s3 + int(0.15 * (s4 - s3))
    burst_end = s3 + int(0.25 * (s4 - s3))
    burst_len = burst_end - burst_start
    np.random.seed(77)
    noise[burst_start:burst_end] = np.random.normal(0, 0.2, burst_len) * np.linspace(1, 0, burst_len)
    
    L = 0.5 * sig_a + 0.15 * sig_b + drone + noise
    R = 0.15 * sig_a + 0.5 * sig_b + drone + noise
    
    env = fade(n, fade_in=0.5, fade_out=3.0)
    L *= env
    R *= env
    
    write_wav("qc_2_entanglement", np.array([L, R]))


# ── Piece 3: Quantum Walk ──────────────────────────────────────────────────

def quantum_walk():
    """Quantum vs classical random walk on a 1D lattice.
    
    Left = classical walker (Gaussian, diffuse)
    Right = quantum walker (peaked edges, interference fringes)
    
    200 steps, each ~0.275s. Position maps to frequency.
    Classical stays near center; quantum probability races to edges.
    """
    print("Generating Quantum Walk...")
    dur = 55.0
    n = int(dur * SR)
    
    lattice_size = 41  # -20 to +20 (smaller for speed)
    center = 20
    n_steps = 200
    
    # Frequency mapping: log scale across lattice
    base_freq = 110.0
    lattice_freqs = np.array([base_freq * (2.0 ** ((i - center) / 12.0)) 
                              for i in range(lattice_size)])
    
    # ── Classical random walk ──
    classical = np.zeros(lattice_size)
    classical[center] = 1.0
    classical_history = [classical.copy()]
    for _ in range(n_steps):
        new = np.zeros(lattice_size)
        new[1:] += 0.5 * classical[:-1]
        new[:-1] += 0.5 * classical[1:]
        classical = new
        classical_history.append(classical.copy())
    
    # ── Quantum walk (Hadamard coin) ──
    psi = np.zeros((2, lattice_size), dtype=complex)
    psi[0, center] = 1.0 / np.sqrt(2)
    psi[1, center] = 1j / np.sqrt(2)
    quantum_history = [np.abs(psi[0])**2 + np.abs(psi[1])**2]
    
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    for _ in range(n_steps):
        # Coin
        new_psi = np.zeros_like(psi)
        for j in range(lattice_size):
            v = np.array([psi[0, j], psi[1, j]])
            hv = H @ v
            new_psi[0, j] = hv[0]
            new_psi[1, j] = hv[1]
        # Shift
        shifted = np.zeros_like(new_psi)
        shifted[0, :-1] += new_psi[0, 1:]   # |0⟩ left
        shifted[1, 1:] += new_psi[1, :-1]    # |1⟩ right
        psi = shifted
        quantum_history.append(np.abs(psi[0])**2 + np.abs(psi[1])**2)
    
    # ── Sonify using chunked vectorized synthesis ──
    samples_per_step = n // n_steps
    L = np.zeros(n)
    R = np.zeros(n)
    
    # Pre-compute wavetables for each lattice frequency
    # Use additive synthesis per chunk
    drone_f = 55.0
    
    # Active positions (only sound significant probabilities)
    threshold = 0.003
    
    for step in range(n_steps):
        s0 = step * samples_per_step
        s1 = min(s0 + samples_per_step, n)
        chunk_len = s1 - s0
        if chunk_len <= 0:
            break
        
        c_prob = classical_history[min(step, len(classical_history)-1)]
        q_prob = quantum_history[min(step, len(quantum_history)-1)]
        c_sum = np.sum(c_prob)
        q_sum = np.sum(q_prob)
        if c_sum > 0: c_prob = c_prob / c_sum
        if q_sum > 0: q_prob = q_prob / q_sum
        
        chunk_t = np.arange(chunk_len) / SR + s0 / SR
        c_chunk = np.zeros(chunk_len)
        q_chunk = np.zeros(chunk_len)
        
        # Find active positions for this step
        for j in range(lattice_size):
            freq = lattice_freqs[j]
            phase_base = freq * chunk_t
            
            if c_prob[j] > threshold:
                amp = c_prob[j] ** 0.5 * 0.6
                c_chunk += amp * np.sin(2 * np.pi * phase_base)
            
            if q_prob[j] > threshold:
                amp = q_prob[j] ** 0.5 * 0.6
                q_chunk += (amp * (np.sin(2 * np.pi * phase_base) +
                            0.3 * np.sin(4 * np.pi * phase_base) +
                            0.12 * np.sin(6 * np.pi * phase_base)))
        
        # Drone
        drone_chunk = 0.1 * np.sin(2 * np.pi * drone_f * chunk_t)
        
        # Crossfeed: 80% own side, 20% cross
        L[s0:s1] = 0.8 * c_chunk + 0.2 * q_chunk + drone_chunk
        R[s0:s1] = 0.2 * c_chunk + 0.8 * q_chunk + drone_chunk
    
    env = fade(n, fade_in=0.5, fade_out=3.0)
    L *= env
    R *= env
    
    write_wav("qc_3_quantum_walk", np.array([L, R]))


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Phase 15: Quantum Computing")
    print("=" * 50)
    superposition()
    entanglement()
    quantum_walk()
    print("\nDone.")
