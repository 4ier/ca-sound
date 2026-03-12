#!/usr/bin/env python3
"""Phase 14: Optimization — the search for minima as sound.

Three pieces:
1. Gradient Descent (55s, stereo) — particles descending a Rastrigin landscape
2. Simulated Annealing (50s, stereo) — temperature cooling from chaos to crystal
3. Genetic Algorithm (55s, stereo) — population evolving toward a target melody
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
    import wave, struct
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

# ── Piece 1: Gradient Descent ──────────────────────────────────────────────

def rastrigin_2d(x, y, A=10):
    """Rastrigin function — many local minima, one global at (0,0)."""
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

def gradient_rastrigin(x, y, A=10):
    """Gradient of Rastrigin."""
    dx = 2 * x + A * 2 * np.pi * np.sin(2 * np.pi * x)
    dy = 2 * y + A * 2 * np.pi * np.sin(2 * np.pi * y)
    return dx, dy

def gradient_descent():
    """8 particles descend a Rastrigin landscape simultaneously.
    
    Position → frequency (log mapped)
    Gradient magnitude → harmonic brightness
    Loss value → stereo position (low loss = center, high = wide)
    Particles that find near-global minimum → pure consonant tone
    Trapped particles → sustained beating/dissonance
    """
    print("Generating Gradient Descent...")
    dur = 55.0
    n = int(dur * SR)
    L = np.zeros(n)
    R = np.zeros(n)
    
    np.random.seed(42)
    n_particles = 8
    # Start positions spread across landscape
    starts = np.random.uniform(-4.0, 4.0, (n_particles, 2))
    
    lr = 0.003  # learning rate
    n_steps = 2000
    step_dur = dur / n_steps
    
    # Pre-compute all trajectories
    trajectories = []
    for p in range(n_particles):
        traj = [(starts[p, 0], starts[p, 1])]
        x, y = starts[p, 0], starts[p, 1]
        for s in range(n_steps):
            dx, dy = gradient_rastrigin(x, y)
            # Add tiny noise for realism
            x -= lr * dx + np.random.normal(0, 0.001)
            y -= lr * dy + np.random.normal(0, 0.001)
            x = np.clip(x, -5, 5)
            y = np.clip(y, -5, 5)
            traj.append((x, y))
        trajectories.append(traj)
    
    # Base frequencies for particles (pentatonic-ish spread)
    base_freqs = [110, 146.83, 164.81, 196, 220, 261.63, 293.66, 329.63]
    
    # Synthesize each particle
    for p in range(n_particles):
        traj = trajectories[p]
        for s in range(n_steps):
            x0, y0 = traj[s]
            x1, y1 = traj[s + 1]
            
            loss = rastrigin_2d(x0, y0)
            grad_mag = np.sqrt(gradient_rastrigin(x0, y0)[0]**2 + gradient_rastrigin(x0, y0)[1]**2)
            
            # Frequency: base + position offset
            freq = base_freqs[p] * (1 + 0.15 * x0)
            freq = np.clip(freq, 55, 880)
            
            # Harmonic count: high gradient = bright, low = pure
            n_harmonics = max(1, min(8, int(1 + grad_mag * 0.3)))
            
            # Amplitude: inversely proportional to loss (low loss = louder)
            amp = 0.08 / (1 + loss * 0.05)
            
            # Stereo: x position maps to pan
            pan = np.clip(0.5 + x0 * 0.1, 0.1, 0.9)
            
            # Synthesize this step
            t0 = int(s * step_dur * SR)
            t1 = min(int((s + 1) * step_dur * SR), n)
            if t1 <= t0:
                continue
            
            t = np.arange(t1 - t0) / SR
            sig = np.zeros(len(t))
            for h in range(1, n_harmonics + 1):
                decay = 1.0 / h**1.2
                sig += decay * np.sin(2 * np.pi * freq * h * t)
            sig *= amp
            
            # Apply fade at step boundaries
            if len(sig) > 100:
                sig[:50] *= np.linspace(0, 1, 50)
                sig[-50:] *= np.linspace(1, 0, 50)
            
            L[t0:t1] += sig * (1 - pan)
            R[t0:t1] += sig * pan
    
    # Low drone representing the loss landscape itself (55 Hz)
    t_full = np.arange(n) / SR
    drone = 0.06 * np.sin(2 * np.pi * 55 * t_full) * fade(n, 2.0, 3.0)
    L += drone
    R += drone
    
    # Convergence resolution: particles near (0,0) get a consonant chord at end
    # Last 5 seconds: check which particles converged
    resolved = []
    for p, traj in enumerate(trajectories):
        final_x, final_y = traj[-1]
        final_loss = rastrigin_2d(final_x, final_y)
        if final_loss < 1.0:  # near global minimum
            resolved.append(p)
    
    if resolved:
        coda_start = int((dur - 5.0) * SR)
        coda_t = np.arange(n - coda_start) / SR
        coda_env = fade(len(coda_t), 1.5, 2.0)
        coda_chord = np.zeros(len(coda_t))
        # D major chord: D2, A2, D3, F#3
        for f in [73.42, 110, 146.83, 185]:
            coda_chord += 0.04 * np.sin(2 * np.pi * f * coda_t)
        coda_chord *= coda_env
        L[coda_start:] += coda_chord
        R[coda_start:] += coda_chord
    
    stereo = np.array([L, R])
    stereo *= fade(n, 1.0, 2.0)
    write_wav("opt_1_gradient_descent", stereo)

# ── Piece 2: Simulated Annealing ──────────────────────────────────────────

def simulated_annealing():
    """Temperature cooling from chaos to crystal.
    
    High T: wild jumps, energetic noise, wide frequency range
    Cooling: jumps narrow, occasional uphill leaps become rarer
    Low T: tight oscillation around optimum, near-pure tone
    
    The Boltzmann acceptance criterion creates dramatic moments:
    accepted uphill moves = sudden frequency jumps against the trend.
    """
    print("Generating Simulated Annealing...")
    dur = 50.0
    n = int(dur * SR)
    L = np.zeros(n)
    R = np.zeros(n)
    
    np.random.seed(123)
    
    # Objective: minimize f(x) = sum of sin peaks (multi-modal)
    def objective(x):
        return x**2 * 0.01 - 4 * np.cos(x) - 2 * np.cos(2.7 * x) + 10
    
    # Annealing schedule
    n_steps = 3000
    T_start = 50.0
    T_end = 0.01
    temps = np.geomspace(T_start, T_end, n_steps)
    
    x = np.random.uniform(-15, 15)  # initial position
    best_x = x
    best_val = objective(x)
    
    history = []
    accepted_uphills = []
    
    for i in range(n_steps):
        T = temps[i]
        # Proposal: jump size proportional to temperature
        jump = np.random.normal(0, 0.5 + T * 0.3)
        x_new = x + jump
        
        val_old = objective(x)
        val_new = objective(x_new)
        delta = val_new - val_old
        
        uphill_accepted = False
        if delta < 0:
            x = x_new
        elif np.random.random() < np.exp(-delta / T):
            x = x_new
            uphill_accepted = True
        
        if objective(x) < best_val:
            best_val = objective(x)
            best_x = x
        
        history.append((x, T, objective(x), uphill_accepted))
        if uphill_accepted:
            accepted_uphills.append(i)
    
    step_dur = dur / n_steps
    
    # Main voice: position → frequency
    for i, (pos, T, val, uphill) in enumerate(history):
        t0 = int(i * step_dur * SR)
        t1 = min(int((i + 1) * step_dur * SR), n)
        if t1 <= t0:
            continue
        
        t = np.arange(t1 - t0) / SR
        
        # Frequency: position mapped to pitch
        freq = 220 * 2**(pos * 0.08)  # ~semitone steps
        freq = np.clip(freq, 65, 1200)
        
        # Temperature → harmonic content
        # High T = many harmonics (bright/noisy), Low T = pure
        progress = i / n_steps
        n_harm = max(1, int(1 + (1 - progress) * 10))
        
        # Amplitude envelope shaped by temperature
        amp = 0.08 * (0.3 + 0.7 * (1 - progress * 0.5))
        
        sig = np.zeros(len(t))
        for h in range(1, n_harm + 1):
            # High T: slight detuning for roughness
            detune = 1.0 + np.random.normal(0, 0.003 * (1 - progress))
            decay = 1.0 / h**1.0
            sig += decay * np.sin(2 * np.pi * freq * h * detune * t)
        sig *= amp
        
        # Uphill acceptance = bright FM burst
        if uphill:
            burst_freq = freq * 1.5
            fm_depth = 300 * (T / T_start)
            carrier = np.sin(2 * np.pi * burst_freq * t + fm_depth * np.sin(2 * np.pi * 7 * t))
            sig += 0.06 * carrier * np.exp(-t * 15)
        
        # Step-level fade
        if len(sig) > 100:
            sig[:50] *= np.linspace(0, 1, 50)
            sig[-50:] *= np.linspace(1, 0, 50)
        
        # Stereo: temperature → width (hot=wide, cold=center)
        width = 0.1 + 0.4 * (T / T_start)
        pan = 0.5 + np.random.uniform(-width, width)
        pan = np.clip(pan, 0.05, 0.95)
        
        L[t0:t1] += sig * (1 - pan)
        R[t0:t1] += sig * pan
    
    # Background texture: temperature as filtered noise
    t_full = np.arange(n) / SR
    noise = np.random.normal(0, 1, n)
    # Simple lowpass: temperature controls cutoff via moving average
    for i in range(n_steps):
        t0 = int(i * step_dur * SR)
        t1 = min(int((i + 1) * step_dur * SR), n)
        T = temps[i]
        amp = 0.02 * (T / T_start)**0.5
        noise[t0:t1] *= amp
    
    L += noise * 0.5
    R += noise * 0.5
    
    # Crystallization chord: last 4 seconds
    crystal_start = int((dur - 4.0) * SR)
    crystal_t = np.arange(n - crystal_start) / SR
    crystal_env = fade(len(crystal_t), 1.5, 1.5)
    # Pure A major: A2, C#3, E3, A3
    crystal = np.zeros(len(crystal_t))
    for f in [110, 138.59, 164.81, 220]:
        crystal += 0.05 * np.sin(2 * np.pi * f * crystal_t)
    crystal *= crystal_env
    L[crystal_start:] += crystal
    R[crystal_start:] += crystal
    
    # Low drone (A1 = 55 Hz) throughout
    drone = 0.04 * np.sin(2 * np.pi * 55 * t_full) * fade(n, 2.0, 2.0)
    L += drone
    R += drone
    
    stereo = np.array([L, R])
    stereo *= fade(n, 1.0, 2.0)
    write_wav("opt_2_simulated_annealing", stereo)

# ── Piece 3: Genetic Algorithm ─────────────────────────────────────────────

def genetic_algorithm():
    """Population evolving toward a target melody.
    
    20 organisms, each a sequence of 8 frequencies.
    Target: a simple pentatonic phrase.
    Fitness: inverse of distance from target.
    
    Initially cacophonous (random), gradually converging.
    Crossover = brief interleaving of two parent voices.
    Mutation = sudden FM burst on affected genes.
    The sound of evolution: noise → harmony.
    """
    print("Generating Genetic Algorithm...")
    dur = 55.0
    n = int(dur * SR)
    L = np.zeros(n)
    R = np.zeros(n)
    
    np.random.seed(77)
    
    # Target melody: D minor pentatonic phrase
    target = np.array([293.66, 349.23, 392.00, 440.00, 523.25, 440.00, 349.23, 293.66])
    gene_len = len(target)
    
    pop_size = 20
    n_generations = 120
    
    # Initialize random population
    population = np.random.uniform(100, 800, (pop_size, gene_len))
    
    def fitness(individual):
        return -np.sum((individual - target)**2)
    
    gen_dur = dur / n_generations
    note_dur = gen_dur / gene_len
    
    all_generations = [population.copy()]
    mutations = []  # (gen, individual, gene_index)
    crossovers = []  # (gen, parent1, parent2)
    
    for g in range(n_generations):
        # Evaluate fitness
        fits = np.array([fitness(ind) for ind in population])
        
        # Selection: tournament
        new_pop = np.zeros_like(population)
        for i in range(pop_size):
            # Tournament of 3
            contestants = np.random.choice(pop_size, 3, replace=False)
            winner = contestants[np.argmax(fits[contestants])]
            new_pop[i] = population[winner]
        
        # Crossover
        for i in range(0, pop_size - 1, 2):
            if np.random.random() < 0.7:
                cx_point = np.random.randint(1, gene_len)
                child1 = np.concatenate([new_pop[i][:cx_point], new_pop[i+1][cx_point:]])
                child2 = np.concatenate([new_pop[i+1][:cx_point], new_pop[i][cx_point:]])
                crossovers.append((g, i, i+1))
                new_pop[i] = child1
                new_pop[i+1] = child2
        
        # Mutation
        mutation_rate = 0.15 * (1 - g / n_generations * 0.7)  # decreasing
        for i in range(pop_size):
            for j in range(gene_len):
                if np.random.random() < mutation_rate:
                    # Mutation magnitude decreases with generations
                    mag = 200 * (1 - g / n_generations * 0.8)
                    new_pop[i][j] += np.random.normal(0, mag)
                    new_pop[i][j] = np.clip(new_pop[i][j], 60, 1200)
                    mutations.append((g, i, j))
        
        # Elitism: keep best from previous gen
        best_idx = np.argmax(fits)
        worst_new = np.argmin([fitness(ind) for ind in new_pop])
        new_pop[worst_new] = population[best_idx]
        
        population = new_pop
        all_generations.append(population.copy())
    
    # Synthesize: each generation plays its "best 5" individuals
    for g in range(n_generations):
        pop = all_generations[g]
        fits = np.array([fitness(ind) for ind in pop])
        best_5 = np.argsort(fits)[-5:]
        
        progress = g / n_generations
        gen_start = int(g * gen_dur * SR)
        
        for rank, idx in enumerate(best_5):
            ind = pop[idx]
            # Amplitude: best individual loudest
            amp = 0.06 * (1.0 - rank * 0.15)
            
            # Stereo spread: early = wide (chaotic), late = narrow (converged)
            width = 0.4 * (1 - progress * 0.7)
            pan = 0.5 + (rank - 2) * width * 0.25
            pan = np.clip(pan, 0.05, 0.95)
            
            for note_i in range(gene_len):
                freq = ind[note_i]
                t0 = gen_start + int(note_i * note_dur * SR)
                t1 = min(t0 + int(note_dur * SR), n)
                if t1 <= t0 or t0 >= n:
                    continue
                
                length = t1 - t0
                t = np.arange(length) / SR
                
                # Harmonic count: increases as we converge (clarity emerging)
                dist_to_target = abs(freq - target[note_i])
                closeness = max(0, 1 - dist_to_target / 400)
                n_harm = int(1 + closeness * 4)
                
                sig = np.zeros(length)
                for h in range(1, n_harm + 1):
                    # Slight detuning for organisms far from target
                    detune = 1.0 + (1 - closeness) * np.random.normal(0, 0.005)
                    sig += (1.0 / h**1.3) * np.sin(2 * np.pi * freq * h * detune * t)
                sig *= amp
                
                # Note envelope
                env = np.ones(length)
                attack = min(int(0.01 * SR), length // 4)
                release = min(int(0.03 * SR), length // 4)
                if attack > 0:
                    env[:attack] *= np.linspace(0, 1, attack)
                if release > 0:
                    env[-release:] *= np.linspace(1, 0, release)
                sig *= env
                
                L[t0:t1] += sig * (1 - pan)
                R[t0:t1] += sig * pan
    
    # Mutation bursts: FM clicks at mutation events
    for g, ind_i, gene_i in mutations:
        t0 = int((g * gen_dur + gene_i * note_dur) * SR)
        burst_len = min(int(0.02 * SR), n - t0)
        if burst_len <= 0 or t0 >= n:
            continue
        t = np.arange(burst_len) / SR
        burst = 0.03 * np.sin(2 * np.pi * 2000 * t) * np.exp(-t * 200)
        pan = np.random.uniform(0.2, 0.8)
        L[t0:t0+burst_len] += burst * (1 - pan)
        R[t0:t0+burst_len] += burst * pan
    
    # Background: target melody as ghost reference (very quiet, grows louder)
    t_full = np.arange(n) / SR
    target_ghost = np.zeros(n)
    melody_dur = dur / 8  # one note per section
    for i, freq in enumerate(target):
        t0 = 0
        ghost_sig = 0.008 * np.sin(2 * np.pi * freq * t_full)
        # Envelope: barely audible at start, clear at end
        progress_env = np.linspace(0.05, 0.6, n)
        target_ghost += ghost_sig * progress_env
    
    # The ghost is very subtle — just a hint of where evolution is heading
    target_ghost *= 0.15
    L += target_ghost
    R += target_ghost
    
    # Final generation: play target melody clearly as resolution (last 6s)
    coda_start = int((dur - 6.0) * SR)
    coda_note_dur = 5.0 / gene_len
    for i, freq in enumerate(target):
        t0 = coda_start + int(i * coda_note_dur * SR)
        t1 = min(t0 + int(coda_note_dur * SR), n)
        if t1 <= t0:
            continue
        t = np.arange(t1 - t0) / SR
        sig = np.zeros(len(t))
        for h in range(1, 6):
            sig += (1.0 / h**1.5) * np.sin(2 * np.pi * freq * h * t)
        sig *= 0.08 * fade(len(t), 0.05, 0.15)
        L[t0:t1] += sig * 0.45
        R[t0:t1] += sig * 0.55
    
    # Low D drone
    drone = 0.03 * np.sin(2 * np.pi * 73.42 * t_full) * fade(n, 2.0, 2.0)
    L += drone
    R += drone
    
    stereo = np.array([L, R])
    stereo *= fade(n, 1.0, 2.5)
    write_wav("opt_3_genetic_algorithm", stereo)

# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Phase 14: Optimization")
    print("=" * 50)
    gradient_descent()
    simulated_annealing()
    genetic_algorithm()
    print("\nDone.")
