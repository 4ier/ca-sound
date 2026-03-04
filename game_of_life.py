"""
Phase 6: Game of Life — Conway's cellular automaton as music
============================================================

Three pieces exploring different musical dimensions of GoL:

1. **Glider Symphony** (60s, stereo) — Track a glider's diagonal journey across
   a 64x64 grid. Population density → chord richness, center of mass → stereo
   position, glider's local neighborhood → melodic motif on top. Sparse beginning,
   then add a Gosper glider gun at t=200 to flood the field with gliders —
   texture thickens from solo to orchestra.

2. **Still Life Chorale** (45s) — The 7 canonical still lifes (block, beehive,
   loaf, boat, ship, tub, pond) each become a sustained chord voiced by their
   cell count and geometry. Play them sequentially as a slow chorale, then
   overlay all simultaneously for a final cluster chord. Stability as harmony.

3. **Methuselah** (55s, stereo) — R-pentomino's famous 1103-generation eruption.
   Population curve → bass frequency, birth/death balance → brightness,
   bounding box expansion → stereo width. The chaos settles into still lifes
   and oscillators — musical turbulence resolving to consonance.
"""

import numpy as np
import struct
import os

SAMPLE_RATE = 44100

def write_wav(filename, data, sr=SAMPLE_RATE):
    """Write mono or stereo float array to WAV."""
    if data.ndim == 1:
        channels = 1
        samples = data
    else:
        channels = 2
        samples = data.T.flatten()
    samples = np.clip(samples, -1, 1)
    samples_int = (samples * 32767).astype(np.int16)
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", filename)
    with open(path, 'wb') as f:
        n = len(samples_int)
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + n * 2))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, channels, sr, sr * channels * 2, channels * 2, 16))
        f.write(b'data')
        f.write(struct.pack('<I', n * 2))
        f.write(samples_int.tobytes())
    print(f"  Written: {path}")

# ─── GoL Engine ───

def gol_step(grid):
    """One step of Conway's Game of Life with wraparound."""
    n = np.zeros_like(grid)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            n += np.roll(np.roll(grid, di, axis=0), dj, axis=1)
    return ((grid == 1) & ((n == 2) | (n == 3)) | (grid == 0) & (n == 3)).astype(np.int8)

def run_gol(grid, steps):
    """Run GoL for N steps, return list of grids."""
    history = [grid.copy()]
    for _ in range(steps):
        grid = gol_step(grid)
        history.append(grid)
    return history

def place_pattern(grid, pattern, r, c):
    """Place a pattern (list of (dr, dc)) onto grid."""
    for dr, dc in pattern:
        grid[(r + dr) % grid.shape[0], (c + dc) % grid.shape[1]] = 1

# ─── Patterns ───

GLIDER = [(0,1), (1,2), (2,0), (2,1), (2,2)]

GOSPER_GUN = [
    (0,24), (1,22),(1,24), (2,12),(2,13),(2,20),(2,21),(2,34),(2,35),
    (3,11),(3,15),(3,20),(3,21),(3,34),(3,35), (4,0),(4,1),(4,10),(4,16),(4,20),(4,21),
    (5,0),(5,1),(5,10),(5,14),(5,16),(5,17),(5,22),(5,24), (6,10),(6,16),(6,24),
    (7,11),(7,15), (8,12),(8,13)
]

R_PENTOMINO = [(0,1),(0,2),(1,0),(1,1),(2,1)]

STILL_LIFES = {
    'block':   [(0,0),(0,1),(1,0),(1,1)],
    'beehive': [(0,1),(0,2),(1,0),(1,3),(2,1),(2,2)],
    'loaf':    [(0,1),(0,2),(1,0),(1,3),(2,1),(2,3),(3,2)],
    'boat':    [(0,0),(0,1),(1,0),(1,2),(2,1)],
    'ship':    [(0,0),(0,1),(1,0),(1,2),(2,1),(2,2)],
    'tub':     [(0,1),(1,0),(1,2),(2,1)],
    'pond':    [(0,1),(0,2),(1,0),(1,3),(2,0),(2,3),(3,1),(3,2)],
}

# ─── Synthesis helpers ───

def sine(freq, duration, sr=SAMPLE_RATE):
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t)

def envelope(signal, attack=0.01, release=0.05, sr=SAMPLE_RATE):
    n = len(signal)
    env = np.ones(n)
    att = min(int(attack * sr), n)
    rel = min(int(release * sr), n)
    env[:att] = np.linspace(0, 1, att)
    env[-rel:] = np.linspace(1, 0, rel)
    return signal * env

def smooth(arr, window=5):
    """Simple moving average smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')

# ─── Piece 1: Glider Symphony ───

def glider_symphony():
    print("Generating Glider Symphony...")
    duration = 60.0
    size = 64
    steps = 1200
    
    grid = np.zeros((size, size), dtype=np.int8)
    # Single glider at top-left
    place_pattern(grid, GLIDER, 2, 2)
    
    # Run first 200 steps (solo glider)
    history_solo = run_gol(grid, 200)
    
    # Add Gosper gun at step 200
    grid_with_gun = history_solo[-1].copy()
    place_pattern(grid_with_gun, GOSPER_GUN, 5, 5)
    
    # Run remaining steps
    history_gun = run_gol(grid_with_gun, steps - 200)
    
    history = history_solo + history_gun[1:]
    
    # Extract features per step
    populations = np.array([g.sum() for g in history], dtype=float)
    
    centers_x = []
    centers_y = []
    for g in history:
        if g.sum() > 0:
            ys, xs = np.where(g)
            centers_x.append(xs.mean() / size)
            centers_y.append(ys.mean() / size)
        else:
            centers_x.append(0.5)
            centers_y.append(0.5)
    centers_x = np.array(centers_x)
    
    # Synthesize
    samples_per_step = int(duration * SAMPLE_RATE / len(history))
    total_samples = int(duration * SAMPLE_RATE)
    left = np.zeros(total_samples)
    right = np.zeros(total_samples)
    
    # Base frequency and scale
    base_freq = 110.0  # A2
    scale_ratios = [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2]  # major scale
    
    max_pop = max(populations.max(), 1)
    
    for i, g in enumerate(history):
        start = i * samples_per_step
        if start >= total_samples:
            break
        end = min(start + samples_per_step, total_samples)
        seg_len = end - start
        if seg_len <= 0:
            continue
        
        t = np.arange(seg_len) / SAMPLE_RATE
        pop_norm = populations[i] / max_pop
        cx = centers_x[i]  # 0..1 → stereo pan
        
        # Chord richness proportional to population
        n_harmonics = max(1, int(pop_norm * 8))
        sig = np.zeros(seg_len)
        for h in range(n_harmonics):
            ratio = scale_ratios[h % len(scale_ratios)] * (1 + h // len(scale_ratios))
            freq = base_freq * ratio
            amp = 0.3 / (h + 1)
            sig += amp * np.sin(2 * np.pi * freq * t)
        
        # Melodic motif on top — frequency from population derivative
        if i > 0:
            delta = populations[i] - populations[i-1]
            motif_freq = 440 + delta * 10
            motif_freq = np.clip(motif_freq, 200, 2000)
            sig += 0.15 * np.sin(2 * np.pi * motif_freq * t) * pop_norm
        
        # Apply gentle envelope to avoid clicks
        if seg_len > 100:
            fade = 50
            sig[:fade] *= np.linspace(0, 1, fade)
            sig[-fade:] *= np.linspace(1, 0, fade)
        
        # Stereo pan from center of mass x
        pan_r = cx
        pan_l = 1 - cx
        left[start:end] += sig * pan_l
        right[start:end] += sig * pan_r
    
    # Normalize
    peak = max(np.abs(left).max(), np.abs(right).max(), 1e-6)
    stereo = np.stack([left / peak * 0.85, right / peak * 0.85])
    write_wav("gol_1_glider_symphony.wav", stereo)

# ─── Piece 2: Still Life Chorale ───

def still_life_chorale():
    print("Generating Still Life Chorale...")
    duration = 45.0
    total = int(duration * SAMPLE_RATE)
    signal = np.zeros(total)
    
    base = 130.81  # C3
    
    names = list(STILL_LIFES.keys())
    n_patterns = len(names)
    
    # Each still life gets ~5s solo, then 10s combined
    solo_dur = 5.0
    combined_dur = duration - solo_dur * n_patterns
    
    for idx, name in enumerate(names):
        cells = STILL_LIFES[name]
        n_cells = len(cells)
        
        # Geometry → chord voicing
        # Use cell positions to determine intervals
        xs = [c[1] for c in cells]
        ys = [c[0] for c in cells]
        spread_x = max(xs) - min(xs) + 1
        spread_y = max(ys) - min(ys) + 1
        
        # Map cell count to harmonic richness, geometry to specific intervals
        freqs = []
        for i, (dy, dx) in enumerate(cells):
            # Each cell is a voice: position determines frequency offset
            ratio = 1 + (dx * 0.25 + dy * 0.125)
            freqs.append(base * ratio * (1 + idx * 0.15))
        
        # Solo section
        start = int(idx * solo_dur * SAMPLE_RATE)
        seg_len = int(solo_dur * SAMPLE_RATE)
        t = np.arange(seg_len) / SAMPLE_RATE
        
        chord = np.zeros(seg_len)
        for f in freqs:
            chord += (0.2 / n_cells) * np.sin(2 * np.pi * f * t)
        
        # Gentle envelope
        chord = envelope(chord, attack=0.3, release=0.5)
        end = min(start + seg_len, total)
        signal[start:end] += chord[:end-start]
        
        # Also add to combined section
        comb_start = int((solo_dur * n_patterns) * SAMPLE_RATE)
        comb_len = int(combined_dur * SAMPLE_RATE)
        if comb_len > 0:
            t2 = np.arange(comb_len) / SAMPLE_RATE
            comb_chord = np.zeros(comb_len)
            for f in freqs:
                comb_chord += (0.12 / n_cells) * np.sin(2 * np.pi * f * t2)
            # Fade in for combined section
            fade_in = min(int(0.5 * SAMPLE_RATE), comb_len)
            comb_chord[:fade_in] *= np.linspace(0, 1, fade_in)
            fade_out = min(int(2.0 * SAMPLE_RATE), comb_len)
            comb_chord[-fade_out:] *= np.linspace(1, 0, fade_out)
            end2 = min(comb_start + comb_len, total)
            signal[comb_start:end2] += comb_chord[:end2-comb_start]
    
    peak = np.abs(signal).max()
    if peak > 0:
        signal = signal / peak * 0.85
    write_wav("gol_2_still_life_chorale.wav", signal)

# ─── Piece 3: Methuselah ───

def methuselah():
    print("Generating Methuselah (R-pentomino)...")
    duration = 55.0
    size = 128  # need larger grid for R-pentomino
    steps = 1103  # R-pentomino stabilizes around step 1103
    
    grid = np.zeros((size, size), dtype=np.int8)
    place_pattern(grid, R_PENTOMINO, size//2, size//2)
    
    print("  Running simulation (1103 steps)...")
    history = run_gol(grid, steps)
    
    # Extract features
    populations = np.array([g.sum() for g in history], dtype=float)
    
    births = []
    deaths = []
    bbox_widths = []
    for i in range(len(history)):
        g = history[i]
        ys, xs = np.where(g)
        if len(xs) > 0:
            bbox_widths.append((xs.max() - xs.min() + ys.max() - ys.min()) / size)
        else:
            bbox_widths.append(0)
        if i > 0:
            b = ((history[i] == 1) & (history[i-1] == 0)).sum()
            d = ((history[i] == 0) & (history[i-1] == 1)).sum()
            births.append(b)
            deaths.append(d)
        else:
            births.append(0)
            deaths.append(0)
    
    births = np.array(births, dtype=float)
    deaths = np.array(deaths, dtype=float)
    bbox_widths = np.array(bbox_widths)
    
    # Smooth features
    populations_s = smooth(populations, 15)
    births_s = smooth(births, 15)
    deaths_s = smooth(deaths, 15)
    bbox_s = smooth(bbox_widths, 15)
    
    # Synthesize
    total = int(duration * SAMPLE_RATE)
    samples_per_step = total / len(history)
    left = np.zeros(total)
    right = np.zeros(total)
    
    max_pop = populations_s.max() if populations_s.max() > 0 else 1
    max_bd = max(births_s.max(), deaths_s.max(), 1)
    max_bbox = max(bbox_s.max(), 0.01)
    
    # Generate continuous audio by interpolating features
    # Bass: population → frequency (55-220 Hz)
    # Brightness: birth-death balance → harmonic content
    # Width: bbox → stereo spread
    
    chunk_size = 1024
    for s in range(0, total, chunk_size):
        end = min(s + chunk_size, total)
        seg_len = end - s
        
        # Which step are we at?
        step_idx = int(s / total * (len(history) - 1))
        step_idx = min(step_idx, len(history) - 1)
        
        pop_n = populations_s[step_idx] / max_pop
        birth_n = births_s[step_idx] / max_bd
        death_n = deaths_s[step_idx] / max_bd
        bbox_n = bbox_s[step_idx] / max_bbox
        
        t = (np.arange(seg_len) + s) / SAMPLE_RATE
        
        # Bass drone — population maps to pitch
        bass_freq = 55 + pop_n * 165  # 55-220 Hz
        sig = 0.4 * np.sin(2 * np.pi * bass_freq * t)
        
        # Add harmonics based on activity (births = bright, deaths = dark)
        activity = (birth_n + death_n) / 2
        brightness = birth_n / (birth_n + death_n + 1e-6)
        
        # Bright harmonics (births dominate)
        for h in range(2, int(2 + activity * 8)):
            amp = 0.15 * brightness / h
            sig += amp * np.sin(2 * np.pi * bass_freq * h * t)
        
        # Dark low harmonics (deaths dominate)
        sig += 0.2 * (1 - brightness) * activity * np.sin(2 * np.pi * bass_freq * 0.5 * t)
        
        # High chaotic voice — more present during turbulent phases
        turbulence = abs(birth_n - death_n)
        if turbulence > 0.1:
            noise_freq = 440 + turbulence * 800
            sig += 0.1 * turbulence * np.sin(2 * np.pi * noise_freq * t + 
                                               3 * np.sin(2 * np.pi * noise_freq * 0.7 * t))
        
        # Stereo from bounding box
        pan = 0.5 + 0.4 * (bbox_n - 0.5)
        left[s:end] += sig * (1 - pan)
        right[s:end] += sig * pan
    
    # Final fade out (the settling)
    fade_samples = int(3.0 * SAMPLE_RATE)
    fade = np.linspace(1, 0, fade_samples)
    left[-fade_samples:] *= fade
    right[-fade_samples:] *= fade
    
    peak = max(np.abs(left).max(), np.abs(right).max(), 1e-6)
    stereo = np.stack([left / peak * 0.85, right / peak * 0.85])
    write_wav("gol_3_methuselah.wav", stereo)

# ─── Main ───

if __name__ == "__main__":
    print("=== Phase 6: Game of Life ===\n")
    glider_symphony()
    print()
    still_life_chorale()
    print()
    methuselah()
    print("\nDone!")
