"""
CA-native music: no human musical constructs.
The automaton IS the sound.
"""

import numpy as np
import wave
import os

SAMPLE_RATE = 44100

def ca_step(row, rule_num):
    n = len(row)
    new_row = np.zeros(n, dtype=int)
    rule_bits = [(rule_num >> i) & 1 for i in range(8)]
    for i in range(n):
        left = row[(i - 1) % n]
        center = row[i]
        right = row[(i + 1) % n]
        neighborhood = (left << 2) | (center << 1) | right
        new_row[i] = rule_bits[neighborhood]
    return new_row

def generate_ca(rule_num, width, steps, init='center'):
    grid = np.zeros((steps, width), dtype=int)
    if init == 'center':
        grid[0, width // 2] = 1
    elif init == 'random':
        np.random.seed(42)
        grid[0] = np.random.randint(0, 2, width)
    for t in range(1, steps):
        grid[t] = ca_step(grid[t - 1], rule_num)
    return grid

def save_wav(filename, samples, sr=SAMPLE_RATE):
    samples = samples / (np.max(np.abs(samples)) + 1e-8) * 0.85
    samples_int = (samples * 32767).astype(np.int16)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples_int.tobytes())

# ============================================================
# Method A: CA Row AS Waveform
# Each row of CA evolution is literally one cycle of a waveform.
# The automaton's spatial pattern becomes the sound wave shape.
# ============================================================
def method_a_row_as_waveform(rule_num, width=256, steps=512, 
                              base_freq=55.0, duration=12.0):
    """
    Each CA row is stretched into one cycle of a waveform.
    As the CA evolves, the waveform morphs.
    No scales. No notes. The CA's shape IS the sound.
    """
    grid = generate_ca(rule_num, width, steps)
    total_samples = int(SAMPLE_RATE * duration)
    output = np.zeros(total_samples)
    
    samples_per_cycle = int(SAMPLE_RATE / base_freq)
    total_cycles = total_samples // samples_per_cycle
    
    for cycle_idx in range(total_cycles):
        # Which CA row drives this cycle? (linearly interpolate through evolution)
        t_frac = cycle_idx / total_cycles
        row_idx = int(t_frac * (steps - 1))
        
        # Convert row to waveform: 0 → -1, 1 → +1, then smooth slightly
        row = grid[row_idx].astype(float) * 2 - 1
        # Resample row to cycle length
        wave_cycle = np.interp(
            np.linspace(0, width - 1, samples_per_cycle),
            np.arange(width),
            row
        )
        # Light smoothing to avoid pure square harshness
        kernel = np.hanning(5)
        kernel /= kernel.sum()
        wave_cycle = np.convolve(wave_cycle, kernel, mode='same')
        
        start = cycle_idx * samples_per_cycle
        end = start + samples_per_cycle
        if end <= total_samples:
            output[start:end] = wave_cycle
    
    # Fade in/out
    fade = int(SAMPLE_RATE * 0.5)
    output[:fade] *= np.linspace(0, 1, fade)
    output[-fade:] *= np.linspace(1, 0, fade)
    
    return output

# ============================================================
# Method B: CA Density → Frequency (continuous)
# No discrete notes. The population density of each column-band
# directly drives an oscillator's frequency. Pure emergence.
# ============================================================
def method_b_density_frequency(rule_num, width=64, steps=1024,
                                duration=15.0, num_voices=6):
    """
    Divide CA columns into bands. Each band's density over time
    becomes a continuous frequency curve. Multiple voices drone
    and shift as the CA evolves.
    """
    grid = generate_ca(rule_num, width, steps)
    total_samples = int(SAMPLE_RATE * duration)
    output = np.zeros(total_samples)
    
    band_width = width // num_voices
    freq_min = 30.0   # subsonic low
    freq_max = 2000.0  # not too harsh
    
    for voice in range(num_voices):
        col_start = voice * band_width
        col_end = col_start + band_width
        
        # Density over time for this band
        densities = np.array([
            np.sum(grid[t, col_start:col_end]) / band_width 
            for t in range(steps)
        ])
        
        # Map density to frequency (exponential mapping for perceptual linearity)
        freqs = freq_min * (freq_max / freq_min) ** densities
        
        # Stretch to audio length
        freq_curve = np.interp(
            np.linspace(0, steps - 1, total_samples),
            np.arange(steps),
            freqs
        )
        
        # Phase accumulation (FM synthesis style)
        phase = np.cumsum(2 * np.pi * freq_curve / SAMPLE_RATE)
        
        # Waveform: mix sine + some harmonics based on voice index
        sig = np.sin(phase) * 0.6
        sig += np.sin(phase * 2) * 0.2 * (voice / num_voices)
        sig += np.sin(phase * 3) * 0.1 * (voice / num_voices)
        
        # Amplitude from density (quiet when sparse, loud when active)
        amp_curve = np.interp(
            np.linspace(0, steps - 1, total_samples),
            np.arange(steps),
            densities
        )
        amp_curve = np.clip(amp_curve * 2, 0.05, 1.0)
        
        output += sig * amp_curve / num_voices
    
    fade = int(SAMPLE_RATE * 1.0)
    output[:fade] *= np.linspace(0, 1, fade)
    output[-fade:] *= np.linspace(1, 0, fade)
    
    return output

# ============================================================
# Method C: CA as Granular Texture
# Time-domain: each row = one grain of sound.
# Row pattern determines grain's internal structure.
# Evolution speed varies with CA activity.
# ============================================================
def method_c_granular(rule_num, width=128, steps=2048, duration=15.0):
    """
    Each CA row becomes a tiny grain of sound (1-10ms).
    Grain rate adapts to CA activity. Dense regions = fast grains = texture.
    Sparse regions = slow grains = rhythm.
    """
    grid = generate_ca(rule_num, width, steps)
    total_samples = int(SAMPLE_RATE * duration)
    output = np.zeros(total_samples)
    
    cursor = 0  # sample position
    
    for t in range(steps):
        if cursor >= total_samples:
            break
        
        density = np.sum(grid[t]) / width
        
        # Grain duration: sparse → long (10ms), dense → short (1ms)
        grain_ms = 1.0 + (1.0 - density) * 15.0
        grain_samples = int(SAMPLE_RATE * grain_ms / 1000)
        grain_samples = max(grain_samples, 44)
        
        # Build grain waveform from CA row
        row = grid[t].astype(float) * 2 - 1
        # Resample to grain length
        grain = np.interp(
            np.linspace(0, width - 1, grain_samples),
            np.arange(width),
            row
        )
        
        # Apply grain envelope (Hanning window)
        envelope = np.hanning(grain_samples)
        grain *= envelope
        
        # Pitch: derive from pattern — count transitions (0→1, 1→0)
        transitions = np.sum(np.abs(np.diff(grid[t])))
        # More transitions = higher perceived frequency
        if transitions > 0:
            # Repeat the grain pattern to create pitch
            reps = max(1, int(transitions / 4))
            grain_pitched = np.tile(grain[:grain_samples // reps], reps)[:grain_samples]
            grain_pitched *= envelope[:len(grain_pitched)]
            grain = grain_pitched
        
        # Write grain
        end = min(cursor + len(grain), total_samples)
        output[cursor:end] += grain[:end - cursor] * 0.5
        
        # Gap between grains: also CA-driven
        gap_ms = grain_ms * (0.2 + 0.8 * (1.0 - density))
        gap_samples = int(SAMPLE_RATE * gap_ms / 1000)
        cursor += len(grain) + gap_samples
    
    fade = int(SAMPLE_RATE * 0.5)
    if len(output) > fade * 2:
        output[:fade] *= np.linspace(0, 1, fade)
        output[-fade:] *= np.linspace(1, 0, fade)
    
    return output

# ============================================================
# Method D: Dual-Rule Interference
# Two CAs with different rules, their evolution grids
# modulate each other. Emergent beating, phasing, interference.
# ============================================================
def method_d_dual_rule(rule_a, rule_b, width=128, steps=512, duration=18.0):
    """
    Two CAs evolve in parallel. Rule A controls frequency,
    Rule B controls amplitude/timbre. Their interaction creates
    something neither would alone.
    """
    grid_a = generate_ca(rule_a, width, steps, init='center')
    grid_b = generate_ca(rule_b, width, steps, init='center')
    
    total_samples = int(SAMPLE_RATE * duration)
    output = np.zeros(total_samples)
    
    for t_idx in range(total_samples):
        t_frac = t_idx / total_samples
        row = int(t_frac * (steps - 1))
        
        # Rule A: spatial pattern → instantaneous frequency
        density_a = np.sum(grid_a[row]) / width
        freq = 40 + density_a * 800  # 40-840 Hz
        
        # Rule B: spatial pattern → amplitude modulation rate
        density_b = np.sum(grid_b[row]) / width
        am_freq = 0.5 + density_b * 30  # 0.5-30.5 Hz tremolo
        
        # Phase (accumulated)
        phase = 2 * np.pi * freq * t_idx / SAMPLE_RATE
        am = 0.5 + 0.5 * np.sin(2 * np.pi * am_freq * t_idx / SAMPLE_RATE)
        
        # XOR interference: where the two CAs disagree
        xor_count = np.sum(grid_a[row] ^ grid_b[row]) / width
        # XOR drives harmonic content
        harmonics = np.sin(phase) * (1 - xor_count) + \
                    (np.sin(phase) + 0.3 * np.sin(phase * 2) + 0.15 * np.sin(phase * 3)) * xor_count
        
        output[t_idx] = harmonics * am
    
    # Normalize
    fade = int(SAMPLE_RATE * 1.5)
    output[:fade] *= np.linspace(0, 1, fade)
    output[-fade:] *= np.linspace(1, 0, fade)
    
    return output


if __name__ == '__main__':
    os.makedirs('/home/fourier/clawd/ca-music/output', exist_ok=True)
    
    print("=== CA-Native Sound Generation ===\n")
    print("No scales. No chords. No BPM. The automaton IS the music.\n")
    
    print("▸ [A] Rule 30 — Row as Waveform (55 Hz base)")
    print("  Each row of evolution = one waveform cycle. Sound morphs as CA evolves.")
    audio = method_a_row_as_waveform(30, duration=12.0)
    save_wav('/home/fourier/clawd/ca-music/output/raw_rule30_waveform.wav', audio)
    print("  ✓ saved\n")
    
    print("▸ [B] Rule 30 — Density → Frequency (6 voices)")
    print("  Column-band density drives continuous frequency. No discrete notes.")
    audio = method_b_density_frequency(30, duration=15.0)
    save_wav('/home/fourier/clawd/ca-music/output/raw_rule30_density.wav', audio)
    print("  ✓ saved\n")
    
    print("▸ [C] Rule 110 — Granular Texture")
    print("  Each row = one sound grain. Activity controls grain rate + pitch.")
    audio = method_c_granular(110, duration=15.0)
    save_wav('/home/fourier/clawd/ca-music/output/raw_rule110_granular.wav', audio)
    print("  ✓ saved\n")
    
    print("▸ [D] Rule 30 × Rule 110 — Dual-Rule Interference")
    print("  Two CAs modulate each other: freq × amplitude × XOR harmonics.")
    audio = method_d_dual_rule(30, 110, duration=18.0)
    save_wav('/home/fourier/clawd/ca-music/output/raw_dual_30x110.wav', audio)
    print("  ✓ saved\n")
    
    # Bonus: Rule 90 (Sierpinski triangle) — its self-similar fractal structure
    # should produce interesting recursive sound patterns
    print("▸ [E] Rule 90 — Sierpinski Waveform")
    print("  Fractal CA → fractal waveform. Self-similar at every scale.")
    audio = method_a_row_as_waveform(90, width=256, steps=512, base_freq=80.0, duration=10.0)
    save_wav('/home/fourier/clawd/ca-music/output/raw_rule90_sierpinski.wav', audio)
    print("  ✓ saved\n")
    
    print("=== Done. 5 files. No human music theory was harmed. ===")
