"""
Cellular Automata Music Generator
Generates audio from Rule 30 and Rule 110 evolution patterns.
"""

import numpy as np
import wave
import struct
import os

SAMPLE_RATE = 44100

# --- CA Engine ---
def ca_step(row, rule_num):
    """Apply 1D elementary CA rule to a row."""
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
    """Generate CA evolution grid."""
    grid = np.zeros((steps, width), dtype=int)
    if init == 'center':
        grid[0, width // 2] = 1
    elif init == 'random':
        grid[0] = np.random.randint(0, 2, width)
    for t in range(1, steps):
        grid[t] = ca_step(grid[t - 1], rule_num)
    return grid

# --- Audio Synthesis ---
def make_note(freq, duration, sr=SAMPLE_RATE, decay=0.8, wave_type='triangle'):
    """Generate a single note with envelope."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Envelope: attack + decay
    env = np.exp(-decay * t / duration)
    env[:int(sr * 0.005)] = np.linspace(0, 1, int(sr * 0.005))  # 5ms attack
    
    if wave_type == 'sine':
        sig = np.sin(2 * np.pi * freq * t)
    elif wave_type == 'triangle':
        sig = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
    elif wave_type == 'square_soft':
        # Soft square: sum of odd harmonics with rolloff
        sig = np.zeros_like(t)
        for k in range(1, 8, 2):
            sig += np.sin(2 * np.pi * freq * k * t) / (k ** 1.5)
        sig /= np.max(np.abs(sig))
    
    return sig * env

def make_kick(duration=0.15, sr=SAMPLE_RATE):
    """Simple kick drum."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 150 * np.exp(-8 * t / duration) + 40
    phase = np.cumsum(2 * np.pi * freq / sr)
    sig = np.sin(phase) * np.exp(-5 * t / duration)
    return sig

def make_hihat(duration=0.05, sr=SAMPLE_RATE):
    """Simple hi-hat."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    noise = np.random.randn(len(t))
    env = np.exp(-30 * t / duration)
    return noise * env * 0.3

# --- Scales ---
# A minor pentatonic: A C D E G (good for dark/metal vibes)
A_MINOR_PENTA = [220.0, 261.6, 293.7, 329.6, 392.0,
                 440.0, 523.3, 587.3, 659.3, 784.0,
                 880.0, 1046.5, 1174.7, 1318.5, 1568.0, 1760.0]

# D minor (dorian feel)
D_MINOR = [146.8, 164.8, 174.6, 196.0, 220.0, 233.1, 261.6,
           293.7, 329.6, 349.2, 392.0, 440.0, 466.2, 523.3,
           587.3, 659.3]

def save_wav(filename, samples, sr=SAMPLE_RATE):
    """Save numpy array as 16-bit WAV."""
    samples = samples / (np.max(np.abs(samples)) + 1e-8) * 0.8
    samples_int = (samples * 32767).astype(np.int16)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples_int.tobytes())

# --- Composition Methods ---

def method1_direct_mapping(rule_num, scale, bpm=140, steps=64, width=16):
    """
    Method 1: Direct mapping — active cell = note trigger.
    Each column maps to a scale degree, each row is a time step.
    """
    grid = generate_ca(rule_num, width, steps)
    step_dur = 60.0 / bpm / 4  # 16th note
    note_dur = step_dur * 0.8
    total_dur = steps * step_dur
    output = np.zeros(int(SAMPLE_RATE * total_dur))
    
    for t in range(steps):
        active = np.where(grid[t] == 1)[0]
        for col in active:
            freq = scale[col % len(scale)]
            note = make_note(freq, note_dur, wave_type='triangle')
            start = int(t * step_dur * SAMPLE_RATE)
            end = start + len(note)
            if end <= len(output):
                output[start:end] += note * 0.15
    
    return output, grid

def method2_constrained_rhythm(rule_num, scale, bpm=130, steps=64, width=16):
    """
    Method 2: CA drives rhythm, notes constrained to chord tones.
    Row density controls which chord inversion plays.
    """
    grid = generate_ca(rule_num, width, steps)
    step_dur = 60.0 / bpm / 4
    total_dur = steps * step_dur
    output = np.zeros(int(SAMPLE_RATE * total_dur))
    
    # Chord progression (i - VI - III - VII in A minor)
    chords = [
        [220.0, 261.6, 329.6],   # Am
        [174.6, 220.0, 261.6],   # F
        [261.6, 329.6, 392.0],   # C
        [196.0, 246.9, 293.7],   # G
    ]
    
    for t in range(steps):
        density = np.sum(grid[t]) / width
        chord = chords[(t // 16) % len(chords)]
        active = np.where(grid[t] == 1)[0]
        
        for col in active:
            freq = chord[col % len(chord)]
            # Octave shift based on column position
            octave = 1 + (col // len(chord)) * 0.5
            note = make_note(freq * octave, step_dur * 1.5, 
                           wave_type='sine', decay=1.2)
            start = int(t * step_dur * SAMPLE_RATE)
            end = start + len(note)
            if end <= len(output):
                output[start:end] += note * 0.08
        
        # Kick on high density beats
        if density > 0.4 and t % 4 == 0:
            kick = make_kick()
            start = int(t * step_dur * SAMPLE_RATE)
            end = start + len(kick)
            if end <= len(output):
                output[start:end] += kick * 0.5
        
        # Hi-hat driven by CA
        if grid[t][0] == 1:
            hh = make_hihat()
            start = int(t * step_dur * SAMPLE_RATE)
            end = start + len(hh)
            if end <= len(output):
                output[start:end] += hh * 0.4
    
    return output, grid

def method3_parameter_modulation(rule_num, bpm=120, steps=128, width=16):
    """
    Method 3: CA modulates parameters, not notes directly.
    Fixed bass line, CA controls filter/texture.
    """
    grid = generate_ca(rule_num, width, steps)
    step_dur = 60.0 / bpm / 4
    total_dur = steps * step_dur
    output = np.zeros(int(SAMPLE_RATE * total_dur))
    
    # Bass line (fixed pattern, A minor)
    bass_pattern = [110.0, 110.0, 0, 130.8, 0, 110.0, 146.8, 0,
                    130.8, 130.8, 0, 110.0, 0, 146.8, 164.8, 0]
    
    for t in range(steps):
        density = np.sum(grid[t]) / width
        bass_freq = bass_pattern[t % len(bass_pattern)]
        
        if bass_freq > 0:
            # CA density controls timbre: low density = sine, high = harsh
            if density < 0.3:
                wave_t = 'sine'
                dec = 1.5
            elif density < 0.6:
                wave_t = 'triangle'
                dec = 1.0
            else:
                wave_t = 'square_soft'
                dec = 0.6
            
            note = make_note(bass_freq, step_dur * 2, wave_type=wave_t, decay=dec)
            start = int(t * step_dur * SAMPLE_RATE)
            end = start + len(note)
            if end <= len(output):
                output[start:end] += note * 0.3
        
        # Ambient pad from CA - use right half density
        right_density = np.sum(grid[t][width//2:]) / (width//2)
        if right_density > 0.3:
            pad_freq = 440.0 * (1 + right_density * 0.5)
            pad = make_note(pad_freq, step_dur * 4, wave_type='sine', decay=2.0)
            start = int(t * step_dur * SAMPLE_RATE)
            end = start + len(pad)
            if end <= len(output):
                output[start:end] += pad * 0.05 * right_density
        
        # Percussion from left edge
        if grid[t][0] == 1 and t % 2 == 0:
            kick = make_kick()
            start = int(t * step_dur * SAMPLE_RATE)
            end = start + len(kick)
            if end <= len(output):
                output[start:end] += kick * 0.4
        
        if grid[t][-1] == 1:
            hh = make_hihat()
            start = int(t * step_dur * SAMPLE_RATE)
            end = start + len(hh)
            if end <= len(output):
                output[start:end] += hh * 0.3
    
    return output, grid


def visualize_grid(grid, filename, max_width=64, max_rows=32):
    """Save ASCII art of CA grid."""
    h, w = grid.shape
    rows = min(h, max_rows)
    cols = min(w, max_width)
    lines = []
    for t in range(rows):
        line = ''.join('█' if grid[t][c] else '·' for c in range(cols))
        lines.append(line)
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))
    return '\n'.join(lines[:16])  # return preview


if __name__ == '__main__':
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(outdir, exist_ok=True)
    
    print("=== Generating CA Music Demos ===\n")
    
    # --- Rule 30 ---
    print("▸ Rule 30 — Direct Mapping (A minor pentatonic, 140 BPM)")
    audio, grid = method1_direct_mapping(30, A_MINOR_PENTA, bpm=140, steps=64)
    save_wav(os.path.join(outdir, 'rule30_direct.wav'), audio)
    viz = visualize_grid(grid, os.path.join(outdir, 'rule30_grid.txt'))
    print(f"  Grid preview:\n{viz}\n")
    
    print("▸ Rule 30 — Chord-Constrained Rhythm (130 BPM)")
    audio, grid = method2_constrained_rhythm(30, A_MINOR_PENTA, bpm=130, steps=64)
    save_wav(os.path.join(outdir, 'rule30_constrained.wav'), audio)
    
    print("▸ Rule 30 — Parameter Modulation (120 BPM)")
    audio, grid = method3_parameter_modulation(30, bpm=120, steps=128)
    save_wav(os.path.join(outdir, 'rule30_modulation.wav'), audio)
    
    # --- Rule 110 ---
    print("\n▸ Rule 110 — Direct Mapping (A minor pentatonic, 140 BPM)")
    audio, grid = method1_direct_mapping(110, A_MINOR_PENTA, bpm=140, steps=64)
    save_wav(os.path.join(outdir, 'rule110_direct.wav'), audio)
    viz = visualize_grid(grid, os.path.join(outdir, 'rule110_grid.txt'))
    print(f"  Grid preview:\n{viz}\n")
    
    print("▸ Rule 110 — Chord-Constrained Rhythm (130 BPM)")
    audio, grid = method2_constrained_rhythm(110, A_MINOR_PENTA, bpm=130, steps=64)
    save_wav(os.path.join(outdir, 'rule110_constrained.wav'), audio)
    
    print("▸ Rule 110 — Parameter Modulation (120 BPM)")
    audio, grid = method3_parameter_modulation(110, bpm=120, steps=128)
    save_wav(os.path.join(outdir, 'rule110_modulation.wav'), audio)
    
    print(f"\n=== Done! 6 files in {outdir} ===")
