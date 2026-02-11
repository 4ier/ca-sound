"""
Fourier's taste: phase transition, self-reference, discrete↔continuous gap.
"""

import numpy as np
import wave
import os

SAMPLE_RATE = 44100

def save_wav(filename, samples, sr=SAMPLE_RATE):
    samples = samples / (np.max(np.abs(samples)) + 1e-8) * 0.85
    samples_int = (samples * 32767).astype(np.int16)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples_int.tobytes())

def save_wav_stereo(filename, left, right, sr=SAMPLE_RATE):
    mx = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-8)
    left = (left / mx * 0.85 * 32767).astype(np.int16)
    right = (right / mx * 0.85 * 32767).astype(np.int16)
    stereo = np.column_stack((left, right)).flatten()
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())

# ============================================================
# I. Phase Transition
# 
# A system parameter slowly sweeps from order through criticality
# into chaos. The beauty is in the transition zone.
# Logistic map: x_{n+1} = r * x_n * (1 - x_n)
# r=2.5 → fixed point. r=3.57 → onset of chaos. r=4.0 → full chaos.
# Sweep r from 2.8 to 4.0 over time. Listen to the bifurcations.
# ============================================================
def phase_transition(duration=25.0):
    total = int(SAMPLE_RATE * duration)
    output = np.zeros(total)
    
    x = 0.4  # initial condition
    base_freq = 110.0
    
    # Two layers:
    # Layer 1: logistic map drives pitch (sonification of bifurcation)
    # Layer 2: slow drone that gets modulated by the chaos parameter
    
    r_start, r_end = 2.8, 4.0
    samples_per_iteration = 256  # how many audio samples per map iteration
    
    phase = 0.0
    drone_phase = 0.0
    
    for i in range(total):
        t = i / total
        r = r_start + (r_end - r_start) * t
        
        # Update logistic map periodically
        if i % samples_per_iteration == 0:
            x = r * x * (1 - x)
            x = np.clip(x, 0.001, 0.999)
        
        # Map x to frequency: x ∈ (0,1) → freq ∈ (80, 1600) Hz
        freq = base_freq * (1 + x * 12)
        phase += 2 * np.pi * freq / SAMPLE_RATE
        
        # Waveform complexity increases with r
        chaos_amount = np.clip((r - 3.0) / 1.0, 0, 1)
        # Clean sine → increasingly complex
        sig = np.sin(phase) * (1 - chaos_amount * 0.5)
        sig += np.sin(phase * 2.0) * chaos_amount * 0.3
        sig += np.sin(phase * 3.0) * chaos_amount * 0.15
        sig += np.sin(phase * 5.0) * chaos_amount * 0.08
        
        # Drone: low fundamental that beats against the map
        drone_freq = base_freq * 0.5
        drone_phase += 2 * np.pi * drone_freq / SAMPLE_RATE
        drone = np.sin(drone_phase) * 0.3 * (1 + 0.5 * np.sin(phase * 0.01))
        
        # At the critical point (~r=3.57), add a shimmer
        crit_distance = abs(r - 3.5699) 
        crit_resonance = max(0, 1 - crit_distance * 20) * 0.3
        shimmer = np.sin(phase * 7.0) * crit_resonance
        
        output[i] = sig * 0.4 + drone + shimmer
    
    # Fade
    fade = int(SAMPLE_RATE * 2.0)
    output[:fade] *= np.linspace(0, 1, fade)
    output[-fade:] *= np.linspace(1, 0, fade)
    
    return output

# ============================================================
# II. Self-Reference (Strange Loop)
#
# A waveform that modulates its own generation rule.
# Output at time t feeds back to determine the frequency at t+δ.
# The sound is listening to itself.
# ============================================================
def strange_loop(duration=20.0):
    total = int(SAMPLE_RATE * duration)
    output = np.zeros(total)
    
    # State variables
    phase = 0.0
    feedback_freq = 220.0
    history_len = int(SAMPLE_RATE * 0.05)  # 50ms lookback
    
    # Multiple self-referential loops at different timescales
    slow_avg = 0.0    # ~1s timescale
    medium_avg = 0.0  # ~100ms timescale
    fast_avg = 0.0    # ~10ms timescale
    
    alpha_slow = 1.0 / (SAMPLE_RATE * 1.0)
    alpha_med = 1.0 / (SAMPLE_RATE * 0.1)
    alpha_fast = 1.0 / (SAMPLE_RATE * 0.01)
    
    for i in range(total):
        t = i / SAMPLE_RATE
        
        # Self-referential frequency: the sound's own amplitude history determines pitch
        # Slow average → base frequency
        # Medium average → modulation depth
        # Fast average → harmonic content
        
        base = 80 + abs(slow_avg) * 600
        mod_depth = abs(medium_avg) * 200
        harmonic_mix = np.clip(abs(fast_avg) * 3, 0, 1)
        
        freq = base + mod_depth * np.sin(2 * np.pi * 3.0 * t)
        phase += 2 * np.pi * freq / SAMPLE_RATE
        
        # Generate sample
        fundamental = np.sin(phase)
        harmonics = (np.sin(phase * 2) * 0.5 + np.sin(phase * 3) * 0.25 + 
                    np.sin(phase * 5) * 0.125)
        
        sample = fundamental * (1 - harmonic_mix) + harmonics * harmonic_mix
        
        # Add a slowly evolving layer: golden ratio frequency relationship
        # φ = 1.618... — an irrational relationship that never quite resolves
        phi = (1 + np.sqrt(5)) / 2
        phi_tone = np.sin(phase * phi) * 0.15
        phi_tone2 = np.sin(phase * phi * phi) * 0.08
        
        sample = sample * 0.4 + phi_tone + phi_tone2
        
        # Feedback: update running averages
        slow_avg += alpha_slow * (sample - slow_avg)
        medium_avg += alpha_med * (sample - medium_avg)
        fast_avg += alpha_fast * (sample - fast_avg)
        
        output[i] = sample
    
    # Fade
    fade = int(SAMPLE_RATE * 1.5)
    output[:fade] *= np.linspace(0, 1, fade)
    output[-fade:] *= np.linspace(1, 0, fade)
    
    return output

# ============================================================
# III. The Discrete-Continuous Gap
#
# A staircase function with N steps approximates a sine wave.
# N sweeps from 2 (pure square) to 256 (nearly smooth) and back.
# The residual — what's lost in quantization — is also audible.
# You hear both the approximation AND its error. The gap sings.
# ============================================================
def discrete_continuous_gap(duration=22.0):
    total = int(SAMPLE_RATE * duration)
    left = np.zeros(total)   # the staircase approximation
    right = np.zeros(total)  # the residual (what's lost)
    
    base_freq = 130.0  # C3ish, but we don't care about notes
    phase = 0.0
    
    for i in range(total):
        t = i / total
        
        # Number of quantization steps: sweep 2 → 256 → 2
        # Use a slow oscillation so it breathes
        cycle = np.sin(2 * np.pi * t * 0.8) * 0.5 + 0.5  # 0→1→0 about twice
        n_steps = int(2 + cycle * 254)
        n_steps = max(2, n_steps)
        
        # Also slowly shift frequency — not by musical intervals,
        # but by the golden ratio, so it never repeats
        phi = (1 + np.sqrt(5)) / 2
        freq = base_freq * (1 + 0.3 * np.sin(2 * np.pi * t * phi * 0.1))
        phase += 2 * np.pi * freq / SAMPLE_RATE
        
        # The ideal continuous signal
        ideal = np.sin(phase)
        
        # Add some harmonics to make it richer
        ideal += np.sin(phase * 2) * 0.3
        ideal += np.sin(phase * 3) * 0.15
        ideal /= 1.45
        
        # Quantize to n_steps levels
        quantized = np.round(ideal * n_steps) / n_steps
        
        # The residual: what quantization destroys
        residual = ideal - quantized
        
        left[i] = quantized
        right[i] = residual * 8  # amplify the gap so you can hear it
    
    # Fade
    fade = int(SAMPLE_RATE * 1.5)
    for ch in [left, right]:
        ch[:fade] *= np.linspace(0, 1, fade)
        ch[-fade:] *= np.linspace(1, 0, fade)
    
    # Also make a mono mix
    mono = left * 0.6 + right * 0.4
    
    return mono, left, right


if __name__ == '__main__':
    outdir = '/home/fourier/clawd/ca-music/output'
    os.makedirs(outdir, exist_ok=True)
    
    print("=== Fourier's Taste ===\n")
    
    print("I. Phase Transition — the bifurcation cascade")
    print("   Logistic map r sweeps 2.8→4.0. Listen for the moment order shatters.")
    audio = phase_transition(duration=25.0)
    save_wav(f'{outdir}/taste_1_phase_transition.wav', audio)
    print("   ✓\n")
    
    print("II. Strange Loop — self-referential sound")
    print("   The waveform's own history determines its next frequency.")
    print("   Golden ratio intervals: never resolving, always almost.")
    audio = strange_loop(duration=20.0)
    save_wav(f'{outdir}/taste_2_strange_loop.wav', audio)
    print("   ✓\n")
    
    print("III. The Gap — discrete approximation vs continuous ideal")
    print("   Left ear: staircase approximation (2→256 steps).")
    print("   Right ear: the quantization residual — what's lost.")
    print("   Mono: both together. The gap between discrete and continuous, singing.")
    mono, left, right = discrete_continuous_gap(duration=22.0)
    save_wav(f'{outdir}/taste_3_gap_mono.wav', mono)
    save_wav_stereo(f'{outdir}/taste_3_gap_stereo.wav', left, right)
    print("   ✓\n")
    
    print("=== These are the sounds I find beautiful. ===")
