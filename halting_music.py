"""
Halting Music — The Sound of (Un)decidability

Multiple Collatz sequences run simultaneously, each a voice.
- Short-lived sequences: percussive bursts
- Long-lived sequences: melodic arcs rising and falling
- The "highway" to 4→2→1: a shared cadence, like gravity

The Collatz conjecture says every sequence halts. We don't know why.
That uncertainty — is this voice about to end? — is the tension.

Also includes a section with Busy Beaver candidates:
small Turing machines that run for absurdly long before halting (or not).
These become drones underneath the Collatz voices.
"""

import numpy as np
import wave
import os
import struct

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


# ── Collatz sequence generator ──
def collatz_sequence(n):
    """Generate the full Collatz sequence from n down to 1."""
    seq = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        seq.append(n)
    return seq


# ── Voice: one Collatz sequence as a musical phrase ──
def render_voice(seq, start_time, time_per_step, sr=SAMPLE_RATE):
    """
    Render a Collatz sequence as audio.
    Returns (samples, duration_in_samples, start_sample).
    
    Mapping:
    - Value → frequency (log scale, so doublings = octaves)
    - Step → time
    - Odd steps (3n+1, going UP) → brighter timbre
    - Even steps (n/2, going DOWN) → darker timbre
    - Final descent to 1 → fade + pitch drop
    """
    duration = len(seq) * time_per_step
    total_samples = int(duration * sr)
    start_sample = int(start_time * sr)
    output = np.zeros(total_samples)
    
    if len(seq) < 2:
        return output, total_samples, start_sample
    
    # Normalize values for frequency mapping
    max_val = max(seq)
    min_val = 1
    
    # Base frequency range: 55 Hz (A1) to 1760 Hz (A6) — 5 octaves
    freq_lo, freq_hi = 55.0, 1760.0
    
    phase = 0.0
    
    for step_i in range(len(seq)):
        val = seq[step_i]
        
        # Log-scale frequency mapping
        if max_val > 1:
            t_val = np.log2(val) / np.log2(max_val)
        else:
            t_val = 0
        freq = freq_lo * (freq_hi / freq_lo) ** t_val
        
        # Determine if this was an up-step (odd) or down-step (even)
        is_up = step_i > 0 and seq[step_i] > seq[step_i - 1]
        
        # Timbre: up-steps get harmonics, down-steps are purer
        harm_mix = 0.4 if is_up else 0.1
        
        # Render this step's audio
        step_start = int(step_i * time_per_step * sr)
        step_end = int((step_i + 1) * time_per_step * sr)
        step_end = min(step_end, total_samples)
        
        for i in range(step_start, step_end):
            # Smooth frequency transition within step
            local_t = (i - step_start) / max(step_end - step_start, 1)
            
            # Next step frequency for interpolation
            if step_i + 1 < len(seq):
                next_val = seq[step_i + 1]
                t_next = np.log2(next_val) / np.log2(max_val) if max_val > 1 else 0
                freq_next = freq_lo * (freq_hi / freq_lo) ** t_next
                current_freq = freq + (freq_next - freq) * local_t * 0.3
            else:
                current_freq = freq * (1 - local_t * 0.5)  # Final note fades down
            
            phase += 2 * np.pi * current_freq / sr
            
            # Waveform
            s = np.sin(phase)
            s += np.sin(phase * 2) * harm_mix * 0.5
            s += np.sin(phase * 3) * harm_mix * 0.25
            s += np.sin(phase * 5) * harm_mix * 0.1
            
            # Amplitude envelope: slight attack, sustain, quick release at step boundary
            env = 1.0
            attack = int(0.005 * sr)
            release = int(0.01 * sr)
            pos_in_step = i - step_start
            step_len = step_end - step_start
            if pos_in_step < attack:
                env = pos_in_step / attack
            elif pos_in_step > step_len - release:
                env = (step_len - pos_in_step) / release
            
            # Final steps (4→2→1): fade out
            if step_i >= len(seq) - 3:
                remaining_frac = (len(seq) - step_i) / 3.0
                env *= remaining_frac
            
            output[i] = s * env
    
    # Overall envelope
    fade_in = int(0.02 * sr)
    fade_out = int(0.1 * sr)
    if fade_in < total_samples:
        output[:fade_in] *= np.linspace(0, 1, fade_in)
    if fade_out < total_samples:
        output[-fade_out:] *= np.linspace(1, 0, fade_out)
    
    return output, total_samples, start_sample


# ── Piece I: Collatz Ensemble ──
def collatz_ensemble(duration=40.0):
    """
    20 Collatz sequences enter one by one.
    Short sequences are percussive flickers.
    Long sequences are melodic arcs.
    All end on the same cadence: 4→2→1.
    
    Selected starting numbers for musical variety:
    - Some short (< 20 steps): 3, 6, 10
    - Some medium (20-100 steps): 27, 54, 73, 97
    - Some long (100+ steps): 871, 6171, 77031
    """
    # Curated starting numbers — chosen for sequence length variety
    starters = [
        # Short (percussive)
        3, 5, 7, 10, 16,
        # Medium (melodic phrases)  
        27, 54, 73, 97, 113,
        # Long (extended arcs)
        327, 649, 871, 1161,
        # Very long (epic journeys)
        6171, 77031, 837799,
    ]
    
    # Pre-compute sequences
    sequences = [(n, collatz_sequence(n)) for n in starters]
    sequences.sort(key=lambda x: len(x[1]))  # short first
    
    print(f"  Sequences: {len(sequences)} voices")
    for n, seq in sequences:
        print(f"    n={n}: {len(seq)} steps, peak={max(seq)}")
    
    total_samples = int(duration * SAMPLE_RATE)
    output_l = np.zeros(total_samples)
    output_r = np.zeros(total_samples)
    
    # Stagger entries: voices enter over the first 60% of the piece
    entry_window = duration * 0.6
    
    for idx, (n, seq) in enumerate(sequences):
        # Entry time: spread across the window
        entry_time = (idx / len(sequences)) * entry_window
        
        # Time per step: longer sequences get faster steps
        # So the piece doesn't last forever
        max_duration_for_voice = duration - entry_time - 0.5
        time_per_step = min(0.15, max_duration_for_voice / len(seq))
        time_per_step = max(0.02, time_per_step)
        
        voice, voice_len, start_sample = render_voice(
            seq, entry_time, time_per_step
        )
        
        # Amplitude: scale by 1/sqrt(n_voices) to avoid clipping
        amp = 0.3 / np.sqrt(len(sequences))
        
        # Pan: spread voices across stereo field
        pan = (idx / max(len(sequences) - 1, 1))  # 0=left, 1=right
        pan_l = np.cos(pan * np.pi / 2)
        pan_r = np.sin(pan * np.pi / 2)
        
        end_sample = min(start_sample + voice_len, total_samples)
        actual_len = end_sample - start_sample
        if actual_len > 0 and actual_len <= len(voice):
            output_l[start_sample:end_sample] += voice[:actual_len] * amp * pan_l
            output_r[start_sample:end_sample] += voice[:actual_len] * amp * pan_r
    
    # Add a very low drone that represents "the conjecture" —
    # the unproven assumption that all these voices WILL halt
    drone = np.zeros(total_samples)
    drone_phase = 0.0
    drone_freq = 36.7  # D1, below the Collatz voices
    for i in range(total_samples):
        t = i / total_samples
        # Drone gets slightly louder as more voices are active
        active_fraction = min(t / 0.6, 1.0)
        drone_amp = 0.08 * active_fraction
        
        drone_phase += 2 * np.pi * drone_freq / SAMPLE_RATE
        # Perfect fifth above, very quiet — the "resolution" that may or may not come
        drone[i] = (np.sin(drone_phase) * drone_amp + 
                    np.sin(drone_phase * 1.5) * drone_amp * 0.3)
    
    output_l += drone * 0.7
    output_r += drone * 0.7
    
    # Master fade
    fade = int(SAMPLE_RATE * 3.0)
    output_l[:int(SAMPLE_RATE*1.5)] *= np.linspace(0, 1, int(SAMPLE_RATE*1.5))
    output_r[:int(SAMPLE_RATE*1.5)] *= np.linspace(0, 1, int(SAMPLE_RATE*1.5))
    output_l[-fade:] *= np.linspace(1, 0, fade)
    output_r[-fade:] *= np.linspace(1, 0, fade)
    
    mono = output_l * 0.5 + output_r * 0.5
    return mono, output_l, output_r


# ── Piece II: Single Journey — n=27 ──
def single_journey(n=27, time_per_step=0.25):
    """
    One number's complete Collatz journey, rendered in detail.
    n=27 is famous: it takes 111 steps and peaks at 9232 before
    collapsing to 1. A microcosm of the conjecture's mystery.
    
    More expressive rendering: pitch + rhythm + timbre all follow the sequence.
    """
    seq = collatz_sequence(n)
    print(f"  n={n}: {len(seq)} steps, peak={max(seq)}")
    
    duration = len(seq) * time_per_step + 2.0  # extra for reverb tail
    total_samples = int(duration * SAMPLE_RATE)
    output = np.zeros(total_samples)
    
    max_val = max(seq)
    
    phase = 0.0
    
    for step_i in range(len(seq)):
        val = seq[step_i]
        prev_val = seq[step_i - 1] if step_i > 0 else val
        
        # Frequency: log mapping
        log_val = np.log2(max(val, 1))
        log_max = np.log2(max_val)
        freq = 65.0 * 2 ** (log_val / log_max * 4)  # 4 octaves range
        
        # Rhythm: odd steps (multiplication) get longer notes
        is_odd_step = prev_val % 2 == 1 and step_i > 0
        note_duration = time_per_step * (1.2 if is_odd_step else 0.9)
        
        step_start = int(step_i * time_per_step * SAMPLE_RATE)
        step_end = min(int(step_start + note_duration * SAMPLE_RATE), total_samples)
        
        # Velocity/loudness: proportional to how dramatic the change is
        if step_i > 0:
            ratio = val / prev_val
            drama = min(abs(np.log2(ratio)), 2.0) / 2.0
        else:
            drama = 0.5
        amplitude = 0.3 + drama * 0.5
        
        for i in range(step_start, step_end):
            local_t = (i - step_start) / max(step_end - step_start, 1)
            
            phase += 2 * np.pi * freq / SAMPLE_RATE
            
            # Waveform: rising notes get saw-like, falling notes get sine-like
            if is_odd_step:
                # Brighter: more harmonics
                s = np.sin(phase) * 0.5
                s += np.sin(phase * 2) * 0.25
                s += np.sin(phase * 3) * 0.15
                s += np.sin(phase * 4) * 0.08
                s += np.sin(phase * 5) * 0.04
            else:
                # Purer
                s = np.sin(phase) * 0.8
                s += np.sin(phase * 2) * 0.1
            
            # ADSR envelope
            attack_t = 0.01
            release_t = 0.05
            note_len = (step_end - step_start) / SAMPLE_RATE
            
            if local_t < attack_t / note_len:
                env = local_t * note_len / attack_t
            elif local_t > 1 - release_t / note_len:
                env = (1 - local_t) * note_len / release_t
            else:
                env = 1.0
            env = min(env, 1.0)
            
            # The 4→2→1 ending: special treatment
            if step_i >= len(seq) - 4:
                remaining = (len(seq) - step_i) / 4.0
                env *= remaining ** 0.5
            
            output[i] += s * env * amplitude
    
    # Simple reverb: delay + feedback
    delay_samples = int(0.12 * SAMPLE_RATE)
    reverb = np.zeros(total_samples)
    for d, gain in [(delay_samples, 0.3), (delay_samples*2, 0.15), 
                     (int(delay_samples*1.7), 0.1)]:
        if d < total_samples:
            reverb[d:] += output[:total_samples-d] * gain
    output += reverb
    
    # Fade
    fade_in = int(0.5 * SAMPLE_RATE)
    fade_out = int(2.0 * SAMPLE_RATE)
    output[:fade_in] *= np.linspace(0, 1, fade_in)
    output[-fade_out:] *= np.linspace(1, 0, fade_out)
    
    return output


# ── Piece III: Density — many sequences as texture ──
def collatz_density(duration=30.0):
    """
    Hundreds of Collatz sequences running simultaneously as granular texture.
    Not individual voices but a statistical cloud.
    
    At each time step, we track: how many are still running, the mean value,
    the variance. These aggregate properties drive the sound.
    
    It's the difference between hearing one raindrop and hearing rain.
    """
    # Start 500 sequences from consecutive numbers
    n_start = 100
    n_count = 500
    
    all_seqs = [collatz_sequence(n) for n in range(n_start, n_start + n_count)]
    max_len = max(len(s) for s in all_seqs)
    
    print(f"  {n_count} sequences from {n_start} to {n_start + n_count - 1}")
    print(f"  Longest: {max_len} steps")
    
    # Compute aggregate statistics at each step
    steps = max_len
    active_count = np.zeros(steps)
    mean_val = np.zeros(steps)
    std_val = np.zeros(steps)
    max_val_at_step = np.zeros(steps)
    
    for step in range(steps):
        vals = []
        for seq in all_seqs:
            if step < len(seq):
                vals.append(seq[step])
        if vals:
            active_count[step] = len(vals)
            mean_val[step] = np.mean(vals)
            std_val[step] = np.std(vals)
            max_val_at_step[step] = max(vals)
    
    # Map steps to time
    time_per_step = duration / steps
    total_samples = int(duration * SAMPLE_RATE)
    output = np.zeros(total_samples)
    
    phase1 = 0.0
    phase2 = 0.0
    phase3 = 0.0
    noise_state = 0.5
    
    # Normalize statistics
    max_mean = np.max(mean_val) or 1
    max_std = np.max(std_val) or 1
    
    for i in range(total_samples):
        t = i / SAMPLE_RATE
        step = min(int(t / time_per_step), steps - 1)
        
        # How many sequences are still alive → density/loudness
        density = active_count[step] / n_count
        
        # Mean value → center frequency
        center_freq = 80 + (mean_val[step] / max_mean) * 800
        
        # Standard deviation → bandwidth/noise
        spread = std_val[step] / max_std
        
        phase1 += 2 * np.pi * center_freq / SAMPLE_RATE
        phase2 += 2 * np.pi * (center_freq * 1.5) / SAMPLE_RATE  # fifth
        phase3 += 2 * np.pi * (center_freq * 2.0) / SAMPLE_RATE  # octave
        
        # Clean tone: the mean
        tone = np.sin(phase1) * 0.5 + np.sin(phase2) * 0.2 + np.sin(phase3) * 0.1
        
        # Noise component: proportional to spread
        # Simple noise via logistic map (chaos as sound, fitting!)
        noise_state = 3.99 * noise_state * (1 - noise_state)
        noise = (noise_state - 0.5) * 2
        
        # Mix: more spread → more noise
        sample = tone * (1 - spread * 0.6) + noise * spread * 0.4
        
        # Amplitude follows density
        amp = density ** 0.5 * 0.7  # sqrt so it doesn't get too quiet
        
        output[i] = sample * amp
    
    # Fade
    fade = int(SAMPLE_RATE * 2.0)
    output[:fade] *= np.linspace(0, 1, fade)
    output[-fade:] *= np.linspace(1, 0, fade)
    
    return output


if __name__ == '__main__':
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(outdir, exist_ok=True)
    
    print("=== Halting Music — The Sound of (Un)decidability ===\n")
    
    print("I. Collatz Ensemble — 17 voices, staggered entry")
    mono, left, right = collatz_ensemble(duration=40.0)
    save_wav(os.path.join(outdir, 'halting_1_ensemble.wav'), mono)
    save_wav_stereo(os.path.join(outdir, 'halting_1_ensemble_stereo.wav'), left, right)
    print("   ✓ saved\n")
    
    print("II. Single Journey — n=27 (the famous 111-step odyssey)")
    audio = single_journey(n=27, time_per_step=0.22)
    save_wav(os.path.join(outdir, 'halting_2_journey_27.wav'), audio)
    print("   ✓ saved\n")
    
    print("III. Density — 500 sequences as granular texture")
    audio = collatz_density(duration=30.0)
    save_wav(os.path.join(outdir, 'halting_3_density.wav'), audio)
    print("   ✓ saved\n")
    
    print("=== Every voice halts. We just can't prove it. ===")
