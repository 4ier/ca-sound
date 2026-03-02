"""
Lambda Calculus — The Sound of Computation Itself

Three pieces exploring lambda calculus reduction as music:

1. Church Numerals (45s) — Numbers 0-10 built from pure abstraction.
   Each numeral applies its function n times; more applications = richer
   harmonics building up like overtones. Zero is silence becoming sound.

2. Y Combinator Unfold (40s) — The fixed-point combinator's self-application
   creates infinite recursive descent. Each level of β-reduction maps to
   a frequency that's a ratio of the previous (golden ratio spiral).
   The recursion unfolds like a fractal zoom.

3. SKI Calculus (50s) — The three combinators S, K, I as three instruments.
   Evaluate several SKI expressions step by step. Reduction = simplification =
   the music thins out until only the result remains. Complexity collapsing
   into clarity.

Concept: Lambda calculus is the most minimal model of computation.
No tape, no states — just abstraction and application. The music should
feel like pure thought crystallizing into form.
"""

import struct
import math
import os
import numpy as np

RATE = 44100

def write_wav(filename, data, sr=RATE, channels=1):
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    data = np.clip(data, -1.0, 1.0)
    pcm = (data * 32767).astype(np.int16)
    raw = pcm.tobytes()
    with open(filename, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(raw)))
        f.write(b'WAVEfmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, channels, sr, sr*channels*2, channels*2, 16))
        f.write(b'data')
        f.write(struct.pack('<I', len(raw)))
        f.write(raw)

PHI = (1 + math.sqrt(5)) / 2  # golden ratio


# --- Piece 1: Church Numerals ---
# Church numeral n = λf.λx. f(f(...f(x)...)) with n applications
# 0 = λf.λx. x          (silence → a single seed tone)
# 1 = λf.λx. f(x)       (one harmonic added)
# n = n harmonics         (richer and richer)

def generate_church_numerals(dur=45):
    print("  Generating Church Numerals...")
    N = int(dur * RATE)
    out = np.zeros(N)
    t = np.arange(N) / RATE
    
    base_freq = 130.81  # C3
    
    # Each numeral gets a segment
    numerals = list(range(11))  # 0 through 10
    seg_dur = dur / len(numerals)
    
    for n in numerals:
        i0 = int(n * seg_dur * RATE)
        i1 = min(int((n + 1) * seg_dur * RATE), N)
        t_seg = t[i0:i1]
        t_local = np.arange(i1 - i0) / RATE
        sd = seg_dur
        
        # ADSR with longer attack for lower numerals
        attack = 0.3 * (1 - n/10) + 0.05
        env = np.zeros(i1 - i0)
        for j, tl in enumerate(t_local):
            if tl < attack:
                env[j] = tl / attack
            elif tl < attack + 0.1:
                env[j] = 1.0 - 0.3 * (tl - attack) / 0.1
            elif tl < sd - 0.2:
                env[j] = 0.7
            else:
                env[j] = 0.7 * max(0, (sd - tl) / 0.2)
        
        if n == 0:
            # Church 0: the identity on x, f is never applied
            # A single quiet seed tone emerging from silence
            out[i0:i1] += 0.1 * env * np.sin(2 * np.pi * base_freq * t_seg)
        else:
            # Church n: f applied n times → n harmonics
            amp_total = 0.3
            signal = np.zeros(i1 - i0)
            for h in range(1, n + 1):
                # Each application of f adds a harmonic
                # Higher applications are quieter (like overtone series)
                h_amp = 1.0 / h
                # Slight detuning for richness (applications aren't perfect)
                detune = 1.0 + 0.001 * (h - 1) * math.sin(h * 0.7)
                signal += h_amp * np.sin(2 * np.pi * base_freq * h * detune * t_seg)
            
            # Normalize
            peak = np.max(np.abs(signal)) + 1e-10
            signal = signal / peak * amp_total
            out[i0:i1] += signal * env
    
    # Crossfade between segments for smoothness
    write_wav('output/lambda_1_church.wav', out)
    print("    → output/lambda_1_church.wav")


# --- Piece 2: Y Combinator Unfold ---
# Y = λf.(λx.f(x x))(λx.f(x x))
# When applied: Y g → g(Y g) → g(g(Y g)) → g(g(g(Y g))) → ...
# Each unfolding = one level deeper. Frequency spirals via golden ratio.

def generate_y_combinator(dur=40):
    print("  Generating Y Combinator Unfold...")
    N = int(dur * RATE)
    L = np.zeros(N)
    R = np.zeros(N)
    t = np.arange(N) / RATE
    
    # Recursion levels
    levels = 20
    base_freq = 440.0  # A4
    
    # Each level starts slightly after the previous (cascading)
    # and its frequency is base * PHI^(-level) (descending spiral)
    # Duration of each level: starts long, gets shorter (acceleration)
    
    level_start = 0.0
    for lv in range(levels):
        # Frequency: golden ratio descent
        freq = base_freq * PHI ** (-lv * 0.5)
        if freq < 30: freq = 30 + (lv % 7) * 20  # wrap around if too low
        
        # Duration shrinks: each level is shorter
        lv_dur = max(0.5, (dur - 5) / levels * (1 - lv / levels * 0.6))
        
        i0 = int(level_start * RATE)
        i1 = min(int((level_start + lv_dur) * RATE), N)
        if i0 >= N: break
        
        t_local = np.arange(i1 - i0) / RATE
        # Envelope: quick attack, long sustain fade
        env = np.exp(-t_local / (lv_dur * 0.6)) * 0.2
        
        # Self-application: the waveform uses itself as modulator
        # Deeper levels = more FM modulation (self-reference)
        mod_depth = min(lv * 0.3, 3.0)
        carrier = np.sin(2 * np.pi * freq * t[i0:i1])
        modulator = np.sin(2 * np.pi * freq * PHI * t[i0:i1])
        signal = np.sin(2 * np.pi * freq * t[i0:i1] + mod_depth * modulator)
        
        # Pan: spiral across stereo field
        pan_angle = lv * math.pi / 7
        lg = math.cos(pan_angle) * 0.5 + 0.5
        rg = math.sin(pan_angle) * 0.5 + 0.5
        
        L[i0:i1] += signal * env * lg
        R[i0:i1] += signal * env * rg
        
        # Overlap: next level starts before this one ends
        level_start += lv_dur * 0.7
    
    # Final: the infinite recursion is represented by a sustained chord
    # that fades to nothing (the computation never terminates)
    final_start = int(min(level_start, dur - 5) * RATE)
    if final_start < N:
        t_f = np.arange(N - final_start) / RATE
        fd = (N - final_start) / RATE
        env_f = 0.15 * np.exp(-t_f / (fd * 0.4))
        # Chord of golden ratio frequencies
        chord = np.zeros(N - final_start)
        for i in range(5):
            f = base_freq * PHI ** (-i * 0.5)
            chord += np.sin(2 * np.pi * f * t[final_start:]) / (i + 1)
        chord = chord / np.max(np.abs(chord) + 1e-10)
        L[final_start:] += chord * env_f * 0.7
        R[final_start:] += chord * env_f * 0.7
    
    out = np.empty(N * 2)
    out[0::2] = L; out[1::2] = R
    write_wav('output/lambda_2_ycombinator.wav', out, channels=2)
    print("    → output/lambda_2_ycombinator.wav")


# --- Piece 3: SKI Calculus ---
# S, K, I combinators: the simplest Turing-complete system
# S x y z = x z (y z)     -- distribute and apply
# K x y = x               -- discard second
# I x = x                 -- identity
#
# We evaluate several expressions, mapping each combinator to a sound:
# S = bright attack (complex operation)
# K = damped thud (discarding)
# I = pure sine (passing through unchanged)

def generate_ski(dur=50):
    print("  Generating SKI Calculus...")
    N = int(dur * RATE)
    L = np.zeros(N)
    R = np.zeros(N)
    t = np.arange(N) / RATE
    
    # SKI reduction traces (manually computed)
    # Each trace is a list of tokens after each reduction step
    traces = [
        # S K K x = K x (K x) = x  (SKK = I, the identity)
        ['S','K','K','x', '→', 'K','x','·','K','x', '→', 'x'],
        # S I I x = I x (I x) = x x  (self-application)
        ['S','I','I','x', '→', 'I','x','·','I','x', '→', 'x','x'],
        # S (K S) K = <combinator B>  (composition)
        ['S','(','K','S',')','K','x','y','z', '→', 'K','S','x','·','K','x','·','y','z',
         '→', 'S','·','x','·','y','z', '→', 'x','z','·','y','z'],
        # K I x y = I y = y  (flip and apply identity)
        ['K','I','x','y', '→', 'I','y', '→', 'y'],
    ]
    
    # Sound for each token type
    S_FREQ = 587.33   # D5 - bright
    K_FREQ = 146.83   # D3 - low thud
    I_FREQ = 293.66   # D4 - pure middle
    ARROW_FREQ = 0    # silence (pause)
    VAR_FREQ = 220.0  # A3 for variables
    
    def token_sound(tok, t_arr, t_global):
        """Generate sound for a single token."""
        sd = len(t_arr) / RATE
        t_l = t_arr
        
        if tok == 'S':
            # Bright: FM synthesis with fast attack
            env = np.exp(-t_l / (sd * 0.5)) * 0.3
            mod = np.sin(2 * np.pi * S_FREQ * 3 * t_global)
            return env * np.sin(2 * np.pi * S_FREQ * t_global + 2 * mod)
        elif tok == 'K':
            # Damped thud: low with noise-like attack
            env = np.exp(-t_l / (sd * 0.3)) * 0.25
            s = np.sin(2 * np.pi * K_FREQ * t_global)
            s += 0.3 * np.sin(2 * np.pi * K_FREQ * 2.01 * t_global)  # inharmonic
            return env * s
        elif tok == 'I':
            # Pure sine, gentle
            env_arr = np.zeros_like(t_l)
            for j, tl in enumerate(t_l):
                if tl < 0.05: env_arr[j] = tl / 0.05
                elif tl < sd - 0.1: env_arr[j] = 1.0
                else: env_arr[j] = max(0, (sd - tl) / 0.1)
            return 0.25 * env_arr * np.sin(2 * np.pi * I_FREQ * t_global)
        elif tok == '→':
            return np.zeros_like(t_l)
        elif tok in ('(', ')', '·'):
            # Tiny click
            env = np.exp(-t_l / 0.01) * 0.05
            return env * np.sin(2 * np.pi * 1000 * t_global)
        else:
            # Variable: warm sine at variable pitch
            # Different variables get different pitches
            pitch_offset = (ord(tok[0]) - ord('x')) * 50
            freq = VAR_FREQ + pitch_offset
            env = np.exp(-t_l / (sd * 0.6)) * 0.2
            return env * np.sin(2 * np.pi * freq * t_global)
    
    # Layout: each trace gets a portion of the duration
    trace_dur = dur / len(traces)
    
    for ti, trace in enumerate(traces):
        trace_start = ti * trace_dur
        # Count meaningful tokens (skip arrows for timing)
        tokens = [tok for tok in trace]
        n_tokens = len(tokens)
        tok_dur = min(trace_dur / max(n_tokens, 1), 0.8)
        
        for si, tok in enumerate(tokens):
            ts = trace_start + si * tok_dur
            i0 = int(ts * RATE)
            i1 = min(int((ts + tok_dur) * RATE), N)
            if i0 >= N or i1 <= i0: continue
            
            t_local = np.arange(i1 - i0) / RATE
            signal = token_sound(tok, t_local, t[i0:i1])
            
            # As reduction progresses, sound moves from wide stereo to center
            progress = si / max(n_tokens - 1, 1)
            pan = 0.5 * (1 - progress)  # starts wide, ends center
            if si % 2 == 0:
                L[i0:i1] += signal * (0.5 + pan)
                R[i0:i1] += signal * (0.5 - pan * 0.5)
            else:
                L[i0:i1] += signal * (0.5 - pan * 0.5)
                R[i0:i1] += signal * (0.5 + pan)
    
    out = np.empty(N * 2)
    out[0::2] = L; out[1::2] = R
    write_wav('output/lambda_3_ski.wav', out, channels=2)
    print("    → output/lambda_3_ski.wav")


if __name__ == '__main__':
    print("Lambda Calculus — The Sound of Computation Itself")
    print("=" * 50)
    generate_church_numerals()
    generate_y_combinator()
    generate_ski()
    print("\nDone.")
