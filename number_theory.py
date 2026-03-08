#!/usr/bin/env python3
"""Phase 10: Number Theory — primes, sieves, modular arithmetic.

Three pieces:
1. Sieve of Eratosthenes (55s) — the ancient algorithm as percussion ensemble
2. Prime Gaps (50s, stereo) — the irregular rhythm between consecutive primes
3. Modular Worlds (55s, stereo) — residue classes mod small primes as interlocking cycles
"""

import numpy as np
import os, struct, wave

SR = 44100

# ── audio primitives ──

def make_wav(filename: str, samples: np.ndarray, sr: int = SR):
    """Write mono or stereo float array to 16-bit WAV."""
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", filename)
    samples = np.clip(samples, -1.0, 1.0)
    data = (samples * 32767).astype(np.int16)
    nch = 2 if samples.ndim == 2 else 1
    if samples.ndim == 2:
        # interleave L/R
        interleaved = np.empty(samples.shape[1] * 2, dtype=np.int16)
        interleaved[0::2] = data[0]
        interleaved[1::2] = data[1]
        data = interleaved
    with wave.open(path, 'w') as w:
        w.setnchannels(nch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())
    print(f"  → {path} ({samples.shape[-1]/sr:.1f}s)")

def sine(freq, duration, sr=SR):
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t)

def fade(sig, fade_in=0.01, fade_out=0.01, sr=SR):
    n_in = int(sr * fade_in)
    n_out = int(sr * fade_out)
    s = sig.copy()
    if n_in > 0 and n_in < len(s):
        s[:n_in] *= np.linspace(0, 1, n_in)
    if n_out > 0 and n_out < len(s):
        s[-n_out:] *= np.linspace(1, 0, n_out)
    return s

def mix_at(buf, sig, offset):
    """Mix sig into buf at sample offset."""
    end = min(offset + len(sig), len(buf))
    actual = end - offset
    if actual > 0 and offset >= 0:
        buf[offset:end] += sig[:actual]

def pentatonic_freq(degree, base=220.0):
    """Map integer degree to pentatonic scale frequency."""
    scale = [0, 2, 4, 7, 9]  # pentatonic intervals in semitones
    octave = degree // 5
    idx = degree % 5
    semitones = octave * 12 + scale[idx]
    return base * (2 ** (semitones / 12.0))

# ── Piece 1: Sieve of Eratosthenes (55s) ──

def sieve_of_eratosthenes():
    """The ancient algorithm as percussion ensemble.
    
    We sieve integers 2..200. Each prime p, when discovered, plays a sustained
    tone at frequency mapped from p. Each composite knocked out plays a short
    percussive click. The sieve sweeps left to right in time — early primes
    (2,3,5,7) create deep rhythmic pulses as their multiples are eliminated,
    while larger primes add bright sustained tones to an accumulating chord.
    """
    print("Sieve of Eratosthenes...")
    N = 200
    duration = 55.0
    total = int(SR * duration)
    buf = np.zeros(total)
    
    # Run sieve, record events
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    
    events = []  # (time_frac, number, event_type)
    # time: number n maps to position (n-2)/(N-2) in the piece
    
    primes_found = []
    
    for i in range(2, N + 1):
        t_frac = (i - 2) / (N - 2)
        if is_prime[i]:
            events.append((t_frac, i, 'prime'))
            primes_found.append(i)
            # Eliminate multiples
            for j in range(i*i, N + 1, i):
                if is_prime[j]:
                    is_prime[j] = False
                    # Composite elimination event — timed at the composite's position
                    elim_frac = (j - 2) / (N - 2)
                    events.append((elim_frac, j, 'elim'))
    
    # Map primes to frequencies: log scale from 55 Hz (prime 2) to ~2000 Hz
    def prime_freq(p):
        return 55.0 * (2 ** (np.log2(p / 2) * 2.5))
    
    # Prime discovery: sustained tone that rings for the rest of the piece (fading slowly)
    for t_frac, p, etype in events:
        if etype != 'prime':
            continue
        start = int(t_frac * 0.85 * total)  # leave room for final chord
        remain = total - start
        if remain < SR * 0.1:
            continue
        freq = prime_freq(p)
        if freq > 8000:
            continue
        t = np.arange(remain) / SR
        # Sustained tone with slow decay
        amp = 0.04 * np.exp(-t / (duration * 0.6)) / (1 + np.log2(p))
        tone = amp * np.sin(2 * np.pi * freq * t)
        # Add slight vibrato for warmth
        vib = 1.0 + 0.003 * np.sin(2 * np.pi * 4.5 * t)
        tone = amp * np.sin(2 * np.pi * freq * np.cumsum(vib) / SR)
        tone = fade(tone, 0.005, 0.05)
        mix_at(buf, tone, start)
    
    # Composite elimination: short percussive click
    for t_frac, n, etype in events:
        if etype != 'elim':
            continue
        start = int(t_frac * 0.85 * total)
        freq = 55.0 * (2 ** (np.log2(n / 2) * 2.5))
        if freq > 10000:
            freq = 10000
        click_dur = 0.015
        t = np.arange(int(SR * click_dur)) / SR
        env = np.exp(-t / 0.004)
        click = 0.03 * env * np.sin(2 * np.pi * freq * t)
        mix_at(buf, click, start)
    
    # Ending: final 8 seconds — all primes < 50 chord together, crescendo then fade
    chord_start = int(0.86 * total)
    chord_dur = total - chord_start
    t = np.arange(chord_dur) / SR
    chord_env = np.sin(np.pi * t / (chord_dur / SR)) ** 0.5 * 0.3  # arch envelope
    chord = np.zeros(chord_dur)
    small_primes = [p for p in primes_found if p < 50]
    for p in small_primes:
        freq = prime_freq(p)
        chord += chord_env / len(small_primes) * np.sin(2 * np.pi * freq * t)
    mix_at(buf, fade(chord, 0.1, 0.5), chord_start)
    
    # Gentle low drone throughout: fundamental on prime 2 = 55 Hz
    t_full = np.arange(total) / SR
    drone = 0.06 * np.sin(2 * np.pi * 55 * t_full) * np.exp(-t_full / (duration * 0.8))
    drone = fade(drone, 0.5, 2.0)
    buf += drone
    
    buf = fade(buf, 0.1, 1.0)
    make_wav("nt_1_sieve.wav", buf)

# ── Piece 2: Prime Gaps (50s, stereo) ──

def prime_gaps():
    """The irregular rhythm between consecutive primes.
    
    First 300 primes. Each prime plays a note; the GAP to the next prime
    determines the silence duration and the pitch interval. Small gaps (twins!)
    create rapid-fire pairs. Large gaps create dramatic pauses.
    
    Left channel: pure prime tone (frequency from prime value).
    Right channel: gap resonance (frequency from gap size, sustained).
    """
    print("Prime Gaps...")
    # Generate primes
    def sieve(limit):
        s = [True] * (limit + 1)
        s[0] = s[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if s[i]:
                for j in range(i*i, limit + 1, i):
                    s[j] = False
        return [i for i in range(2, limit + 1) if s[i]]
    
    primes = sieve(2500)[:300]
    gaps = [primes[i+1] - primes[i] for i in range(len(primes) - 1)]
    
    duration = 50.0
    total = int(SR * duration)
    L = np.zeros(total)
    R = np.zeros(total)
    
    # Time allocation: each prime gets time proportional to the gap after it
    # (normalized so total = 48s, leaving 2s for fade)
    total_gap = sum(gaps)
    time_per_gap_unit = 48.0 / total_gap
    
    cursor = 0.0  # seconds
    for i, (p, g) in enumerate(zip(primes[:-1], gaps)):
        sample_pos = int(cursor * SR)
        if sample_pos >= total:
            break
        
        # Left: prime note — pentatonic mapping, duration proportional to gap
        note_dur = max(0.03, g * time_per_gap_unit * 0.6)
        note_dur = min(note_dur, 1.5)
        freq = pentatonic_freq(i % 25, base=130.0)
        n_samp = int(SR * note_dur)
        if n_samp > 0 and sample_pos + n_samp < total:
            t = np.arange(n_samp) / SR
            # Brighter for twin primes (gap=2), darker for large gaps
            n_harmonics = max(1, 8 - g // 2)
            tone = np.zeros(n_samp)
            for h in range(1, n_harmonics + 1):
                tone += (1.0 / h) * np.sin(2 * np.pi * freq * h * t)
            env = np.exp(-t / (note_dur * 0.7))
            tone *= env * 0.12
            tone = fade(tone, 0.003, 0.01)
            mix_at(L, tone, sample_pos)
        
        # Right: gap resonance — sustained FM tone whose frequency = gap mapped
        gap_freq = 110 * (2 ** (g / 12.0))  # gap as semitone interval
        gap_dur = g * time_per_gap_unit
        n_gap = int(SR * gap_dur)
        if n_gap > 0 and sample_pos + n_gap < total:
            t = np.arange(n_gap) / SR
            mod = 1.0 + (g / 30.0) * np.sin(2 * np.pi * (g * 0.5) * t)
            env = 0.08 * np.exp(-t / (gap_dur * 0.8))
            tone = env * np.sin(2 * np.pi * gap_freq * np.cumsum(mod) / SR)
            tone = fade(tone, 0.005, 0.02)
            mix_at(R, tone, sample_pos)
        
        cursor += g * time_per_gap_unit
    
    # Low drone: 55 Hz throughout
    t_full = np.arange(total) / SR
    drone = 0.05 * np.sin(2 * np.pi * 55 * t_full)
    drone *= np.concatenate([np.linspace(0, 1, SR), np.ones(total - 2*SR), np.linspace(1, 0, SR)])
    L += drone * 0.7
    R += drone * 0.7
    
    stereo = np.array([fade(L, 0.1, 1.0), fade(R, 0.1, 1.0)])
    make_wav("nt_2_prime_gaps.wav", stereo)

# ── Piece 3: Modular Worlds (55s, stereo) ──

def modular_worlds():
    """Residue classes mod small primes as interlocking cycles.
    
    Each prime p ∈ {2,3,5,7,11,13} creates a repeating cycle of length p.
    Numbers 1..500 are played in sequence; at each n, the residue n mod p
    determines pitch within that prime's voice. The voices layer to create
    polyrhythmic interference patterns — the Chinese Remainder Theorem
    made audible.
    
    Stereo: smaller primes left, larger primes right.
    """
    print("Modular Worlds...")
    duration = 55.0
    total = int(SR * duration)
    L = np.zeros(total)
    R = np.zeros(total)
    
    mod_primes = [2, 3, 5, 7, 11, 13]
    N = 500
    time_per_n = (duration - 3.0) / N  # leave 3s for ending
    
    # Each prime gets a base frequency and a scale
    base_freqs = {
        2: 110,    # low, left
        3: 165,    # low-mid, left-center
        5: 220,    # mid, center-left
        7: 330,    # mid, center-right
        11: 440,   # high, right-center
        13: 550,   # high, right
    }
    
    # Stereo position: 0=left, 1=right
    pan = {2: 0.1, 3: 0.25, 5: 0.4, 7: 0.6, 11: 0.75, 13: 0.9}
    
    for n in range(1, N + 1):
        t_start = int((n - 1) * time_per_n * SR)
        if t_start >= total:
            break
        
        for p in mod_primes:
            r = n % p  # residue
            base = base_freqs[p]
            # Residue r maps to r-th note of harmonic series above base
            freq = base * (1 + r * 0.5)
            
            note_dur = time_per_n * 0.8
            n_samp = int(SR * note_dur)
            if n_samp < 10:
                continue
            
            t = np.arange(n_samp) / SR
            # When residue = 0, special: brighter attack (cycle complete)
            if r == 0:
                # FM burst for cycle completion
                mod_depth = 3.0
                mod_freq_val = freq * 1.5
                sig = np.sin(2 * np.pi * freq * t + mod_depth * np.sin(2 * np.pi * mod_freq_val * t))
                env = 0.06 * np.exp(-t / (note_dur * 0.5))
            else:
                sig = np.sin(2 * np.pi * freq * t)
                env = 0.03 * np.exp(-t / (note_dur * 0.6))
            
            sig *= env
            sig = fade(sig, 0.002, 0.005)
            
            # Pan
            p_val = pan[p]
            l_gain = np.cos(p_val * np.pi / 2)
            r_gain = np.sin(p_val * np.pi / 2)
            
            if t_start + n_samp < total:
                mix_at(L, sig * l_gain, t_start)
                mix_at(R, sig * r_gain, t_start)
    
    # Ending: all primes play their "zero residue" simultaneously — a CRT chord
    chord_start = int((duration - 3.0) * SR)
    chord_dur = int(3.0 * SR)
    t = np.arange(chord_dur) / SR
    chord_env = 0.15 * np.sin(np.pi * t / 3.0) ** 0.7
    for p in mod_primes:
        base = base_freqs[p]
        # Play base + first few harmonics
        tone = np.zeros(chord_dur)
        for h in range(1, 5):
            tone += (1.0/h) * np.sin(2 * np.pi * base * h * t)
        tone *= chord_env / len(mod_primes)
        p_val = pan[p]
        l_gain = np.cos(p_val * np.pi / 2)
        r_gain = np.sin(p_val * np.pi / 2)
        mix_at(L, tone * l_gain, chord_start)
        mix_at(R, tone * r_gain, chord_start)
    
    stereo = np.array([fade(L, 0.2, 1.5), fade(R, 0.2, 1.5)])
    make_wav("nt_3_modular_worlds.wav", stereo)


if __name__ == "__main__":
    print("=== Phase 10: Number Theory ===")
    sieve_of_eratosthenes()
    prime_gaps()
    modular_worlds()
    print("Done.")
