#!/usr/bin/env python3
"""Phase 13: Cryptography — the sound of secrets.

Three pieces exploring cryptographic primitives as audible structures:
1. One-Time Pad: plaintext melody destroyed by XOR with random key
2. Hash Avalanche: SHA-256 avalanche effect — one bit flip, total transformation
3. Diffie-Hellman: two private melodies converging to a shared secret harmony
"""

import numpy as np
import hashlib
import struct
import os

SR = 44100

def fade(n):
    """Cosine fade envelope."""
    t = np.linspace(0, np.pi / 2, n)
    return np.sin(t)

def sine(freq, dur, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t)

def fm_tone(freq, dur, mod_ratio=2.0, mod_depth=1.0, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    mod = mod_depth * freq * np.sin(2 * np.pi * mod_ratio * freq * t)
    return np.sin(2 * np.pi * freq * t + mod)

def env_adsr(n, a=0.05, d=0.1, s=0.6, r=0.2):
    """Simple ADSR envelope."""
    samples = np.zeros(n)
    a_n = int(a * n)
    d_n = int(d * n)
    r_n = int(r * n)
    s_n = n - a_n - d_n - r_n
    if s_n < 0:
        s_n = 0
    idx = 0
    if a_n > 0:
        samples[idx:idx+a_n] = np.linspace(0, 1, a_n)
        idx += a_n
    if d_n > 0:
        samples[idx:idx+d_n] = np.linspace(1, s, d_n)
        idx += d_n
    if s_n > 0:
        samples[idx:idx+s_n] = s
        idx += s_n
    if r_n > 0:
        samples[idx:idx+r_n] = np.linspace(s, 0, r_n)
    return samples

def normalize(sig, peak=0.85):
    mx = np.max(np.abs(sig))
    if mx > 0:
        sig = sig * peak / mx
    return sig

def write_wav(path, data, sr=SR, stereo=False):
    if stereo:
        data = np.clip(data, -1, 1)
        left = data[0]
        right = data[1]
        interleaved = np.empty(len(left) + len(right), dtype=np.float64)
        interleaved[0::2] = left
        interleaved[1::2] = right
        raw = (interleaved * 32767).astype(np.int16).tobytes()
        nc = 2
        n = len(left)
    else:
        data = np.clip(data, -1, 1)
        raw = (data * 32767).astype(np.int16).tobytes()
        nc = 1
        n = len(data)
    with open(path, 'wb') as f:
        f.write(b'RIFF')
        dsize = len(raw)
        f.write(struct.pack('<I', 36 + dsize))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, nc, sr, sr * nc * 2, nc * 2, 16))
        f.write(b'data')
        f.write(struct.pack('<I', dsize))
        f.write(raw)


# ─── Piece 1: One-Time Pad ─────────────────────────────────────────────
def one_time_pad(dur=55):
    """Plaintext melody XOR'd with random key → ciphertext noise.
    
    Structure:
    - Section A (0-18s): Plaintext melody — clear, musical, pentatonic
    - Section B (18-36s): XOR encryption happening in real-time — melody dissolving
    - Section C (36-50s): Ciphertext — the same data, now unintelligible noise
    - Coda (50-55s): A ghost of the plaintext leaks through — decryption hint
    """
    n = int(SR * dur)
    out = np.zeros(n)
    
    # Pentatonic scale for plaintext melody
    scale = [261.63, 293.66, 329.63, 392.00, 440.00,  # C4 D4 E4 G4 A4
             523.25, 587.33, 659.25, 783.99, 880.00]   # C5 D5 E5 G5 A5
    
    # Generate a repeating plaintext melody (16 notes, looped)
    np.random.seed(42)  # Deterministic "message"
    melody_notes = [scale[i] for i in np.random.choice(len(scale), 16)]
    melody_durs = [0.3 + 0.2 * np.random.random() for _ in range(16)]
    
    # Build plaintext signal: clear musical notes
    def build_melody(start_t, end_t, gain=1.0):
        sig = np.zeros(n)
        t_cursor = start_t
        note_idx = 0
        while t_cursor < end_t:
            freq = melody_notes[note_idx % 16]
            nd = melody_durs[note_idx % 16]
            if t_cursor + nd > end_t:
                nd = end_t - t_cursor
            s = int(t_cursor * SR)
            e = int((t_cursor + nd) * SR)
            if e > n:
                e = n
            seg_len = e - s
            if seg_len > 0:
                env = env_adsr(seg_len)
                t_n = np.arange(seg_len) / SR
                tone = np.sin(2 * np.pi * freq * t_n) * env
                tone += 0.3 * np.sin(2 * np.pi * freq * 2 * t_n) * env
                tone += 0.15 * np.sin(2 * np.pi * freq * 3 * t_n) * env
                sig[s:e] += tone * gain
            t_cursor += nd + 0.05
            note_idx += 1
        return sig
    
    # Generate "key" — random noise, but structured in same note slots
    def build_key_noise(start_t, end_t, gain=1.0):
        sig = np.zeros(n)
        rng = np.random.RandomState(99)  # Different seed = random key
        t_cursor = start_t
        note_idx = 0
        while t_cursor < end_t:
            nd = melody_durs[note_idx % 16]
            if t_cursor + nd > end_t:
                nd = end_t - t_cursor
            s = int(t_cursor * SR)
            e = int((t_cursor + nd) * SR)
            seg_len = e - s
            if seg_len > 0:
                rfreq = 100 + rng.random() * 800
                t_n = np.arange(seg_len) / SR
                mr = 1.0 + rng.random() * 5
                md = rng.random() * 3
                mod = md * rfreq * np.sin(2 * np.pi * mr * rfreq * t_n)
                tone = np.sin(2 * np.pi * rfreq * t_n + mod)
                tone *= env_adsr(seg_len)
                sig[s:e] += tone * gain
            t_cursor += nd + 0.05
            note_idx += 1
        return sig
    
    # Section A: Pure plaintext
    out += build_melody(0, 18, gain=0.7)
    
    # Section B: XOR dissolving — crossfade from plaintext to noise
    plain_b = build_melody(18, 36, gain=0.7)
    noise_b = build_key_noise(18, 36, gain=0.7)
    s_b = int(18 * SR)
    e_b = int(36 * SR)
    xfade = np.linspace(0, 1, e_b - s_b)
    out[s_b:e_b] += plain_b[s_b:e_b] * (1 - xfade) + noise_b[s_b:e_b] * xfade
    
    # Section C: Pure ciphertext — unintelligible
    out += build_key_noise(36, 50, gain=0.6)
    
    # Coda: Ghost of plaintext — very quiet, filtered
    ghost = build_melody(50, 55, gain=0.15)
    # Low-pass by averaging (crude but effective)
    kernel = np.ones(200) / 200
    s_c = int(50 * SR)
    ghost_seg = np.convolve(ghost[s_c:], kernel, mode='same')
    out[s_c:s_c+len(ghost_seg)] += ghost_seg[:n-s_c] if len(ghost_seg) > n-s_c else ghost_seg
    
    # Soft low drone throughout
    t = np.arange(n) / SR
    drone = 0.08 * sine(55, dur) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.03 * t))
    out += drone[:n]
    
    # Fade in/out
    fi = int(0.5 * SR)
    fo = int(1.5 * SR)
    out[:fi] *= np.linspace(0, 1, fi)
    out[-fo:] *= np.linspace(1, 0, fo)
    
    return normalize(out)


# ─── Piece 2: Hash Avalanche ───────────────────────────────────────────
def hash_avalanche(dur=50):
    """SHA-256 avalanche effect: flip one bit, hear total transformation.
    
    Structure:
    - 25 pairs of hashes, each pair differs by exactly 1 bit in input
    - Each pair plays sequentially: original hash sound → bit-flipped hash sound
    - Mapping: each byte of hash → frequency + amplitude of one partial
    - The contrast between near-identical inputs producing alien outputs = the drama
    
    Stereo: left = original, right = flipped version
    """
    n = int(SR * dur)
    left = np.zeros(n)
    right = np.zeros(n)
    
    pair_dur = dur / 25  # ~2s per pair
    
    for i in range(25):
        # Create input: just the number i as bytes
        input_bytes = i.to_bytes(4, 'big')
        # Flip one bit in the input
        flipped = bytearray(input_bytes)
        flipped[3] ^= 0x01  # Flip LSB
        
        hash_orig = hashlib.sha256(input_bytes).digest()
        hash_flip = hashlib.sha256(bytes(flipped)).digest()
        
        t_start = i * pair_dur
        s = int(t_start * SR)
        seg_n = int(pair_dur * SR)
        if s + seg_n > n:
            seg_n = n - s
        
        # Convert hash to sound: 32 bytes → 16 partials (pairs of bytes)
        def hash_to_sound(h, length):
            sig = np.zeros(length)
            t = np.arange(length) / SR
            base_freq = 110  # A2
            for j in range(16):
                # Two bytes → frequency offset and amplitude
                freq_byte = h[j * 2]
                amp_byte = h[j * 2 + 1]
                freq = base_freq * (1 + j * 0.5) + (freq_byte / 255.0) * 50
                amp = (amp_byte / 255.0) * 0.15
                sig += amp * np.sin(2 * np.pi * freq * t)
            sig *= env_adsr(length, a=0.08, d=0.15, s=0.5, r=0.3)
            return sig
        
        if seg_n > 0:
            # First half: original, second half: flipped
            half = seg_n // 2
            orig_sound = hash_to_sound(hash_orig, half)
            flip_sound = hash_to_sound(hash_flip, half)
            
            # Original on both channels first
            left[s:s+half] += orig_sound * 0.8
            right[s:s+half] += orig_sound * 0.3  # Quieter echo on right
            
            # Flipped on both channels second (emphasis on right)
            left[s+half:s+seg_n] += flip_sound * 0.3
            right[s+half:s+seg_n] += flip_sound * 0.8
            
            # Bit-flip click at the transition
            click_pos = s + half
            click_len = min(int(0.01 * SR), n - click_pos)
            if click_len > 0:
                click_t = np.arange(click_len) / SR
                click = 0.5 * np.sin(2 * np.pi * 2000 * click_t) * np.exp(-click_t * 500)
                left[click_pos:click_pos+click_len] += click
                right[click_pos:click_pos+click_len] += click
    
    # Hamming distance visualization as low drone
    # Show how different the hashes are despite similar inputs
    t = np.arange(n) / SR
    drone = 0.06 * sine(73.42, dur)  # D2
    left += drone[:n]
    right += drone[:n]
    
    # Fade
    fi, fo = int(0.3 * SR), int(1.0 * SR)
    left[:fi] *= np.linspace(0, 1, fi)
    right[:fi] *= np.linspace(0, 1, fi)
    left[-fo:] *= np.linspace(1, 0, fo)
    right[-fo:] *= np.linspace(1, 0, fo)
    
    return normalize(np.array([left, right]), peak=0.8)


# ─── Piece 3: Diffie-Hellman ──────────────────────────────────────────
def diffie_hellman(dur=55):
    """Two parties derive a shared secret over a public channel.
    
    Structure:
    - Alice (left) has private melody A, Bob (right) has private melody B
    - Phase 1 (0-15s): Private melodies play independently — no relation
    - Phase 2 (15-35s): Public exchange — each sends g^private mod p
      Sonified as: original melodies modulated by a shared "public" carrier
    - Phase 3 (35-50s): Shared secret emerges — both arrive at same harmony
      (g^ab mod p) despite never hearing each other's private melody
    - Coda (50-55s): Shared secret chord sustained, privates fade to silence
    
    The magic: convergence without communication of secrets.
    """
    n = int(SR * dur)
    left = np.zeros(n)
    right = np.zeros(n)
    
    # Alice's private melody: bright, ascending, major feel
    alice_freqs = [329.63, 392.00, 440.00, 493.88, 523.25, 587.33, 659.25, 783.99]  # E4→G5
    # Bob's private melody: warm, descending, minor feel  
    bob_freqs = [493.88, 440.00, 392.00, 349.23, 329.63, 293.66, 261.63, 220.00]  # B4→A3
    
    # Shared secret: a chord both converge to
    secret_freqs = [220.00, 329.63, 440.00, 554.37]  # A3-E4-A4-C#5 (A major)
    
    note_dur = 0.8
    gap = 0.1
    
    # Phase 1: Private melodies, completely independent
    for i in range(8):
        t_start = i * (note_dur + gap) + 0.5
        s = int(t_start * SR)
        seg = int(note_dur * SR)
        if s + seg > n:
            break
        
        # Alice: bright FM tone, left channel
        a_tone = fm_tone(alice_freqs[i], note_dur, mod_ratio=2, mod_depth=0.5)[:seg]
        a_tone += 0.3 * sine(alice_freqs[i] * 2, note_dur)[:seg]
        a_tone *= env_adsr(seg, a=0.05, d=0.1, s=0.7, r=0.15)
        left[s:s+seg] += a_tone * 0.6
        
        # Bob: warm sine tone, right channel
        b_tone = sine(bob_freqs[i], note_dur)[:seg]
        b_tone += 0.4 * sine(bob_freqs[i] * 1.5, note_dur)[:seg]  # Fifth harmonic
        b_tone += 0.2 * sine(bob_freqs[i] * 0.5, note_dur)[:seg]  # Sub
        b_tone *= env_adsr(seg, a=0.08, d=0.15, s=0.6, r=0.2)
        right[s:s+seg] += b_tone * 0.6
    
    # Phase 2: Public exchange — melodies modulated by shared carrier
    pub_carrier_freq = 164.81  # E3 — the "public channel"
    for i in range(12):
        t_start = 15.5 + i * (note_dur + gap) * 0.85
        s = int(t_start * SR)
        seg = int(note_dur * 0.85 * SR)
        if s + seg > n:
            break
        
        af = alice_freqs[i % 8]
        bf = bob_freqs[i % 8]
        t_local = np.arange(seg) / SR
        
        # Alice's public value: her melody * carrier (ring modulation)
        carrier = np.sin(2 * np.pi * pub_carrier_freq * t_local)
        a_pub = np.sin(2 * np.pi * af * t_local) * (0.5 + 0.5 * carrier)
        a_pub *= env_adsr(seg)
        left[s:s+seg] += a_pub * 0.5
        right[s:s+seg] += a_pub * 0.15  # Leaks to public (Bob can hear)
        
        # Bob's public value: his melody * carrier
        b_pub = np.sin(2 * np.pi * bf * t_local) * (0.5 + 0.5 * carrier)
        b_pub *= env_adsr(seg)
        right[s:s+seg] += b_pub * 0.5
        left[s:s+seg] += b_pub * 0.15  # Leaks to public (Alice can hear)
        
        # Shared carrier drone (public channel is audible)
        pub_drone = 0.08 * carrier * env_adsr(seg, a=0.02, s=0.3)
        left[s:s+seg] += pub_drone
        right[s:s+seg] += pub_drone
    
    # Phase 3: Shared secret emerges — convergence!
    # Both channels gradually align to the same chord
    convergence_start = 35.0
    convergence_end = 50.0
    for i in range(len(secret_freqs)):
        freq = secret_freqs[i]
        t_start = convergence_start + i * 1.5
        s = int(t_start * SR)
        e_time = convergence_end
        e = int(e_time * SR)
        if e > n:
            e = n
        seg = e - s
        if seg <= 0:
            continue
        
        t_local = np.arange(seg) / SR
        total_dur = e_time - t_start
        
        # Both channels play the same frequency — shared secret!
        tone = np.sin(2 * np.pi * freq * t_local)
        tone += 0.3 * np.sin(2 * np.pi * freq * 2 * t_local)  # Warm harmonic
        
        # Envelope: gradual swell
        env = np.minimum(t_local / 2.0, 1.0)  # 2s fade in
        env *= np.where(t_local > total_dur - 2, np.linspace(1, 0.4, seg)[-int(2*SR):].tolist() + [0.4] * (seg - int(2*SR)) if seg > int(2*SR) else np.linspace(1, 0.4, seg), 1.0) if False else 1.0
        env = np.minimum(t_local / 2.0, 1.0)
        
        tone *= env * 0.3
        left[s:s+seg] += tone
        right[s:s+seg] += tone
    
    # During Phase 3, private melodies become quieter (secrets consumed)
    for i in range(6):
        t_start = 36 + i * 2
        s = int(t_start * SR)
        seg = int(1.2 * SR)
        if s + seg > n:
            break
        fade_factor = 0.4 * (1 - i / 6)
        
        af = alice_freqs[i % 8]
        a_ghost = sine(af, 1.2)[:seg] * env_adsr(seg) * fade_factor
        left[s:s+seg] += a_ghost
        
        bf = bob_freqs[i % 8]
        b_ghost = sine(bf, 1.2)[:seg] * env_adsr(seg) * fade_factor
        right[s:s+seg] += b_ghost
    
    # Coda: Pure shared chord, both channels identical
    coda_s = int(50 * SR)
    coda_e = int(55 * SR)
    coda_len = coda_e - coda_s
    if coda_len > 0:
        t_local = np.arange(coda_len) / SR
        chord = np.zeros(coda_len)
        for freq in secret_freqs:
            chord += 0.2 * np.sin(2 * np.pi * freq * t_local)
            chord += 0.08 * np.sin(2 * np.pi * freq * 2 * t_local)
        chord *= np.linspace(0.8, 0, coda_len)  # Fade out
        left[coda_s:coda_e] += chord
        right[coda_s:coda_e] += chord
    
    # Global fade
    fi, fo = int(0.5 * SR), int(1.5 * SR)
    left[:fi] *= np.linspace(0, 1, fi)
    right[:fi] *= np.linspace(0, 1, fi)
    left[-fo:] *= np.linspace(1, 0, fo)
    right[-fo:] *= np.linspace(1, 0, fo)
    
    return normalize(np.array([left, right]), peak=0.8)


# ─── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    
    print("Generating One-Time Pad...")
    write_wav("output/crypto_1_one_time_pad.wav", one_time_pad())
    
    print("Generating Hash Avalanche...")
    write_wav("output/crypto_2_hash_avalanche.wav", hash_avalanche(), stereo=True)
    
    print("Generating Diffie-Hellman...")
    write_wav("output/crypto_3_diffie_hellman.wav", diffie_hellman(), stereo=True)
    
    print("Done! Files in output/")
