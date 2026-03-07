"""
Phase 9: Information Theory — 信息的听觉维度
==============================================

Three pieces exploring compression, entropy, and coding:

1. Entropy Gradient (55s) — Shannon entropy as harmonic complexity.
   A byte stream transitions from perfect order (00000...) through
   increasing randomness to maximum entropy. Low entropy = pure sine,
   high entropy = dense harmonic cluster.

2. Huffman Tree (50s) — Frequency-weighted coding as melody.
   English letter frequencies become note durations and pitches.
   Common letters (e,t,a) = short bright motifs. Rare letters (z,q,x) =
   long deep tones. The tree structure itself becomes a descending melody.

3. LZ Window (55s, stereo) — Pattern matching as musical memory.
   A text is scanned with a sliding window. New symbols = new notes.
   Back-references = the referenced phrase replays (echo/canon).
   Compression ratio → stereo width: compressible = narrow, random = wide.
"""

import numpy as np
import struct
import os

SR = 44100

def write_wav(path, data, sr=SR):
    """Write mono or stereo float array as 16-bit WAV."""
    if data.ndim == 1:
        channels = 1
    else:
        channels = data.shape[0]
        data = data.T.flatten()
    data = np.clip(data, -1, 1)
    pcm = (data * 32767).astype(np.int16)
    with open(path, 'wb') as f:
        nbytes = pcm.nbytes
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + nbytes))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, channels, sr, sr * channels * 2, channels * 2, 16))
        f.write(b'data')
        f.write(struct.pack('<I', nbytes))
        f.write(pcm.tobytes())

def fade(n, fade_in=0.01, fade_out=0.01, sr=SR):
    """Generate fade envelope."""
    env = np.ones(n)
    fi = int(fade_in * sr)
    fo = int(fade_out * sr)
    if fi > 0:
        env[:fi] = np.linspace(0, 1, fi)
    if fo > 0:
        env[-fo:] = np.linspace(1, 0, fo)
    return env

def mix_to_length(tracks, length):
    """Mix list of (offset_samples, audio_array) into a buffer of given length."""
    buf = np.zeros(length)
    for offset, audio in tracks:
        end = min(offset + len(audio), length)
        buf[offset:end] += audio[:end - offset]
    return buf

# ─────────────────────────────────────────────
# 1. Entropy Gradient
# ─────────────────────────────────────────────
def entropy_gradient():
    """Shannon entropy as harmonic complexity.
    
    Sweep from order to chaos: generate byte sequences with controlled entropy,
    map entropy level to number of harmonics and their distribution.
    """
    duration = 55.0
    n_samples = int(duration * SR)
    n_steps = 200  # number of entropy levels
    step_dur = duration / n_steps
    step_samples = int(step_dur * SR)
    
    output = np.zeros(n_samples)
    
    base_freq = 110.0  # A2
    
    for i in range(n_steps):
        # Entropy goes from ~0 to ~8 bits (max for bytes)
        entropy_level = (i / (n_steps - 1)) ** 1.5  # nonlinear sweep, linger in low entropy
        
        t = np.arange(step_samples) / SR
        offset = i * step_samples
        
        # Number of harmonics: 1 (pure) to 24 (dense)
        n_harmonics = max(1, int(1 + 23 * entropy_level))
        
        signal = np.zeros(step_samples)
        
        for h in range(1, n_harmonics + 1):
            # At low entropy: harmonics are integer multiples (consonant)
            # At high entropy: harmonics drift to inharmonic (dissonant)
            detune = 1.0 + (np.random.random() - 0.5) * 0.03 * entropy_level
            freq = base_freq * h * detune
            
            # Amplitude: 1/h for harmonic series, flattened by entropy
            amp = (1.0 / h) ** (1.0 - 0.6 * entropy_level)
            amp *= 0.15 / max(1, n_harmonics ** 0.3)
            
            signal += amp * np.sin(2 * np.pi * freq * t + np.random.random() * 2 * np.pi * entropy_level)
        
        # Add noise proportional to entropy
        noise = np.random.randn(step_samples) * 0.04 * entropy_level ** 2
        signal += noise
        
        # Apply envelope
        env = fade(step_samples, 0.005, 0.005)
        signal *= env
        
        end = min(offset + step_samples, n_samples)
        output[offset:end] += signal[:end - offset]
    
    # Master envelope
    output *= fade(n_samples, 0.5, 1.0)
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output * 0.85 / peak
    
    return output

# ─────────────────────────────────────────────
# 2. Huffman Tree
# ─────────────────────────────────────────────
def huffman_tree():
    """English letter frequencies as pitch and duration.
    
    Each letter gets a note: frequency from Huffman code length (short code = high pitch),
    duration proportional to code length. Letters arrive in frequency order (most common first),
    then the tree is traversed depth-first as a descending melody.
    """
    duration = 50.0
    
    # English letter frequencies (approximate)
    letter_freq = {
        'e': 12.7, 't': 9.1, 'a': 8.2, 'o': 7.5, 'i': 7.0,
        'n': 6.7, 's': 6.3, 'h': 6.1, 'r': 6.0, 'd': 4.3,
        'l': 4.0, 'c': 2.8, 'u': 2.8, 'm': 2.4, 'w': 2.4,
        'f': 2.2, 'g': 2.0, 'y': 2.0, 'p': 1.9, 'b': 1.5,
        'v': 1.0, 'k': 0.8, 'j': 0.15, 'x': 0.15, 'q': 0.10,
        'z': 0.07,
    }
    
    # Simple Huffman-like code length estimation: -log2(freq)
    total = sum(letter_freq.values())
    code_lengths = {}
    for letter, freq in letter_freq.items():
        p = freq / total
        code_lengths[letter] = max(2, int(-np.log2(p) + 0.5))
    
    # Sort by frequency (most common first)
    sorted_letters = sorted(letter_freq.keys(), key=lambda x: -letter_freq[x])
    
    # Part 1: Letters arrive in frequency order (30s)
    part1_dur = 30.0
    n_letters = len(sorted_letters)
    note_gap = part1_dur / n_letters
    
    tracks = []
    
    # Pentatonic scale for pleasant mapping
    scale = [0, 2, 4, 7, 9]  # pentatonic intervals
    base_midi = 72  # C5
    
    for idx, letter in enumerate(sorted_letters):
        cl = code_lengths[letter]
        freq_ratio = letter_freq[letter] / total
        
        # Pitch: short code (common) = high, long code (rare) = low
        # Map code length 2-9 to MIDI range
        midi = base_midi + 24 - cl * 3
        freq = 440 * 2 ** ((midi - 69) / 12)
        
        # Duration: proportional to code length (rare letters ring longer)
        note_dur = 0.15 + cl * 0.08
        n_samp = int(note_dur * SR)
        t = np.arange(n_samp) / SR
        
        # Timbre: common letters = pure, rare = rich harmonics
        n_harm = min(cl, 8)
        signal = np.zeros(n_samp)
        for h in range(1, n_harm + 1):
            amp = 0.3 / h
            signal += amp * np.sin(2 * np.pi * freq * h * t)
        
        signal *= fade(n_samp, 0.005, note_dur * 0.4)
        
        offset = int(idx * note_gap * SR)
        tracks.append((offset, signal))
    
    # Part 2: Tree traversal — descending arpeggios (20s)
    part2_start = int(part1_dur * SR)
    part2_dur = 20.0
    
    # Simulate tree traversal: go deep, come back
    # Depth 1→max_depth→1, cycling
    max_depth = max(code_lengths.values())
    traversal = []
    for d in range(1, max_depth + 1):
        traversal.append(d)
    for d in range(max_depth - 1, 0, -1):
        traversal.append(d)
    # Repeat to fill time
    traversal = traversal * 4
    
    trav_gap = part2_dur / len(traversal)
    
    for idx, depth in enumerate(traversal):
        midi = base_midi + 24 - depth * 3
        freq = 440 * 2 ** ((midi - 69) / 12)
        
        note_dur = 0.1 + depth * 0.04
        n_samp = int(note_dur * SR)
        t = np.arange(n_samp) / SR
        
        # FM synthesis: depth modulates FM index
        mod_freq = freq * 2.0
        fm_index = depth * 0.5
        signal = 0.25 * np.sin(2 * np.pi * freq * t + fm_index * np.sin(2 * np.pi * mod_freq * t))
        signal *= fade(n_samp, 0.003, note_dur * 0.5)
        
        offset = part2_start + int(idx * trav_gap * SR)
        tracks.append((offset, signal))
    
    total_samples = int(duration * SR)
    output = mix_to_length(tracks, total_samples)
    
    # Add a low drone on the root frequency
    t_all = np.arange(total_samples) / SR
    root_freq = 440 * 2 ** ((base_midi - 12 - 69) / 12)
    drone = 0.08 * np.sin(2 * np.pi * root_freq * t_all)
    drone *= fade(total_samples, 2.0, 3.0)
    output += drone
    
    output *= fade(total_samples, 0.3, 1.5)
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output * 0.85 / peak
    
    return output

# ─────────────────────────────────────────────
# 3. LZ Window — Pattern as Echo
# ─────────────────────────────────────────────
def lz_window():
    """LZ-style sliding window compression as stereo musical memory.
    
    Process a text character by character. New (literal) characters get fresh notes.
    Back-references replay the referenced substring as an echo in the other channel.
    Compression ratio modulates stereo width.
    """
    duration = 55.0
    n_samples = int(duration * SR)
    
    # Use a text with repeating patterns
    text = (
        "to be or not to be that is the question "
        "whether tis nobler in the mind to suffer "
        "the slings and arrows of outrageous fortune "
        "or to take arms against a sea of troubles "
        "and by opposing end them to die to sleep "
        "no more and by a sleep to say we end "
        "the heartache and the thousand natural shocks "
    )
    
    # Assign each unique character a pitch (pentatonic)
    unique_chars = sorted(set(text))
    char_to_midi = {}
    base = 60  # C4
    penta = [0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24, 26, 28, 31, 33]
    for i, ch in enumerate(unique_chars):
        char_to_midi[ch] = base + penta[i % len(penta)]
    
    # Simple LZ77-like parsing
    window_size = 20
    tokens = []  # (is_ref, data) where data is char or (offset, length)
    pos = 0
    while pos < len(text):
        best_offset = 0
        best_length = 0
        search_start = max(0, pos - window_size)
        for offset in range(1, pos - search_start + 1):
            length = 0
            while (pos + length < len(text) and 
                   text[pos - offset + length] == text[pos + length] and
                   length < 15):
                length += 1
            if length > best_length and length >= 3:
                best_length = length
                best_offset = offset
        
        if best_length >= 3:
            tokens.append(('ref', best_offset, best_length, text[pos:pos + best_length]))
            pos += best_length
        else:
            tokens.append(('lit', text[pos]))
            pos += 1
    
    # Generate audio
    left_tracks = []
    right_tracks = []
    
    time_per_token = duration / len(tokens)
    
    for idx, token in enumerate(tokens):
        t_offset = int(idx * time_per_token * SR)
        
        if token[0] == 'lit':
            ch = token[1]
            midi = char_to_midi[ch]
            freq = 440 * 2 ** ((midi - 69) / 12)
            
            note_dur = min(time_per_token * 0.9, 0.3)
            n_samp = int(note_dur * SR)
            t = np.arange(n_samp) / SR
            
            # Fresh note: bright FM synthesis (new information)
            mod = freq * 3
            signal = 0.3 * np.sin(2 * np.pi * freq * t + 1.5 * np.sin(2 * np.pi * mod * t))
            signal *= fade(n_samp, 0.003, note_dur * 0.6)
            
            # Literal = center-left
            left_tracks.append((t_offset, signal * 0.7))
            right_tracks.append((t_offset, signal * 0.3))
        
        else:  # reference
            _, offset, length, substr = token
            
            # Reference: replay the pattern as echo (softer, in right channel)
            # Original in left, echo in right
            for j, ch in enumerate(substr):
                midi = char_to_midi[ch]
                freq = 440 * 2 ** ((midi - 69) / 12)
                
                sub_dur = min(time_per_token * 0.7 / max(length, 1), 0.2)
                n_samp = int(sub_dur * SR)
                t = np.arange(n_samp) / SR
                
                # Echo: pure sine (remembered, simplified)
                signal = 0.2 * np.sin(2 * np.pi * freq * t)
                # Add slight detuned copy for warmth
                signal += 0.1 * np.sin(2 * np.pi * freq * 1.003 * t)
                signal *= fade(n_samp, 0.003, sub_dur * 0.5)
                
                sub_offset = t_offset + int(j * sub_dur * SR)
                
                # Echo in right channel, ghost in left
                left_tracks.append((sub_offset, signal * 0.2))
                right_tracks.append((sub_offset, signal * 0.8))
    
    left = mix_to_length(left_tracks, n_samples)
    right = mix_to_length(right_tracks, n_samples)
    
    # Add subtle drone
    t_all = np.arange(n_samples) / SR
    drone_freq = 440 * 2 ** ((base - 24 - 69) / 12)
    drone = 0.06 * np.sin(2 * np.pi * drone_freq * t_all)
    drone *= fade(n_samples, 1.0, 2.0)
    left += drone
    right += drone
    
    # Master envelope
    env = fade(n_samples, 0.5, 2.0)
    left *= env
    right *= env
    
    stereo = np.stack([left, right])
    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo = stereo * 0.85 / peak
    
    return stereo

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    outdir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(outdir, exist_ok=True)
    
    print("Generating Entropy Gradient...")
    write_wav(os.path.join(outdir, 'info_1_entropy_gradient.wav'), entropy_gradient())
    
    print("Generating Huffman Tree...")
    write_wav(os.path.join(outdir, 'info_2_huffman_tree.wav'), huffman_tree())
    
    print("Generating LZ Window...")
    write_wav(os.path.join(outdir, 'info_3_lz_window.wav'), lz_window(), SR)
    
    print("Done — 3 tracks in output/")
