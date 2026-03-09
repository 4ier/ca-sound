#!/usr/bin/env python3
"""
Phase 11: Automata & Formal Languages
The Chomsky hierarchy as musical structure — from finite to unbounded.

1. Finite Automaton (55s, stereo)
   A DFA recognizing binary strings divisible by 3. Each input bit triggers a
   state transition; states map to pitches in a pentatonic scale. Accept states
   ring with harmonics, reject states are muted/percussive. Random binary strings
   stream through — the automaton's judgment becomes rhythm.

2. Context-Free Grammar (50s, stereo)
   An L-system grammar (Lindenmayer) generates branching structures. Each
   production rule application = a musical phrase. Terminal symbols are notes,
   non-terminals are rests/silences. The tree grows outward: depth → register,
   branch direction → stereo position. Recursive self-similarity = melodic motifs
   at multiple scales.

3. Pushdown Automaton (55s, stereo)
   A PDA matching nested parentheses. Push = rising pitch + harmonic enrichment,
   pop = falling pitch + simplification. Stack depth → bass drone frequency.
   Well-formed strings resolve to consonance; malformed strings end in dissonance.
   The stack is the memory — and you can hear how deep it goes.
"""

import numpy as np
import os

SR = 44100

def make_dirs():
    os.makedirs("output", exist_ok=True)

def normalize(audio, peak=0.9):
    mx = np.max(np.abs(audio))
    if mx > 0:
        audio = audio * (peak / mx)
    return audio

def write_wav(filename, audio):
    """Write stereo or mono float audio to 16-bit WAV."""
    import struct
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    audio = normalize(audio)
    data = (audio * 32767).astype(np.int16)
    n_frames = len(data)
    n_channels = 2
    with open(filename, 'wb') as f:
        # WAV header
        datasize = n_frames * n_channels * 2
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + datasize))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, n_channels, SR, SR * n_channels * 2, n_channels * 2, 16))
        f.write(b'data')
        f.write(struct.pack('<I', datasize))
        f.write(data.tobytes())

def env_ar(n, attack=0.01, release=0.1):
    """Attack-release envelope."""
    a = int(attack * SR)
    r = int(release * SR)
    e = np.ones(n)
    if a > 0:
        e[:min(a, n)] = np.linspace(0, 1, min(a, n))
    if r > 0 and r < n:
        e[-r:] = np.linspace(1, 0, r)
    return e

def sine(freq, dur, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return np.sin(2 * np.pi * freq * t)

def fm_tone(carrier, mod_ratio, mod_depth, dur, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    mod = mod_depth * np.sin(2 * np.pi * carrier * mod_ratio * t)
    return np.sin(2 * np.pi * carrier * t + mod)

# ─── Track 1: Finite Automaton ───

def finite_automaton():
    """DFA for binary divisibility by 3, sonified."""
    duration = 55.0
    n_samples = int(duration * SR)
    audio_l = np.zeros(n_samples)
    audio_r = np.zeros(n_samples)

    # DFA: states 0,1,2 = remainder mod 3. State 0 = accept.
    # Transitions: state, bit -> new_state
    # state 0: bit 0 -> 0, bit 1 -> 1
    # state 1: bit 0 -> 2, bit 1 -> 0
    # state 2: bit 0 -> 1, bit 1 -> 2
    transitions = {
        (0, 0): 0, (0, 1): 1,
        (1, 0): 2, (1, 1): 0,
        (2, 0): 1, (2, 1): 2,
    }

    # Pentatonic scale for states + accept/reject tones
    state_freqs = {0: 261.63, 1: 329.63, 2: 392.00}  # C4, E4, G4
    accept_chord = [261.63, 329.63, 392.00, 523.25]  # C major
    reject_freq = 185.0  # low F#, dissonant

    # Stereo positions per state
    state_pan = {0: 0.5, 1: 0.25, 2: 0.75}

    # Generate random binary strings of varying length
    rng = np.random.default_rng(42)
    strings = []
    for _ in range(60):
        length = rng.integers(3, 12)
        strings.append(rng.integers(0, 2, size=length).tolist())

    # Time allocation
    time_per_bit = 0.08
    pause_between = 0.15
    accept_dur = 0.4
    t_cursor = 0.5  # start with brief silence

    # Background drone: very soft state-0 hum
    drone_dur = duration
    drone = sine(65.41, drone_dur) * 0.06  # C2 drone
    drone_env = env_ar(len(drone), attack=2.0, release=2.0)
    drone *= drone_env
    audio_l[:len(drone)] += drone * 0.5
    audio_r[:len(drone)] += drone * 0.5

    for bits in strings:
        if t_cursor > duration - 2.0:
            break

        state = 0
        for bit in bits:
            if t_cursor >= duration - 1.0:
                break

            freq = state_freqs[state]
            pan = state_pan[state]
            idx = int(t_cursor * SR)

            # Bit sound: short tick
            bit_dur = time_per_bit
            n = int(bit_dur * SR)
            if idx + n > n_samples:
                break

            if bit == 1:
                # Bit 1: bright FM click
                tone = fm_tone(freq * 2, 3.0, 2.0, bit_dur) * env_ar(n, 0.002, 0.05)
            else:
                # Bit 0: soft sine tap
                tone = sine(freq, bit_dur) * env_ar(n, 0.002, 0.04) * 0.6

            audio_l[idx:idx+n] += tone * (1 - pan) * 0.4
            audio_r[idx:idx+n] += tone * pan * 0.4

            state = transitions[(state, bit)]
            t_cursor += bit_dur

        # String complete — accept or reject?
        idx = int(t_cursor * SR)
        if state == 0:
            # Accept: bright chord
            n = int(accept_dur * SR)
            if idx + n <= n_samples:
                chord = np.zeros(n)
                for f in accept_chord:
                    chord += sine(f, accept_dur) * 0.2
                chord *= env_ar(n, 0.01, 0.2)
                audio_l[idx:idx+n] += chord * 0.5
                audio_r[idx:idx+n] += chord * 0.5
            t_cursor += accept_dur
        else:
            # Reject: short dissonant buzz
            rej_dur = 0.15
            n = int(rej_dur * SR)
            if idx + n <= n_samples:
                buzz = fm_tone(reject_freq, 7.1, 5.0, rej_dur) * env_ar(n, 0.002, 0.08) * 0.3
                audio_l[idx:idx+n] += buzz * 0.5
                audio_r[idx:idx+n] += buzz * 0.5
            t_cursor += rej_dur

        t_cursor += pause_between

    audio = np.column_stack([audio_l, audio_r])
    return audio

# ─── Track 2: Context-Free Grammar (L-System) ───

def context_free_grammar():
    """L-system branching structure as music."""
    duration = 50.0
    n_samples = int(duration * SR)
    audio_l = np.zeros(n_samples)
    audio_r = np.zeros(n_samples)

    # L-system: axiom "F", rules: F -> F[+F]F[-F]F
    # F = draw forward (note), + = turn right, - = turn left
    # [ = push (save), ] = pop (restore)
    axiom = "F"
    rules = {"F": "F[+F]F[-F]F"}

    def expand(s, rules, n):
        for _ in range(n):
            new = []
            for c in s:
                new.append(rules.get(c, c))
            s = "".join(new)
        return s

    # 4 generations, each gets a section of time
    # Gen 1: simple, Gen 4: dense fractal
    pentatonic = [220.0, 246.94, 293.66, 329.63, 392.00,
                  440.0, 493.88, 587.33, 659.26, 784.00]

    t_cursor = 0.3
    section_dur = duration / 4 - 1.0

    for gen in range(1, 5):
        lstring = expand(axiom, rules, gen)

        # Limit to manageable length
        lstring = lstring[:500]

        # Time per symbol scales with generation
        time_per_sym = section_dur / max(len(lstring), 1)
        time_per_sym = max(0.02, min(time_per_sym, 0.3))

        angle = 0.0
        stack = []
        depth = 0
        pitch_idx = 4  # start mid-scale

        for sym in lstring:
            if t_cursor >= duration - 0.5:
                break

            idx = int(t_cursor * SR)

            if sym == 'F':
                # Note: register based on depth, stereo from angle
                register = max(0, min(len(pentatonic) - 1, pitch_idx))
                freq = pentatonic[register]
                # Depth shifts octave
                if depth > 2:
                    freq *= 0.5
                elif depth == 0:
                    freq *= 2.0

                note_dur = time_per_sym * 0.8
                n = int(note_dur * SR)
                if n > 0 and idx + n <= n_samples:
                    # Richer tone for deeper branches
                    harmonics = 1 + depth
                    tone = np.zeros(n)
                    for h in range(1, harmonics + 1):
                        tone += sine(freq * h, note_dur) / (h * h)
                    tone *= env_ar(n, 0.005, note_dur * 0.3) * 0.3

                    # Stereo from angle (normalized)
                    pan = 0.5 + 0.4 * np.sin(angle)
                    audio_l[idx:idx+n] += tone * (1 - pan)
                    audio_r[idx:idx+n] += tone * pan

                t_cursor += time_per_sym

            elif sym == '+':
                angle += np.pi / 5
                pitch_idx = min(pitch_idx + 1, len(pentatonic) - 1)
            elif sym == '-':
                angle -= np.pi / 5
                pitch_idx = max(pitch_idx - 1, 0)
            elif sym == '[':
                stack.append((angle, pitch_idx, depth))
                depth += 1
            elif sym == ']':
                if stack:
                    angle, pitch_idx, depth = stack.pop()
                # Pop sound: soft click
                n = int(0.02 * SR)
                if idx + n <= n_samples:
                    click = np.random.default_rng(int(t_cursor*1000)).normal(0, 0.05, n)
                    click *= env_ar(n, 0.001, 0.015)
                    audio_l[idx:idx+n] += click * 0.3
                    audio_r[idx:idx+n] += click * 0.3

        # Brief pause between generations
        t_cursor += 0.8

    # Subtle background: low drone that swells with each generation
    for i in range(n_samples):
        t = i / SR
        gen_progress = t / duration
        drone_amp = 0.03 * gen_progress
        audio_l[i] += np.sin(2 * np.pi * 55 * t) * drone_amp
        audio_r[i] += np.sin(2 * np.pi * 55 * t) * drone_amp

    audio = np.column_stack([audio_l, audio_r])
    return audio

# ─── Track 3: Pushdown Automaton ───

def pushdown_automaton():
    """PDA matching nested parentheses — the stack as music."""
    duration = 55.0
    n_samples = int(duration * SR)
    audio_l = np.zeros(n_samples)
    audio_r = np.zeros(n_samples)

    # Generate strings of parentheses: some well-formed, some not
    rng = np.random.default_rng(2026)

    def gen_balanced(max_depth, rng):
        """Generate a balanced parenthesis string."""
        s = []
        depth = 0
        length = rng.integers(4, 16)
        for _ in range(length):
            if depth == 0:
                s.append('(')
                depth += 1
            elif depth >= max_depth:
                s.append(')')
                depth -= 1
            else:
                if rng.random() < 0.55:
                    s.append('(')
                    depth += 1
                else:
                    s.append(')')
                    depth -= 1
        while depth > 0:
            s.append(')')
            depth -= 1
        return ''.join(s)

    def gen_unbalanced(rng):
        """Generate an unbalanced string."""
        length = rng.integers(4, 10)
        return ''.join(rng.choice(['(', ')'], size=length))

    strings = []
    for i in range(25):
        if i % 4 == 3:
            strings.append(('unbalanced', gen_unbalanced(rng)))
        else:
            strings.append(('balanced', gen_balanced(4, rng)))

    # Musical mapping
    # Push (open paren): rising pitch, add harmonic layer
    # Pop (close paren): falling pitch, remove harmonic layer
    # Stack depth -> bass drone pitch (deeper = lower)
    base_freq = 220.0  # A3
    push_interval = 1.5  # perfect fifth ratio per level

    t_cursor = 0.3
    time_per_char = 0.25
    pause_between = 0.6

    for kind, pstring in strings:
        if t_cursor > duration - 3.0:
            break

        stack_depth = 0
        max_reached = 0
        valid = True

        for ch in pstring:
            if t_cursor >= duration - 1.5:
                break

            idx = int(t_cursor * SR)
            note_dur = time_per_char * 0.7
            n = int(note_dur * SR)

            if ch == '(':
                stack_depth += 1
                max_reached = max(max_reached, stack_depth)

                # Push: rising freq, bright FM
                freq = base_freq * (push_interval ** (stack_depth - 1))
                if n > 0 and idx + n <= n_samples:
                    # Harmonics = stack depth
                    tone = np.zeros(n)
                    for h in range(1, stack_depth + 1):
                        tone += sine(freq * h, note_dur) / (h * 1.5)
                    # Add FM brightness
                    tone += fm_tone(freq, 2.0, stack_depth * 0.5, note_dur) * 0.3
                    tone *= env_ar(n, 0.008, note_dur * 0.4) * 0.3

                    # Stereo widens with depth
                    spread = min(0.45, stack_depth * 0.1)
                    audio_l[idx:idx+n] += tone * (0.5 + spread)
                    audio_r[idx:idx+n] += tone * (0.5 - spread)

            elif ch == ')':
                if stack_depth <= 0:
                    valid = False
                    # Error: dissonant buzz
                    if n > 0 and idx + n <= n_samples:
                        err = fm_tone(150, 7.3, 8.0, note_dur) * env_ar(n, 0.002, 0.05) * 0.35
                        audio_l[idx:idx+n] += err * 0.5
                        audio_r[idx:idx+n] += err * 0.5
                else:
                    # Pop: falling, pure
                    freq = base_freq * (push_interval ** (stack_depth - 1))
                    stack_depth -= 1

                    if n > 0 and idx + n <= n_samples:
                        tone = sine(freq, note_dur)
                        if stack_depth > 0:
                            tone += sine(freq * 0.5, note_dur) * 0.4
                        tone *= env_ar(n, 0.005, note_dur * 0.5) * 0.3
                        audio_l[idx:idx+n] += tone * 0.5
                        audio_r[idx:idx+n] += tone * 0.5

            # Bass drone tracks stack depth
            drone_n = int(time_per_char * SR)
            if idx + drone_n <= n_samples:
                if stack_depth > 0:
                    drone_freq = 55.0 / (1 + stack_depth * 0.2)
                    t_arr = np.arange(drone_n) / SR
                    drone = np.sin(2 * np.pi * drone_freq * (t_arr + t_cursor)) * 0.08 * min(stack_depth, 4)
                    drone *= env_ar(drone_n, 0.01, 0.01)
                    audio_l[idx:idx+drone_n] += drone
                    audio_r[idx:idx+drone_n] += drone

            t_cursor += time_per_char

        # End of string: resolution or tension
        idx = int(t_cursor * SR)
        resolve_dur = 0.4
        n = int(resolve_dur * SR)

        if valid and stack_depth == 0:
            # Balanced: consonant resolution chord
            if idx + n <= n_samples:
                chord = np.zeros(n)
                for f in [base_freq, base_freq * 1.5, base_freq * 2.0]:
                    chord += sine(f, resolve_dur) * 0.15
                chord *= env_ar(n, 0.01, 0.25)
                audio_l[idx:idx+n] += chord
                audio_r[idx:idx+n] += chord
        else:
            # Unbalanced: unresolved dissonance
            if idx + n <= n_samples:
                dis = fm_tone(base_freq * 1.41, 5.0, 6.0, resolve_dur) * 0.2
                dis *= env_ar(n, 0.01, 0.2)
                audio_l[idx:idx+n] += dis * 0.6
                audio_r[idx:idx+n] += dis * 0.4

        t_cursor += resolve_dur + pause_between

    audio = np.column_stack([audio_l, audio_r])
    return audio

# ─── Main ───

def main():
    make_dirs()

    print("Generating Track 1: Finite Automaton...")
    audio = finite_automaton()
    write_wav("output/auto_1_finite_automaton.wav", audio)
    print(f"  -> output/auto_1_finite_automaton.wav ({len(audio)/SR:.1f}s)")

    print("Generating Track 2: Context-Free Grammar...")
    audio = context_free_grammar()
    write_wav("output/auto_2_context_free_grammar.wav", audio)
    print(f"  -> output/auto_2_context_free_grammar.wav ({len(audio)/SR:.1f}s)")

    print("Generating Track 3: Pushdown Automaton...")
    audio = pushdown_automaton()
    write_wav("output/auto_3_pushdown_automaton.wav", audio)
    print(f"  -> output/auto_3_pushdown_automaton.wav ({len(audio)/SR:.1f}s)")

    print("Done! All tracks in output/")

if __name__ == "__main__":
    main()
