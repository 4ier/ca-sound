#!/usr/bin/env python3
"""Phase 21: Compiler Pipeline -- lexing, parsing, code generation.

Three pieces:
1. Lexer (55s, stereo) -- Raw source code tokenized into a stream of classified tokens.
   Each character class has a distinct timbre: keywords=rich FM brass, identifiers=pure sine,
   operators=percussive clicks, literals=warm harmonics, whitespace=silence.
   Characters stream in left-to-right (stereo pan), tokens coalesce into rhythmic phrases.
   A simple expression language: let x = (3 + y) * f(z, 10); if x > 0 then x else -x
   Token boundaries marked by 2kHz clicks. Nested depth -> bass drone.

2. Parser (50s, stereo) -- Token stream builds an AST via recursive descent.
   Each grammar rule = a musical phrase pattern. Terminals = leaf notes (short, clear).
   Non-terminals = phrase containers (onset chord marks entry, resolution marks exit).
   Recursive descent depth -> octave shifts downward. Shift-reduce conflicts =
   momentary dissonance before resolution. The tree grows bottom-up as harmonics stack.

3. Code Generation (55s, stereo) -- AST walks produce machine instructions.
   High-level nodes dissolve into sequences of low-level ops. Each opcode = distinct
   timbre: LOAD=sine, STORE=reverse env, ADD=bright FM, MUL=deep FM, CMP=click,
   JMP=sweep, CALL=arpeggiated chord, RET=descending resolve.
   Register allocation = voice assignment (8 voices). Stack operations = pitch shifts.
   Final output: the "compiled" sequence plays as a coherent rhythmic machine.
"""
import numpy as np
import os

SR = 44100


def write_wav(path, data):
    import wave
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    data = np.clip(data, -1, 1)
    pcm = (data * 32767).astype(np.int16)
    with wave.open(path, "w") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"  -> {path} ({len(data)/SR:.1f}s)")


def sine(freq, t, phase=0.0):
    return np.sin(2 * np.pi * freq * t + phase)


def fm_tone(carrier, mod_freq, mod_depth, t):
    return np.sin(2 * np.pi * carrier * t + mod_depth * np.sin(2 * np.pi * mod_freq * t))


def envelope(n, attack=0.01, release=0.05):
    env = np.ones(n)
    att = int(attack * SR)
    rel = int(release * SR)
    if att > 0:
        env[:min(att, n)] = np.linspace(0, 1, min(att, n))
    if rel > 0 and rel < n:
        env[-rel:] = np.linspace(1, 0, rel)
    return env


def click(n=200, freq=2000):
    t_arr = np.arange(n) / SR
    return np.sin(2 * np.pi * freq * t_arr) * np.exp(-t_arr * 40)


# Token types and their audio signatures
TOKEN_TYPES = {
    'keyword':    {'base_freq': 220, 'harmonics': 6, 'fm_depth': 3.0, 'attack': 0.005},
    'identifier': {'base_freq': 330, 'harmonics': 2, 'fm_depth': 0.0, 'attack': 0.01},
    'operator':   {'base_freq': 880, 'harmonics': 1, 'fm_depth': 5.0, 'attack': 0.001},
    'literal':    {'base_freq': 440, 'harmonics': 4, 'fm_depth': 0.5, 'attack': 0.008},
    'paren':      {'base_freq': 1200, 'harmonics': 1, 'fm_depth': 0.0, 'attack': 0.001},
    'separator':  {'base_freq': 660, 'harmonics': 1, 'fm_depth': 1.0, 'attack': 0.003},
}

# Source code to tokenize
SOURCE_LINES = [
    "let x = (3 + y) * f(z, 10);",
    "if x > 0 then x else -x;",
    "let sq = fn(n) => n * n;",
    "let result = sq(x) + sq(y);",
]

def tokenize(source):
    """Simple tokenizer for our expression language."""
    tokens = []
    keywords = {'let', 'if', 'then', 'else', 'fn'}
    operators = {'+', '-', '*', '/', '>', '<', '=', '=>'}
    i = 0
    while i < len(source):
        c = source[i]
        if c.isspace():
            i += 1
            continue
        if c in '()':
            tokens.append(('paren', c, i))
            i += 1
        elif c in ',;':
            tokens.append(('separator', c, i))
            i += 1
        elif c == '=' and i + 1 < len(source) and source[i+1] == '>':
            tokens.append(('operator', '=>', i))
            i += 2
        elif c in '+-*/><=':
            tokens.append(('operator', c, i))
            i += 1
        elif c.isdigit():
            j = i
            while j < len(source) and source[j].isdigit():
                j += 1
            tokens.append(('literal', source[i:j], i))
            i = j
        elif c.isalpha() or c == '_':
            j = i
            while j < len(source) and (source[j].isalnum() or source[j] == '_'):
                j += 1
            word = source[i:j]
            if word in keywords:
                tokens.append(('keyword', word, i))
            else:
                tokens.append(('identifier', word, i))
            i = j
        else:
            i += 1
    return tokens


def make_lexer_piece():
    """Piece 1: Lexer -- source code to token stream."""
    print("Generating Lexer...")
    duration = 55.0
    n_total = int(duration * SR)
    out_l = np.zeros(n_total)
    out_r = np.zeros(n_total)

    # Drone at D1 (36.7 Hz)
    t_all = np.arange(n_total) / SR
    drone_freq = 36.7
    drone = 0.08 * sine(drone_freq, t_all) + 0.04 * sine(drone_freq * 2, t_all)
    drone *= envelope(n_total, attack=2.0, release=3.0)
    out_l += drone
    out_r += drone

    # Process all source lines
    all_tokens = []
    for line in SOURCE_LINES:
        all_tokens.extend(tokenize(line))

    n_tokens = len(all_tokens)
    # Spread tokens across 50s (leave 5s for fade)
    active_dur = 50.0
    token_spacing = active_dur / n_tokens

    rng = np.random.default_rng(42)

    for idx, (ttype, value, char_pos) in enumerate(all_tokens):
        sig = TOKEN_TYPES.get(ttype, TOKEN_TYPES['identifier'])
        t_start = 1.0 + idx * token_spacing
        # Token duration proportional to value length
        tok_dur = max(0.08, min(0.5, len(value) * 0.06))
        n_tok = int(tok_dur * SR)
        i_start = int(t_start * SR)
        if i_start + n_tok > n_total:
            break

        t_tok = np.arange(n_tok) / SR
        freq = sig['base_freq'] * (1.0 + 0.1 * rng.standard_normal())

        # Build tone based on token type
        tone = np.zeros(n_tok)
        for h in range(1, sig['harmonics'] + 1):
            amp = 1.0 / h
            if sig['fm_depth'] > 0:
                tone += amp * fm_tone(freq * h, freq * 0.5, sig['fm_depth'] / h, t_tok)
            else:
                tone += amp * sine(freq * h, t_tok)

        tone *= envelope(n_tok, attack=sig['attack'], release=0.03)
        tone *= 0.15

        # Stereo position based on character position in source
        max_pos = max(len(line) for line in SOURCE_LINES)
        pan = np.clip(char_pos / max(max_pos, 1), 0, 1)
        out_l[i_start:i_start + n_tok] += tone * (1 - pan)
        out_r[i_start:i_start + n_tok] += tone * pan

        # Token boundary click
        c = click(150, 2000) * 0.05
        cn = len(c)
        if i_start + cn < n_total:
            out_l[i_start:i_start + cn] += c
            out_r[i_start:i_start + cn] += c

    out = np.stack([out_l, out_r], axis=1)
    out *= envelope(n_total, attack=0.5, release=2.0)[:, None]
    return out


# AST node types for parser piece
class ASTNode:
    def __init__(self, ntype, value=None, children=None):
        self.ntype = ntype
        self.value = value
        self.children = children or []
        self.depth = 0

    def set_depths(self, d=0):
        self.depth = d
        for c in self.children:
            c.set_depths(d + 1)


def build_sample_ast():
    """Build AST for: let x = (3 + y) * f(z, 10)"""
    lit3 = ASTNode('literal', '3')
    idy = ASTNode('identifier', 'y')
    add = ASTNode('binop', '+', [lit3, idy])
    idz = ASTNode('identifier', 'z')
    lit10 = ASTNode('literal', '10')
    call_f = ASTNode('call', 'f', [idz, lit10])
    mul = ASTNode('binop', '*', [add, call_f])
    idx = ASTNode('identifier', 'x')
    let_stmt = ASTNode('let', 'x', [idx, mul])

    # if x > 0 then x else -x
    id_x2 = ASTNode('identifier', 'x')
    lit0 = ASTNode('literal', '0')
    cmp = ASTNode('binop', '>', [id_x2, lit0])
    id_x3 = ASTNode('identifier', 'x')
    id_x4 = ASTNode('identifier', 'x')
    neg = ASTNode('unaryop', '-', [id_x4])
    if_expr = ASTNode('if', None, [cmp, id_x3, neg])

    program = ASTNode('program', None, [let_stmt, if_expr])
    program.set_depths()
    return program


# Node type -> audio signature for parser
PARSE_SIGS = {
    'program':    {'freq': 55,  'harmonics': 8, 'fm': 0.0},
    'let':        {'freq': 110, 'harmonics': 5, 'fm': 1.5},
    'if':         {'freq': 146, 'harmonics': 4, 'fm': 2.0},
    'binop':      {'freq': 220, 'harmonics': 3, 'fm': 1.0},
    'unaryop':    {'freq': 293, 'harmonics': 2, 'fm': 2.5},
    'call':       {'freq': 330, 'harmonics': 4, 'fm': 0.5},
    'identifier': {'freq': 440, 'harmonics': 1, 'fm': 0.0},
    'literal':    {'freq': 392, 'harmonics': 2, 'fm': 0.3},
}


def flatten_ast(node, result=None):
    """DFS flatten AST into visit order."""
    if result is None:
        result = []
    result.append(node)
    for c in node.children:
        flatten_ast(c, result)
    return result


def make_parser_piece():
    """Piece 2: Parser -- token stream builds AST."""
    print("Generating Parser...")
    duration = 50.0
    n_total = int(duration * SR)
    out_l = np.zeros(n_total)
    out_r = np.zeros(n_total)
    t_all = np.arange(n_total) / SR

    # Bass drone C1
    drone = 0.07 * sine(32.7, t_all) + 0.04 * sine(65.4, t_all)
    drone *= envelope(n_total, attack=2.0, release=3.0)
    out_l += drone
    out_r += drone

    ast = build_sample_ast()
    nodes = flatten_ast(ast)
    n_nodes = len(nodes)
    active_dur = 45.0
    node_spacing = active_dur / n_nodes
    max_depth = max(n.depth for n in nodes)

    rng = np.random.default_rng(123)

    for idx, node in enumerate(nodes):
        sig = PARSE_SIGS.get(node.ntype, PARSE_SIGS['identifier'])
        t_start = 1.5 + idx * node_spacing
        # Duration: non-terminals longer than terminals
        is_leaf = len(node.children) == 0
        tok_dur = 0.3 if is_leaf else 0.6 + 0.1 * len(node.children)
        n_tok = int(tok_dur * SR)
        i_start = int(t_start * SR)
        if i_start + n_tok > n_total:
            break

        t_tok = np.arange(n_tok) / SR
        # Octave shift based on depth
        depth_shift = 2.0 ** (-node.depth * 0.3)
        freq = sig['freq'] * depth_shift

        tone = np.zeros(n_tok)
        for h in range(1, sig['harmonics'] + 1):
            amp = 1.0 / (h * 1.2)
            if sig['fm'] > 0:
                tone += amp * fm_tone(freq * h, freq * 0.7, sig['fm'] / h, t_tok)
            else:
                tone += amp * sine(freq * h, t_tok)

        # Non-terminals get onset chord (entry) and resolution (exit)
        if not is_leaf:
            # Entry: rising brightness
            brightness = np.linspace(0.3, 1.0, n_tok)
            tone *= brightness
            # Add "container" resonance
            container_freq = freq * 1.5  # fifth above
            tone += 0.08 * sine(container_freq, t_tok) * envelope(n_tok, 0.01, tok_dur * 0.7)

        tone *= envelope(n_tok, attack=0.005 if is_leaf else 0.02, release=0.05)
        tone *= 0.12

        # Stereo: depth -> narrow center for deep nodes
        width = 1.0 - (node.depth / max(max_depth, 1)) * 0.7
        pan = 0.5 + (rng.random() - 0.5) * width
        out_l[i_start:i_start + n_tok] += tone * (1 - pan)
        out_r[i_start:i_start + n_tok] += tone * pan

        # Node transition click
        c = click(120, 1500 + node.depth * 200) * 0.04
        cn = len(c)
        if i_start + cn < n_total:
            out_l[i_start:i_start + cn] += c
            out_r[i_start:i_start + cn] += c

    # Final chord: all unique frequencies from the AST, fading together
    coda_start = int(46.0 * SR)
    coda_dur = int(3.5 * SR)
    if coda_start + coda_dur <= n_total:
        t_coda = np.arange(coda_dur) / SR
        coda = np.zeros(coda_dur)
        seen_freqs = set()
        for node in nodes:
            sig = PARSE_SIGS.get(node.ntype, PARSE_SIGS['identifier'])
            f = sig['freq']
            if f not in seen_freqs:
                seen_freqs.add(f)
                coda += 0.03 * sine(f, t_coda)
        coda *= envelope(coda_dur, attack=0.5, release=2.0)
        out_l[coda_start:coda_start + coda_dur] += coda
        out_r[coda_start:coda_start + coda_dur] += coda

    out = np.stack([out_l, out_r], axis=1)
    out *= envelope(n_total, attack=0.5, release=2.0)[:, None]
    return out


# Opcodes for code generation piece
OPCODES = {
    'LOAD':  {'freq': 220, 'harmonics': 2, 'fm': 0.0, 'dur': 0.12},
    'STORE': {'freq': 185, 'harmonics': 2, 'fm': 0.0, 'dur': 0.15},
    'ADD':   {'freq': 330, 'harmonics': 3, 'fm': 2.0, 'dur': 0.10},
    'MUL':   {'freq': 165, 'harmonics': 4, 'fm': 3.0, 'dur': 0.12},
    'CMP':   {'freq': 1000, 'harmonics': 1, 'fm': 0.0, 'dur': 0.06},
    'JMP':   {'freq': 440, 'harmonics': 1, 'fm': 5.0, 'dur': 0.20},
    'JZ':    {'freq': 500, 'harmonics': 1, 'fm': 4.0, 'dur': 0.18},
    'CALL':  {'freq': 293, 'harmonics': 5, 'fm': 1.0, 'dur': 0.25},
    'RET':   {'freq': 370, 'harmonics': 3, 'fm': 0.5, 'dur': 0.20},
    'PUSH':  {'freq': 262, 'harmonics': 2, 'fm': 0.0, 'dur': 0.08},
    'POP':   {'freq': 294, 'harmonics': 2, 'fm': 0.0, 'dur': 0.08},
    'NEG':   {'freq': 392, 'harmonics': 2, 'fm': 1.5, 'dur': 0.10},
}


def generate_instruction_sequence():
    """Generate instruction sequence for: let x = (3+y)*f(z,10); if x>0 then x else -x"""
    return [
        # let x = (3 + y) * f(z, 10)
        ('LOAD', 'lit:3', 0),     # load 3
        ('LOAD', 'var:y', 1),     # load y
        ('ADD', None, 2),         # 3 + y
        ('PUSH', None, 3),        # save result
        ('LOAD', 'var:z', 4),     # load z
        ('PUSH', None, 5),        # push arg z
        ('LOAD', 'lit:10', 6),    # load 10
        ('PUSH', None, 7),        # push arg 10
        ('CALL', 'f', 8),         # call f(z, 10)
        ('POP', None, 9),         # restore (3+y)
        ('MUL', None, 10),        # (3+y) * f(z,10)
        ('STORE', 'var:x', 11),   # store to x
        # if x > 0 then x else -x
        ('LOAD', 'var:x', 12),    # load x
        ('LOAD', 'lit:0', 13),    # load 0
        ('CMP', '>', 14),         # compare x > 0
        ('JZ', 'else', 15),       # jump if false
        ('LOAD', 'var:x', 16),    # then: load x
        ('JMP', 'end', 17),       # jump to end
        ('LOAD', 'var:x', 18),    # else: load x
        ('NEG', None, 19),        # negate
        ('RET', None, 20),        # return result
    ]


def make_codegen_piece():
    """Piece 3: Code Generation -- AST to machine instructions."""
    print("Generating Code Generation...")
    duration = 55.0
    n_total = int(duration * SR)
    out_l = np.zeros(n_total)
    out_r = np.zeros(n_total)
    t_all = np.arange(n_total) / SR

    # Very low drone A0 (27.5 Hz) -- the machine hum
    drone = 0.06 * sine(27.5, t_all) + 0.03 * sine(55.0, t_all)
    drone *= envelope(n_total, attack=1.5, release=3.0)
    out_l += drone
    out_r += drone

    instructions = generate_instruction_sequence()
    n_instr = len(instructions)
    active_dur = 48.0
    instr_spacing = active_dur / n_instr

    # 8 "registers" -> 8 stereo positions
    reg_pan = np.linspace(0.15, 0.85, 8)
    rng = np.random.default_rng(77)

    stack_depth = 0

    for idx, (opcode, operand, order) in enumerate(instructions):
        sig = OPCODES.get(opcode, OPCODES['LOAD'])
        t_start = 2.0 + idx * instr_spacing
        tok_dur = sig['dur'] * (1.5 + 0.5 * rng.random())
        n_tok = int(tok_dur * SR)
        i_start = int(t_start * SR)
        if i_start + n_tok > n_total:
            break

        t_tok = np.arange(n_tok) / SR
        freq = sig['freq']

        # Track stack for PUSH/POP pitch shifts
        if opcode == 'PUSH':
            stack_depth += 1
            freq *= (1.0 + stack_depth * 0.05)
        elif opcode == 'POP':
            stack_depth = max(0, stack_depth - 1)
            freq *= (1.0 + stack_depth * 0.05)

        # Build tone
        tone = np.zeros(n_tok)
        for h in range(1, sig['harmonics'] + 1):
            amp = 1.0 / (h * 1.1)
            if sig['fm'] > 0:
                tone += amp * fm_tone(freq * h, freq * 0.6, sig['fm'] / h, t_tok)
            else:
                tone += amp * sine(freq * h, t_tok)

        # Special opcode behaviors
        if opcode == 'JMP' or opcode == 'JZ':
            # Frequency sweep
            sweep = np.linspace(freq, freq * 1.5, n_tok)
            sweep_tone = np.sin(2 * np.pi * np.cumsum(sweep) / SR)
            tone = 0.7 * tone + 0.3 * sweep_tone
        elif opcode == 'CALL':
            # Arpeggiated chord: root, third, fifth
            for interval, delay_frac in [(1.0, 0.0), (1.25, 0.2), (1.5, 0.4)]:
                delay_n = int(delay_frac * n_tok)
                arp_n = n_tok - delay_n
                if arp_n > 0:
                    t_arp = np.arange(arp_n) / SR
                    arp = 0.08 * sine(freq * interval, t_arp) * envelope(arp_n, 0.005, 0.05)
                    tone[delay_n:delay_n + arp_n] += arp
        elif opcode == 'RET':
            # Descending resolution
            desc = np.linspace(freq * 1.2, freq * 0.8, n_tok)
            desc_tone = 0.1 * np.sin(2 * np.pi * np.cumsum(desc) / SR)
            tone += desc_tone
        elif opcode == 'STORE':
            # Reverse envelope (attack at end)
            rev_env = np.flip(envelope(n_tok, attack=0.01, release=0.08))
            tone *= rev_env / (envelope(n_tok, attack=0.005, release=0.03) + 1e-10)

        tone *= envelope(n_tok, attack=0.003, release=0.02)
        tone *= 0.14

        # Register assignment -> stereo position
        reg = order % 8
        pan = reg_pan[reg]
        out_l[i_start:i_start + n_tok] += tone * (1 - pan)
        out_r[i_start:i_start + n_tok] += tone * pan

        # Instruction click
        c = click(100, 2500) * 0.03
        cn = len(c)
        if i_start + cn < n_total:
            out_l[i_start:i_start + cn] += c * 0.7
            out_r[i_start:i_start + cn] += c * 0.7

    # Final: "compiled program" plays back as rhythmic machine
    # All unique opcode frequencies as a mechanical chord
    coda_start = int(50.0 * SR)
    coda_dur = int(4.0 * SR)
    if coda_start + coda_dur <= n_total:
        t_coda = np.arange(coda_dur) / SR
        coda = np.zeros(coda_dur)
        used_ops = set(op for op, _, _ in instructions)
        for op in sorted(used_ops):
            sig = OPCODES[op]
            coda += 0.02 * sine(sig['freq'], t_coda)
        # Add a steady pulse -- the machine clock
        pulse_freq = 4.0  # 4 Hz clock
        pulse = 0.03 * (np.sin(2 * np.pi * pulse_freq * t_coda) > 0.8).astype(float)
        pulse_audio = np.zeros(coda_dur)
        for i in range(len(t_coda)):
            if pulse[i] > 0:
                remaining = min(200, coda_dur - i)
                t_p = np.arange(remaining) / SR
                pulse_audio[i:i+remaining] += 0.05 * np.sin(2 * np.pi * 1000 * t_p) * np.exp(-t_p * 30)
        coda += pulse_audio
        coda *= envelope(coda_dur, attack=0.3, release=2.5)
        out_l[coda_start:coda_start + coda_dur] += coda
        out_r[coda_start:coda_start + coda_dur] += coda

    out = np.stack([out_l, out_r], axis=1)
    out *= envelope(n_total, attack=0.5, release=2.0)[:, None]
    return out


def main():
    os.makedirs("output", exist_ok=True)

    print("=== Phase 21: Compiler Pipeline ===\n")

    data = make_lexer_piece()
    write_wav("output/comp_1_lexer.wav", data)

    data = make_parser_piece()
    write_wav("output/comp_2_parser.wav", data)

    data = make_codegen_piece()
    write_wav("output/comp_3_codegen.wav", data)

    print("\nDone!")


if __name__ == "__main__":
    main()
