#!/usr/bin/env python3
"""Phase 8: Graph Algorithms — Traversal as Music

Three pieces exploring how different graph traversal strategies
create different musical narratives from the same underlying structure.

1. Dijkstra's Meditation (55s) — Shortest path relaxation as harmonic settling.
   Each node relaxation lowers tension; the wavefront expands outward like
   a drop in still water. Settled nodes form a growing drone chord.

2. BFS Waves (50s) — Breadth-first search as rhythmic waves. Each BFS level
   is a musical phrase; nodes at the same depth sound together as chords.
   Width of each level → chord density. Stereo position from graph layout.

3. DFS Descent (50s) — Depth-first search as a solo melodic line diving deep
   then backtracking. Stack depth → pitch (deeper = lower). Backtrack steps
   are heard as ascending returns. Discovery vs. backtrack = two timbres.
"""

import numpy as np
import struct, wave, os, random, math
from collections import defaultdict
import heapq

SR = 44100

def write_wav(fname, data, sr=SR):
    """Write mono or stereo float array to 16-bit WAV."""
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", fname)
    if data.ndim == 1:
        channels = 1
        samples = data
    else:
        channels = 2
        samples = data.flatten('C')  # interleaved
    samples = np.clip(samples, -1, 1)
    samples = (samples * 32767).astype(np.int16)
    with wave.open(path, 'w') as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())
    print(f"  Written: {path} ({len(data)/sr:.1f}s)")

def fade(sig, fade_in=0.01, fade_out=0.01, sr=SR):
    n_in = int(fade_in * sr)
    n_out = int(fade_out * sr)
    if sig.ndim == 1:
        if n_in > 0:
            sig[:n_in] *= np.linspace(0, 1, n_in)
        if n_out > 0:
            sig[-n_out:] *= np.linspace(1, 0, n_out)
    else:
        for ch in range(sig.shape[1]):
            if n_in > 0:
                sig[:n_in, ch] *= np.linspace(0, 1, n_in)
            if n_out > 0:
                sig[-n_out:, ch] *= np.linspace(1, 0, n_out)
    return sig


# --- Graph generation ---

def make_grid_graph(rows, cols, seed=42):
    """Create a weighted grid graph with some random diagonal edges.
    Returns adjacency dict {node: [(neighbor, weight), ...]}, positions {node: (x, y)}."""
    rng = random.Random(seed)
    adj = defaultdict(list)
    pos = {}
    
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            pos[node] = (c / max(1, cols-1), r / max(1, rows-1))
            # right
            if c < cols - 1:
                w = rng.uniform(0.5, 3.0)
                adj[node].append((node + 1, w))
                adj[node + 1].append((node, w))
            # down
            if r < rows - 1:
                w = rng.uniform(0.5, 3.0)
                adj[node].append((node + cols, w))
                adj[node + cols].append((node, w))
            # random diagonal
            if r < rows - 1 and c < cols - 1 and rng.random() < 0.3:
                w = rng.uniform(1.0, 4.0)
                diag = node + cols + 1
                adj[node].append((diag, w))
                adj[diag].append((node, w))
    
    return dict(adj), pos

def make_tree_graph(depth=6, branching=3, seed=42):
    """Create a tree graph for DFS exploration."""
    rng = random.Random(seed)
    adj = defaultdict(list)
    pos = {}
    node_id = 0
    pos[0] = (0.5, 0.0)
    
    def build(parent, d, x_min, x_max):
        nonlocal node_id
        if d >= depth:
            return
        n_children = rng.randint(1, branching)
        width = (x_max - x_min) / n_children
        for i in range(n_children):
            node_id += 1
            child = node_id
            cx = x_min + width * (i + 0.5)
            pos[child] = (cx, d / depth)
            w = rng.uniform(0.5, 2.0)
            adj[parent].append((child, w))
            adj[child].append((parent, w))
            build(child, d + 1, x_min + width * i, x_min + width * (i + 1))
    
    build(0, 1, 0.0, 1.0)
    return dict(adj), pos


# --- Synthesis helpers ---

def sine_tone(freq, duration, sr=SR):
    t = np.arange(int(duration * sr)) / sr
    return np.sin(2 * np.pi * freq * t)

def rich_tone(freq, duration, n_harmonics=4, sr=SR):
    """Sine with harmonics, each -6dB."""
    t = np.arange(int(duration * sr)) / sr
    sig = np.zeros_like(t)
    for h in range(1, n_harmonics + 1):
        sig += (0.5 ** (h - 1)) * np.sin(2 * np.pi * freq * h * t)
    return sig / np.max(np.abs(sig) + 1e-10)

def fm_tone(freq, duration, mod_ratio=2.0, mod_depth=1.0, sr=SR):
    t = np.arange(int(duration * sr)) / sr
    mod = mod_depth * freq * np.sin(2 * np.pi * freq * mod_ratio * t)
    return np.sin(2 * np.pi * freq * t + mod)

def node_to_freq(node, n_nodes, base=220, top=880):
    """Map node id to frequency in a musical range."""
    ratio = node / max(1, n_nodes - 1)
    return base * (top / base) ** ratio

def pos_to_pan(x):
    """x in [0,1] → (left_gain, right_gain)."""
    return (math.cos(x * math.pi / 2), math.sin(x * math.pi / 2))


# --- Piece 1: Dijkstra's Meditation ---

def dijkstra_meditation():
    """Shortest path relaxation as harmonic settling."""
    print("Generating: Dijkstra's Meditation...")
    
    rows, cols = 6, 8
    adj, pos = make_grid_graph(rows, cols, seed=42)
    n_nodes = rows * cols
    source = 0
    duration = 55.0
    
    # Run Dijkstra, record events
    dist = {source: 0.0}
    settled = []
    relaxations = []  # (time_order, node, old_dist, new_dist)
    pq = [(0.0, source)]
    visited = set()
    order = 0
    
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        settled.append((order, u, d))
        order += 1
        if u in adj:
            for v, w in adj[u]:
                nd = d + w
                if v not in dist or nd < dist[v]:
                    old = dist.get(v, float('inf'))
                    dist[v] = nd
                    relaxations.append((order, v, old, nd))
                    heapq.heappush(pq, (nd, v))
    
    total_events = len(settled)
    out = np.zeros((int(duration * SR), 2))
    
    # Each settlement event → a tone added to the growing drone
    # Plus relaxation clicks
    for i, (_, node, d) in enumerate(settled):
        t_start = (i / total_events) * (duration - 3.0)
        t_end = duration
        freq = node_to_freq(node, n_nodes, 110, 660)
        
        # Settled nodes sustain as quiet drone
        n_samples = int((t_end - t_start) * SR)
        start_idx = int(t_start * SR)
        
        t = np.arange(n_samples) / SR
        # Envelope: quick attack, long sustain with slow decay
        env = np.exp(-t / (duration * 0.7)) * 0.15
        tone = rich_tone(freq, t_end - t_start, n_harmonics=3) * env
        
        # Stereo from position
        px, py = pos[node]
        lg, rg = pos_to_pan(px)
        
        end_idx = start_idx + len(tone)
        if end_idx > len(out):
            tone = tone[:len(out) - start_idx]
            end_idx = len(out)
        
        out[start_idx:end_idx, 0] += tone * lg
        out[start_idx:end_idx, 1] += tone * rg
        
        # Relaxation click: short FM burst at settlement moment
        click_dur = 0.08
        click = fm_tone(freq * 1.5, click_dur, mod_ratio=3, mod_depth=2) * 0.3
        click *= np.exp(-np.arange(len(click)) / (SR * 0.02))
        ce = min(start_idx + len(click), len(out))
        cl = ce - start_idx
        out[start_idx:ce, 0] += click[:cl] * lg
        out[start_idx:ce, 1] += click[:cl] * rg
    
    # Normalize
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.85
    
    fade(out, 0.05, 1.0)
    write_wav("graph_1_dijkstra_meditation.wav", out)


# --- Piece 2: BFS Waves ---

def bfs_waves():
    """Breadth-first search as rhythmic waves."""
    print("Generating: BFS Waves...")
    
    rows, cols = 7, 7
    adj, pos = make_grid_graph(rows, cols, seed=123)
    n_nodes = rows * cols
    source = 24  # center node
    duration = 50.0
    
    # BFS with level tracking
    levels = defaultdict(list)
    visited = {source}
    queue = [(source, 0)]
    idx = 0
    while idx < len(queue):
        u, d = queue[idx]
        idx += 1
        levels[d].append(u)
        if u in adj:
            for v, _ in adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append((v, d + 1))
    
    max_level = max(levels.keys())
    out = np.zeros((int(duration * SR), 2))
    
    # Each level = one musical phrase / wave
    level_dur = duration / (max_level + 1.5)
    
    # Scale: pentatonic for pleasant waves
    penta = [0, 2, 4, 7, 9]  # semitones
    base_freq = 220
    
    for level, nodes in sorted(levels.items()):
        t_start = level * level_dur
        phrase_dur = level_dur * 0.9
        
        # Nodes at same level: arpeggiate quickly then sustain as chord
        n = len(nodes)
        arp_time = min(0.3, phrase_dur * 0.3)  # time to arpeggiate
        sustain_time = phrase_dur - arp_time
        
        for j, node in enumerate(sorted(nodes)):
            # Pitch from pentatonic scale based on position
            px, py = pos[node]
            scale_idx = int(px * 12) % len(penta)
            octave = 1 + int(py * 2)
            semitone = penta[scale_idx] + 12 * octave
            freq = base_freq * (2 ** (semitone / 12))
            
            # Arpeggio onset
            onset = t_start + (j / max(n, 1)) * arp_time
            tone_dur = phrase_dur - (onset - t_start)
            
            t = np.arange(int(tone_dur * SR)) / SR
            # Wave-like envelope: swell then fade
            env = np.sin(np.pi * t / tone_dur) ** 0.5 * 0.12
            
            # Brighter tone for outer levels (more harmonics)
            n_harm = min(2 + level, 6)
            tone = rich_tone(freq, tone_dur, n_harm) * env
            
            lg, rg = pos_to_pan(px)
            si = int(onset * SR)
            ei = min(si + len(tone), len(out))
            tl = ei - si
            out[si:ei, 0] += tone[:tl] * lg
            out[si:ei, 1] += tone[:tl] * rg
        
        # Level marker: low pulse
        pulse_freq = 55 * (1 + level * 0.2)
        pulse_dur = 0.15
        pulse = sine_tone(pulse_freq, pulse_dur) * 0.2
        pulse *= np.exp(-np.arange(len(pulse)) / (SR * 0.05))
        pi = int(t_start * SR)
        pe = min(pi + len(pulse), len(out))
        pl = pe - pi
        out[pi:pe, 0] += pulse[:pl] * 0.7
        out[pi:pe, 1] += pulse[:pl] * 0.7
    
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.85
    fade(out, 0.05, 0.8)
    write_wav("graph_2_bfs_waves.wav", out)


# --- Piece 3: DFS Descent ---

def dfs_descent():
    """Depth-first search as a solo melodic line."""
    print("Generating: DFS Descent...")
    
    adj, pos = make_tree_graph(depth=7, branching=3, seed=77)
    n_nodes = len(pos)
    duration = 50.0
    
    # DFS with event recording
    events = []  # (node, depth, 'discover'|'backtrack')
    visited = set()
    
    def dfs(u, depth):
        visited.add(u)
        events.append((u, depth, 'discover'))
        if u in adj:
            children = [(v, w) for v, w in adj[u] if v not in visited]
            # Sort by position for consistent left-to-right
            children.sort(key=lambda vw: pos[vw[0]][0])
            for v, w in children:
                if v not in visited:
                    dfs(v, depth + 1)
                    events.append((u, depth, 'backtrack'))
    
    dfs(0, 0)
    
    total = len(events)
    out = np.zeros((int(duration * SR), 2))
    
    # Map depth to pitch: deeper = lower
    max_depth = max(e[1] for e in events)
    
    # Musical scale: natural minor for melancholy exploration
    minor = [0, 2, 3, 5, 7, 8, 10]
    base_midi = 72  # C5 at depth 0, descending
    
    note_dur = min(0.4, (duration - 1) / total)
    
    for i, (node, depth, kind) in enumerate(events):
        t_start = (i / total) * (duration - 2.0)
        
        # Depth → pitch: 0 = high, max = low
        octave_drop = depth * 12 / max(max_depth, 1)
        scale_idx = depth % len(minor)
        midi = base_midi - octave_drop + minor[scale_idx] / 12 * 3
        freq = 440 * 2 ** ((midi - 69) / 12)
        freq = max(55, min(freq, 2000))
        
        px, _ = pos[node]
        lg, rg = pos_to_pan(px)
        
        if kind == 'discover':
            # Discovery: bright FM tone, descending
            dur = note_dur * 0.9
            tone = fm_tone(freq, dur, mod_ratio=1.5, mod_depth=0.5 + depth * 0.1)
            env = np.exp(-np.arange(len(tone)) / (SR * 0.15)) * 0.35
            tone = tone * env
        else:
            # Backtrack: pure sine, softer, ascending feel
            dur = note_dur * 0.6
            tone = sine_tone(freq * 1.02, dur)  # slightly sharp = tension
            env = np.exp(-np.arange(len(tone)) / (SR * 0.1)) * 0.2
            tone = tone * env
        
        si = int(t_start * SR)
        ei = min(si + len(tone), len(out))
        tl = ei - si
        out[si:ei, 0] += tone[:tl] * lg
        out[si:ei, 1] += tone[:tl] * rg
    
    # Subtle low drone throughout: root note
    drone_t = np.arange(len(out)) / SR
    drone = sine_tone(55, duration) * 0.08
    drone *= np.sin(np.pi * drone_t / duration)  # fade in/out
    out[:, 0] += drone
    out[:, 1] += drone
    
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.85
    fade(out, 0.05, 1.0)
    write_wav("graph_3_dfs_descent.wav", out)


if __name__ == "__main__":
    print("=== Phase 8: Graph Algorithms ===")
    dijkstra_meditation()
    bfs_waves()
    dfs_descent()
    print("Done!")
