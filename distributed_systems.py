#!/usr/bin/env python3
"""Phase 19: Distributed Systems -- consensus, gossip, vector clocks.

Three pieces:
1. Raft Consensus (60s, stereo) -- Leader election, heartbeats, split brain.
   5 nodes form a cluster. Terms begin with election timeout silence,
   then a leader emerges with strong heartbeat pulse. Log replication
   as call-and-response. Network partition splits the cluster --
   two leaders briefly coexist (dissonance), then partition heals
   and the cluster reconverges to a single leader.
2. Gossip Protocol (55s, stereo) -- Epidemic information spreading.
   7 nodes in a ring/mesh. One node receives a "rumor" (bright tone).
   Each tick, infected nodes randomly contact neighbors -- the rumor
   spreads exponentially. Audio: infected=rich harmonics, naive=thin sine.
   Spatial position = stereo. The S-curve of adoption becomes a crescendo.
3. Vector Clocks (55s, stereo) -- Causality and partial ordering.
   4 processes with independent clock rates (polyrhythm). Messages
   between processes synchronize clocks -- frequency convergence on
   communication, divergence during isolation. Concurrent events =
   simultaneous but unrelated tones. Causal chains = melodic sequences.
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
    with wave.open(path, 'w') as w:
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


def crossfade(n):
    return np.linspace(0, 1, n)


def click(n=200, freq=2000):
    t = np.arange(n) / SR
    env = np.exp(-t * 40)
    return env * np.sin(2 * np.pi * freq * t)

# ── Piece 1: Raft Consensus ──────────────────────────────────────────

def raft_consensus():
    """5-node Raft cluster: election, heartbeat, partition, reconvergence."""
    np.random.seed(19)
    dur = 60.0
    n = int(dur * SR)
    out = np.zeros((n, 2))
    t = np.arange(n) / SR

    # 5 nodes: frequencies based on pentatonic (D minor feel)
    node_freqs = np.array([146.83, 174.61, 196.00, 220.00, 261.63])  # D3,F3,G3,A3,C4
    node_pans = np.array([-0.7, -0.35, 0.0, 0.35, 0.7])  # stereo spread

    # 55Hz drone throughout
    drone = 0.08 * sine(55, t) * envelope(n, attack=2.0, release=3.0)
    out[:, 0] += drone
    out[:, 1] += drone

    # Timeline:
    # 0-8s: election timeout -- nodes idle with thin quiet tones (follower state)
    # 8-10s: election -- candidate (node 2) sends vote requests (FM bursts)
    # 10-30s: stable leader -- node 2 heartbeats, others respond
    # 30-38s: network partition -- nodes 0,1 vs 2,3,4
    # 38-45s: split brain -- node 0 becomes leader of minority, dissonance
    # 45-52s: partition heals -- reconvergence
    # 52-60s: unified cluster, single leader, peaceful coda

    # Section A: Election timeout (0-8s) -- thin, uncertain tones
    for i in range(5):
        s, e = 0, int(8 * SR)
        seg_t = t[s:e]
        # Each node: quiet thin sine with slight vibrato (uncertainty)
        vib = 1.5 * np.sin(2 * np.pi * (0.3 + i * 0.1) * seg_t)
        tone = 0.04 * sine(node_freqs[i] + vib, seg_t)
        tone *= envelope(e - s, attack=1.0, release=0.5)
        pan_l = 0.5 * (1 - node_pans[i])
        pan_r = 0.5 * (1 + node_pans[i])
        out[s:e, 0] += tone * pan_l
        out[s:e, 1] += tone * pan_r

    # Random election timeout clicks (nodes timing out)
    for _ in range(12):
        pos = int(np.random.uniform(2, 7.5) * SR)
        node_idx = np.random.randint(5)
        c = click(300, node_freqs[node_idx] * 2)
        cl = len(c)
        if pos + cl < n:
            out[pos:pos+cl, 0] += 0.06 * c * 0.5 * (1 - node_pans[node_idx])
            out[pos:pos+cl, 1] += 0.06 * c * 0.5 * (1 + node_pans[node_idx])

    # Section B: Election (8-10s) -- node 2 campaigns
    leader = 2
    s_elec = int(8 * SR)
    e_elec = int(10 * SR)
    seg_t = t[s_elec:e_elec]
    # Leader candidate: FM burst rising in confidence
    for burst_t in np.linspace(0, 1.5, 4):
        bs = s_elec + int(burst_t * SR)
        bl = int(0.3 * SR)
        if bs + bl < n:
            bt = np.arange(bl) / SR
            tone = 0.15 * fm_tone(node_freqs[leader], 6, 3, bt)
            tone *= envelope(bl, attack=0.01, release=0.1)
            out[bs:bs+bl, 0] += tone * 0.5
            out[bs:bs+bl, 1] += tone * 0.5
    # Vote responses: other nodes send brief acknowledgment
    for i in [0, 1, 3, 4]:
        resp_t = int((8.8 + i * 0.15) * SR)
        rl = int(0.15 * SR)
        if resp_t + rl < n:
            rt = np.arange(rl) / SR
            tone = 0.08 * sine(node_freqs[i] * 1.5, rt)
            tone *= envelope(rl, attack=0.005, release=0.05)
            pan_l = 0.5 * (1 - node_pans[i])
            pan_r = 0.5 * (1 + node_pans[i])
            out[resp_t:resp_t+rl, 0] += tone * pan_l
            out[resp_t:resp_t+rl, 1] += tone * pan_r

    # Section C: Stable leadership (10-30s) -- heartbeat + log replication
    hb_interval = 0.5  # heartbeat every 0.5s
    for hb_time in np.arange(10, 30, hb_interval):
        hb_s = int(hb_time * SR)
        # Leader heartbeat: strong FM pulse
        hl = int(0.08 * SR)
        if hb_s + hl < n:
            ht = np.arange(hl) / SR
            pulse = 0.18 * fm_tone(node_freqs[leader], 8, 2, ht)
            pulse *= envelope(hl, attack=0.002, release=0.03)
            out[hb_s:hb_s+hl, 0] += pulse * 0.5
            out[hb_s:hb_s+hl, 1] += pulse * 0.5

        # Follower acknowledgments (staggered)
        for i in [0, 1, 3, 4]:
            delay = 0.05 + i * 0.03
            rs = int((hb_time + delay) * SR)
            rl = int(0.04 * SR)
            if rs + rl < n:
                rt = np.arange(rl) / SR
                # Followers: gentle response, volume grows with time (trust)
                trust = min(1.0, (hb_time - 10) / 10)
                vol = 0.03 + 0.05 * trust
                tone = vol * sine(node_freqs[i], rt) * envelope(rl, attack=0.002, release=0.02)
                pan_l = 0.5 * (1 - node_pans[i])
                pan_r = 0.5 * (1 + node_pans[i])
                out[rs:rs+rl, 0] += tone * pan_l
                out[rs:rs+rl, 1] += tone * pan_r

    # Log replication events: occasional richer exchanges
    for log_time in np.arange(12, 30, 2.5):
        ls = int(log_time * SR)
        # Leader sends log entry (arpeggio down)
        for j, note_off in enumerate([0, 0.08, 0.16]):
            ns = ls + int(note_off * SR)
            nl = int(0.12 * SR)
            if ns + nl < n:
                nt = np.arange(nl) / SR
                freq = node_freqs[leader] * (2.0 - j * 0.3)
                tone = 0.1 * sine(freq, nt) * envelope(nl, attack=0.005, release=0.05)
                out[ns:ns+nl, 0] += tone * 0.5
                out[ns:ns+nl, 1] += tone * 0.5

    # Section D: Network partition (30-38s)
    # Partition: nodes 0,1 (left) vs 2,3,4 (right)
    # Communication between groups stops. Each side loses the other.
    # Left group: nodes 0,1 grow anxious (vibrato increases, detuning)
    for pt in np.arange(30, 38, 0.3):
        ps = int(pt * SR)
        pl = int(0.25 * SR)
        if ps + pl < n:
            seg_t_local = np.arange(pl) / SR
            anxiety = (pt - 30) / 8  # 0→1
            for i in [0, 1]:
                vib_depth = 3 + 8 * anxiety
                vib = vib_depth * np.sin(2 * np.pi * (2 + 3 * anxiety) * seg_t_local)
                tone = (0.06 + 0.04 * anxiety) * sine(node_freqs[i] + vib, seg_t_local)
                tone *= envelope(pl, attack=0.01, release=0.05)
                pan_l = 0.5 * (1 - node_pans[i])
                pan_r = 0.5 * (1 + node_pans[i])
                out[ps:ps+pl, 0] += tone * pan_l
                out[ps:ps+pl, 1] += tone * pan_r

    # Right group: node 2 continues heartbeats but fewer responses
    for hb_time in np.arange(30, 38, 0.5):
        hb_s = int(hb_time * SR)
        hl = int(0.06 * SR)
        if hb_s + hl < n:
            ht = np.arange(hl) / SR
            pulse = 0.12 * fm_tone(node_freqs[2], 8, 2, ht)
            pulse *= envelope(hl, attack=0.002, release=0.02)
            out[hb_s:hb_s+hl, 0] += pulse * 0.35
            out[hb_s:hb_s+hl, 1] += pulse * 0.65

    # Section E: Split brain (38-45s) -- node 0 declares itself leader
    # Two heartbeats coexist: dissonance
    for sb_time in np.arange(38, 45, 0.5):
        # Node 2 heartbeat (right)
        s1 = int(sb_time * SR)
        hl = int(0.06 * SR)
        if s1 + hl < n:
            ht = np.arange(hl) / SR
            out[s1:s1+hl, 1] += 0.12 * fm_tone(node_freqs[2], 8, 2, ht) * envelope(hl, 0.002, 0.02)
        # Node 0 heartbeat (left) -- slightly detuned, creates beating
        s2 = int((sb_time + 0.05) * SR)  # slightly offset = tension
        if s2 + hl < n:
            ht = np.arange(hl) / SR
            # Tritone relationship with real leader = maximum dissonance
            false_freq = node_freqs[0] * np.sqrt(2)  # tritone
            out[s2:s2+hl, 0] += 0.12 * fm_tone(false_freq, 10, 4, ht) * envelope(hl, 0.002, 0.02)

    # Continuous dissonant cluster during split brain
    sb_s = int(38 * SR)
    sb_e = int(45 * SR)
    sb_t = t[sb_s:sb_e]
    # Beating between node 0 tritone and node 2
    cluster = 0.04 * sine(node_freqs[0] * np.sqrt(2), sb_t) + 0.04 * sine(node_freqs[2], sb_t)
    cluster *= envelope(sb_e - sb_s, attack=1.0, release=1.0)
    out[sb_s:sb_e, 0] += cluster * 0.7
    out[sb_s:sb_e, 1] += cluster * 0.3

    # Section F: Partition heals (45-52s) -- reconvergence
    heal_s = int(45 * SR)
    heal_e = int(52 * SR)
    heal_t = t[heal_s:heal_e]
    # False leader fades, real leader strengthens
    for ht_time in np.arange(45, 52, 0.5):
        progress = (ht_time - 45) / 7  # 0→1
        hs = int(ht_time * SR)
        hl = int(0.08 * SR)
        if hs + hl < n:
            ht_arr = np.arange(hl) / SR
            # Real leader grows
            vol = 0.10 + 0.10 * progress
            pulse = vol * fm_tone(node_freqs[2], 8, 2 - progress, ht_arr)
            pulse *= envelope(hl, 0.002, 0.03)
            out[hs:hs+hl, 0] += pulse * 0.5
            out[hs:hs+hl, 1] += pulse * 0.5

        # All followers gradually rejoin
        for i in [0, 1, 3, 4]:
            delay = 0.06 + i * 0.02
            rs = int((ht_time + delay) * SR)
            rl = int(0.04 * SR)
            if rs + rl < n:
                rt = np.arange(rl) / SR
                vol = 0.03 + 0.06 * progress
                tone = vol * sine(node_freqs[i], rt) * envelope(rl, 0.002, 0.02)
                pan_l = 0.5 * (1 - node_pans[i])
                pan_r = 0.5 * (1 + node_pans[i])
                out[rs:rs+rl, 0] += tone * pan_l
                out[rs:rs+rl, 1] += tone * pan_r

    # Section G: Coda (52-60s) -- peaceful unified cluster
    coda_s = int(52 * SR)
    coda_e = int(60 * SR)
    coda_t = t[coda_s:coda_e]
    coda_env = envelope(coda_e - coda_s, attack=1.0, release=3.0)
    # All 5 nodes form a rich chord
    for i in range(5):
        harm = 0.08 * sine(node_freqs[i], coda_t)
        # Add 2nd harmonic for warmth
        harm += 0.03 * sine(node_freqs[i] * 2, coda_t)
        harm *= coda_env
        pan_l = 0.5 * (1 - node_pans[i])
        pan_r = 0.5 * (1 + node_pans[i])
        out[coda_s:coda_e, 0] += harm * pan_l
        out[coda_s:coda_e, 1] += harm * pan_r

    # Final gentle heartbeat
    for hb in np.arange(53, 58, 0.8):
        hs = int(hb * SR)
        hl = int(0.06 * SR)
        if hs + hl < n:
            ht_arr = np.arange(hl) / SR
            fade = max(0, 1 - (hb - 53) / 5)
            pulse = 0.08 * fade * fm_tone(node_freqs[2], 6, 1, ht_arr)
            pulse *= envelope(hl, 0.002, 0.03)
            out[hs:hs+hl, 0] += pulse * 0.5
            out[hs:hs+hl, 1] += pulse * 0.5

    # Normalize
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out * 0.85 / peak
    return out

# ── Piece 2: Gossip Protocol ─────────────────────────────────────────

def gossip_protocol():
    """7 nodes, epidemic rumor spreading. S-curve adoption as crescendo."""
    np.random.seed(42)
    dur = 55.0
    n = int(dur * SR)
    out = np.zeros((n, 2))
    t = np.arange(n) / SR

    # 7 nodes in a circle, frequencies: D minor pentatonic spread
    num_nodes = 7
    node_freqs = np.array([146.83, 164.81, 174.61, 196.00, 220.00, 246.94, 261.63])
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    node_pans = np.sin(angles)  # -1 to 1 stereo

    # Adjacency: each node connected to 2-3 neighbors
    adj = {i: set() for i in range(num_nodes)}
    for i in range(num_nodes):
        adj[i].add((i + 1) % num_nodes)
        adj[i].add((i - 1) % num_nodes)
        if np.random.random() < 0.4:
            adj[i].add((i + 2) % num_nodes)
    # Make symmetric
    for i in range(num_nodes):
        for j in list(adj[i]):
            adj[j].add(i)

    # Gossip simulation: tick every 0.6s, 80 ticks over ~48s
    tick_dur = 0.6
    num_ticks = 80
    infected = {0}  # node 0 starts with the rumor
    history = [set(infected)]

    for tick in range(1, num_ticks):
        new_infected = set(infected)
        for node in list(infected):
            # Each infected node contacts a random neighbor
            neighbors = list(adj[node])
            target = neighbors[np.random.randint(len(neighbors))]
            if target not in infected:
                new_infected.add(target)
        infected = new_infected
        history.append(set(infected))

    # 55Hz drone
    drone = 0.07 * sine(55, t) * envelope(n, attack=2.0, release=3.0)
    out[:, 0] += drone
    out[:, 1] += drone

    # Render each tick
    for tick in range(num_ticks):
        tick_start = int((3 + tick * tick_dur) * SR)  # start at 3s
        tick_len = int(tick_dur * SR * 0.8)

        if tick_start + tick_len >= n:
            break

        current_infected = history[tick]
        prev_infected = history[tick - 1] if tick > 0 else set()
        newly_infected = current_infected - prev_infected

        seg_t = np.arange(tick_len) / SR
        seg_env = envelope(tick_len, attack=0.01, release=0.15)

        for node in range(num_nodes):
            pan_l = 0.5 * (1 - node_pans[node])
            pan_r = 0.5 * (1 + node_pans[node])

            if node in newly_infected:
                # Newly infected: bright FM burst (the moment of learning)
                tone = 0.18 * fm_tone(node_freqs[node], 5, 4, seg_t)
                tone += 0.08 * sine(node_freqs[node] * 2, seg_t)
                tone *= envelope(tick_len, attack=0.005, release=0.2)
                out[tick_start:tick_start+tick_len, 0] += tone * pan_l
                out[tick_start:tick_start+tick_len, 1] += tone * pan_r
                # Click marking infection event
                c = click(400, node_freqs[node] * 3)
                cl = len(c)
                if tick_start + cl < n:
                    out[tick_start:tick_start+cl, 0] += 0.1 * c * pan_l
                    out[tick_start:tick_start+cl, 1] += 0.1 * c * pan_r

            elif node in current_infected:
                # Already infected: sustained rich tone (spreading knowledge)
                age = sum(1 for h in history[:tick] if node in h)
                harmonics = min(4, 1 + age // 5)
                tone = np.zeros(tick_len)
                for h in range(1, harmonics + 1):
                    tone += (0.06 / h) * sine(node_freqs[node] * h, seg_t)
                tone *= seg_env
                out[tick_start:tick_start+tick_len, 0] += tone * pan_l
                out[tick_start:tick_start+tick_len, 1] += tone * pan_r

            else:
                # Naive: thin quiet sine (ignorant)
                tone = 0.02 * sine(node_freqs[node], seg_t) * seg_env
                out[tick_start:tick_start+tick_len, 0] += tone * pan_l
                out[tick_start:tick_start+tick_len, 1] += tone * pan_r

    # Communication links: when infected node contacts neighbor, brief connector tone
    for tick in range(1, min(num_ticks, 70)):
        tick_start = int((3 + tick * tick_dur) * SR)
        for node in history[tick - 1]:
            if tick_start + int(0.1 * SR) >= n:
                break
            neighbors = list(adj[node])
            target = neighbors[np.random.randint(len(neighbors))]
            # Brief glissando between the two frequencies
            gl_len = int(0.08 * SR)
            if tick_start + gl_len < n:
                gl_t = np.arange(gl_len) / SR
                f1, f2 = node_freqs[node], node_freqs[target]
                gliss_freq = f1 + (f2 - f1) * gl_t / (gl_len / SR)
                gliss = 0.03 * sine(gliss_freq, gl_t) * envelope(gl_len, 0.005, 0.03)
                mid_pan = (node_pans[node] + node_pans[target]) / 2
                out[tick_start:tick_start+gl_len, 0] += gliss * 0.5 * (1 - mid_pan)
                out[tick_start:tick_start+gl_len, 1] += gliss * 0.5 * (1 + mid_pan)

    # Coda (last 5s): all infected, rich chord
    coda_s = int(50 * SR)
    coda_e = int(55 * SR)
    if coda_e <= n:
        coda_t = t[coda_s:coda_e]
        coda_env = envelope(coda_e - coda_s, attack=0.5, release=2.5)
        for node in range(num_nodes):
            for h in range(1, 5):
                harm = (0.06 / h) * sine(node_freqs[node] * h, coda_t)
                harm *= coda_env
                pan_l = 0.5 * (1 - node_pans[node])
                pan_r = 0.5 * (1 + node_pans[node])
                out[coda_s:coda_e, 0] += harm * pan_l
                out[coda_s:coda_e, 1] += harm * pan_r

    peak = np.max(np.abs(out))
    if peak > 0:
        out = out * 0.85 / peak
    return out

# ── Piece 3: Vector Clocks ───────────────────────────────────────────

def vector_clocks():
    """4 processes with independent clocks, message-passing synchronization."""
    np.random.seed(99)
    dur = 55.0
    n = int(dur * SR)
    out = np.zeros((n, 2))
    t = np.arange(n) / SR

    num_procs = 4
    # Each process has a base frequency and rate
    proc_freqs = np.array([130.81, 174.61, 220.00, 293.66])  # C3, F3, A3, D4
    proc_pans = np.array([-0.6, -0.2, 0.2, 0.6])
    # Different clock rates = polyrhythm
    proc_rates = np.array([1.0, 1.333, 1.667, 2.0])  # events per second

    # 55Hz drone
    drone = 0.07 * sine(55, t) * envelope(n, attack=2.0, release=3.0)
    out[:, 0] += drone
    out[:, 1] += drone

    # Simulate vector clocks
    clocks = np.zeros((num_procs, num_procs), dtype=int)  # vc[i][j] = process i's knowledge of j
    events = []  # (time_s, proc, event_type, vc_snapshot)

    # Generate local events and messages
    sim_time = 0
    next_event = np.array([0.0] * num_procs)
    messages_in_flight = []  # (arrive_time, sender, receiver, sender_vc)

    while sim_time < 50:
        # Find next event
        min_local = np.min(next_event)
        min_msg = min((m[0] for m in messages_in_flight), default=999)
        sim_time = min(min_local, min_msg)

        if sim_time >= 50:
            break

        if min_local <= min_msg:
            # Local event on the process with earliest next_event
            proc = np.argmin(next_event)
            clocks[proc][proc] += 1
            events.append((sim_time, proc, 'local', clocks[proc].copy()))

            # Maybe send a message to another process (30% chance)
            if np.random.random() < 0.3:
                target = np.random.choice([p for p in range(num_procs) if p != proc])
                delay = np.random.uniform(0.3, 1.5)
                messages_in_flight.append((sim_time + delay, proc, target, clocks[proc].copy()))
                events.append((sim_time, proc, 'send', clocks[proc].copy()))

            # Schedule next local event
            next_event[proc] = sim_time + np.random.exponential(1.0 / proc_rates[proc])

        else:
            # Message arrives
            msg = min(messages_in_flight, key=lambda m: m[0])
            messages_in_flight.remove(msg)
            arrive_time, sender, receiver, sender_vc = msg
            sim_time = arrive_time

            # Update receiver's vector clock: max(own, received) + increment own
            clocks[receiver] = np.maximum(clocks[receiver], sender_vc)
            clocks[receiver][receiver] += 1
            events.append((sim_time, receiver, 'receive', clocks[receiver].copy()))

    # Render events as sound
    for ev_time, proc, ev_type, vc in events:
        ev_s = int((2.5 + ev_time) * SR)  # offset 2.5s for intro silence

        pan_l = 0.5 * (1 - proc_pans[proc])
        pan_r = 0.5 * (1 + proc_pans[proc])

        # Clock "tick" frequency: base freq * (1 + vc[proc] * 0.02)
        # As clock advances, pitch subtly rises
        clock_val = vc[proc]
        freq_shift = 1.0 + clock_val * 0.015
        freq = proc_freqs[proc] * freq_shift

        if ev_type == 'local':
            # Simple tone: pure sine + 1 harmonic
            el = int(0.15 * SR)
            if ev_s + el < n:
                et = np.arange(el) / SR
                tone = 0.10 * sine(freq, et) + 0.04 * sine(freq * 2, et)
                tone *= envelope(el, attack=0.005, release=0.06)
                out[ev_s:ev_s+el, 0] += tone * pan_l
                out[ev_s:ev_s+el, 1] += tone * pan_r

        elif ev_type == 'send':
            # Outgoing message: FM burst with upward sweep
            el = int(0.2 * SR)
            if ev_s + el < n:
                et = np.arange(el) / SR
                sweep = freq * (1 + 0.5 * et / (el / SR))
                tone = 0.12 * fm_tone(sweep, 4, 3, et)
                tone *= envelope(el, attack=0.005, release=0.08)
                out[ev_s:ev_s+el, 0] += tone * pan_l
                out[ev_s:ev_s+el, 1] += tone * pan_r

        elif ev_type == 'receive':
            # Incoming message: richer tone (sync moment), brief convergence
            el = int(0.25 * SR)
            if ev_s + el < n:
                et = np.arange(el) / SR
                # Rich: 3 harmonics + slight FM warmth
                tone = 0.08 * sine(freq, et)
                tone += 0.05 * sine(freq * 1.5, et)  # perfect fifth
                tone += 0.03 * sine(freq * 2, et)
                tone += 0.03 * fm_tone(freq, 3, 1, et)
                tone *= envelope(el, attack=0.01, release=0.1)
                out[ev_s:ev_s+el, 0] += tone * pan_l
                out[ev_s:ev_s+el, 1] += tone * pan_r
                # Click marking sync
                c = click(200, 2000)
                cl = len(c)
                if ev_s + cl < n:
                    out[ev_s:ev_s+cl, 0] += 0.05 * c * 0.5
                    out[ev_s:ev_s+cl, 1] += 0.05 * c * 0.5

    # Background process pulses: continuous heartbeat for each process
    for proc in range(num_procs):
        pulse_interval = 1.0 / proc_rates[proc]
        for pt in np.arange(2.5, 52.5, pulse_interval):
            ps = int(pt * SR)
            pl = int(0.03 * SR)
            if ps + pl < n:
                pt_arr = np.arange(pl) / SR
                pulse = 0.02 * sine(proc_freqs[proc] * 0.5, pt_arr)
                pulse *= envelope(pl, attack=0.002, release=0.015)
                pan_l = 0.5 * (1 - proc_pans[proc])
                pan_r = 0.5 * (1 + proc_pans[proc])
                out[ps:ps+pl, 0] += pulse * pan_l
                out[ps:ps+pl, 1] += pulse * pan_r

    # Coda (50-55s): all clocks aligned, harmonious chord
    coda_s = int(50 * SR)
    coda_e = int(55 * SR)
    if coda_e <= n:
        coda_t = t[coda_s:coda_e]
        coda_env = envelope(coda_e - coda_s, attack=1.0, release=2.5)
        for proc in range(num_procs):
            for h in range(1, 4):
                harm = (0.07 / h) * sine(proc_freqs[proc] * h, coda_t)
                harm *= coda_env
                pan_l = 0.5 * (1 - proc_pans[proc])
                pan_r = 0.5 * (1 + proc_pans[proc])
                out[coda_s:coda_e, 0] += harm * pan_l
                out[coda_s:coda_e, 1] += harm * pan_r

    peak = np.max(np.abs(out))
    if peak > 0:
        out = out * 0.85 / peak
    return out


# ── Main ──────────────────────────────────────────────────────────────

def main():
    os.makedirs('output', exist_ok=True)
    print("Phase 19: Distributed Systems")
    print("=" * 50)

    print("\n1. Raft Consensus (60s, stereo)")
    data = raft_consensus()
    write_wav('output/dist_1_raft_consensus.wav', data)

    print("\n2. Gossip Protocol (55s, stereo)")
    data = gossip_protocol()
    write_wav('output/dist_2_gossip_protocol.wav', data)

    print("\n3. Vector Clocks (55s, stereo)")
    data = vector_clocks()
    write_wav('output/dist_3_vector_clocks.wav', data)

    print("\nDone!")


if __name__ == '__main__':
    main()
