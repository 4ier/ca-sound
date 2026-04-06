#!/usr/bin/env python3
"""Phase 26: Network Protocols -- handshake, resolution, broadcast.

Three pieces:
1. TCP Handshake (55s, stereo) -- Client sends a rising SYN sweep on the left,
   server answers with descending SYN-ACK on the right plus a click, then ACK
   resolves into a perfect fifth. 20 data packets alternate left/right in a D
   pentatonic arpeggio with quieter ACK echoes. Three packets are lost, leaving
   silence gaps, then retransmit louder with a tritone warning click. FIN closes
   in a four-step descent to a D1 drone while a 55Hz link drone persists.

2. DNS Resolution (50s, stereo) -- Queries launch as bright 880Hz FM from the
   client on the left. The recursive resolver sits in the center and forwards
   through root, TLD, and authoritative layers (110/220/440Hz). Cache hits
   return instantly and fade according to TTL; misses traverse the full stack,
   then descend back with stacked chords. One mixed path yields NXDOMAIN as a
   tritone burst and silence. Final resolved answers bloom into an A major chord.

3. ARP Broadcast (55s, stereo) -- A wide 660Hz FM burst floods the LAN while
   12 hosts answer on a chromatic octave from 220-440Hz, each with its own
   stereo position and harmonic fingerprint. Normal discovery builds the ARP
   table drone, the middle section erupts into a dense storm, then an attacker
   poisons replies with a tritone detune before legitimate hosts reassert and
   the attacker fades away.
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


def click_sound(n=200, freq=2000):
    t_arr = np.arange(n) / SR
    return np.sin(2 * np.pi * freq * t_arr) * np.exp(-t_arr * 40)


PENTA_D = [146.83, 164.81, 196.0, 220.0, 261.63, 293.66, 329.63, 392.0, 440.0, 523.25]
A_MAJOR = [110.0, 138.59, 164.81, 220.0, 277.18, 329.63, 440.0]
HOST_FREQS = np.array([220.0 * (2 ** (i / 12)) for i in range(12)])


# ── Piece 1: TCP Handshake ───────────────────────────────────────────────────

def tcp_handshake():
    """TCP connection lifecycle: handshake, transfer, loss, retransmit, FIN."""
    print("Generating TCP Handshake...")
    duration = 55.0
    N = int(duration * SR)
    t = np.arange(N) / SR
    out_L = np.zeros(N)
    out_R = np.zeros(N)

    drone = sine(55, t) * 0.06 * envelope(N, attack=2.0, release=2.0)
    out_L += drone
    out_R += drone

    syn_start = int(0.8 * SR)
    syn_dur = int(4.8 * SR)
    syn_t = np.arange(syn_dur) / SR
    syn_freqs = np.linspace(220.0, 440.0, syn_dur)
    syn_phase = np.cumsum(2 * np.pi * syn_freqs / SR)
    syn = np.sin(syn_phase + 2.3 * np.sin(2 * np.pi * (4.0 + 1.5 * syn_t) * syn_t))
    syn *= 0.15 * envelope(syn_dur, attack=0.08, release=0.3)
    out_L[syn_start:syn_start+syn_dur] += syn * 0.92
    out_R[syn_start:syn_start+syn_dur] += syn * 0.08

    synack_start = int(5.9 * SR)
    synack_dur = int(4.1 * SR)
    synack_t = np.arange(synack_dur) / SR
    synack_freqs = np.linspace(440.0, 330.0, synack_dur)
    synack_phase = np.cumsum(2 * np.pi * synack_freqs / SR)
    synack = np.sin(synack_phase + 2.7 * np.sin(2 * np.pi * 5.5 * synack_t))
    synack += 0.35 * sine(330.0, synack_t)
    synack *= 0.13 * envelope(synack_dur, attack=0.05, release=0.25)
    out_L[synack_start:synack_start+synack_dur] += synack * 0.12
    out_R[synack_start:synack_start+synack_dur] += synack * 0.88
    c = click_sound(240, 1650) * 0.14
    out_R[synack_start:synack_start+240] += c
    out_L[synack_start:synack_start+240] += c * 0.15

    ack_start = int(10.8 * SR)
    ack_dur = int(2.2 * SR)
    ack_t = np.arange(ack_dur) / SR
    ack = sine(220.0, ack_t) * 0.10 + sine(330.0, ack_t) * 0.08 + sine(440.0, ack_t) * 0.03
    ack *= envelope(ack_dur, attack=0.08, release=0.5)
    out_L[ack_start:ack_start+ack_dur] += ack * 0.55
    out_R[ack_start:ack_start+ack_dur] += ack * 0.55

    packet_times = np.linspace(13.5, 41.5, 20)
    lost_packets = {4, 10, 16}
    for i, start_time in enumerate(packet_times):
        start = int(start_time * SR)
        dur = int(0.56 * SR)
        if start + dur >= N:
            continue
        base_idx = (i * 2) % (len(PENTA_D) - 2)
        arp_notes = [PENTA_D[base_idx], PENTA_D[base_idx + 1], PENTA_D[base_idx + 2]]
        pan = 0.18 if i % 2 == 0 else 0.82

        if i in lost_packets:
            retrans_start = start + int(0.42 * SR)
            if retrans_start + dur >= N:
                continue
            packet = np.zeros(dur)
            note_dur = dur // 3
            for j, freq in enumerate(arp_notes):
                ns = j * note_dur
                ne = dur if j == 2 else min(dur, (j + 1) * note_dur)
                tone_t = np.arange(ne - ns) / SR
                tone = fm_tone(freq * 1.5, freq * 0.75, 1.6, tone_t) * 0.22
                tone += sine(freq, tone_t) * 0.10
                packet[ns:ne] += tone
            packet *= envelope(dur, attack=0.02, release=0.08)
            out_L[retrans_start:retrans_start+dur] += packet * (1 - pan) * 1.15
            out_R[retrans_start:retrans_start+dur] += packet * pan * 1.15

            tc = click_sound(260, int(900 * np.sqrt(2))) * 0.16
            out_L[retrans_start:retrans_start+260] += tc * 0.45
            out_R[retrans_start:retrans_start+260] += tc * 0.55

            echo_start = retrans_start + int(0.22 * SR)
            echo_dur = int(0.22 * SR)
            if echo_start + echo_dur < N:
                echo_t = np.arange(echo_dur) / SR
                echo = sine(arp_notes[-1], echo_t) * 0.06 + sine(arp_notes[-1] * 1.5, echo_t) * 0.024
                echo *= envelope(echo_dur, attack=0.01, release=0.08)
                out_L[echo_start:echo_start+echo_dur] += echo * 0.25
                out_R[echo_start:echo_start+echo_dur] += echo * 0.25
            continue

        packet = np.zeros(dur)
        note_dur = dur // 3
        for j, freq in enumerate(arp_notes):
            ns = j * note_dur
            ne = dur if j == 2 else min(dur, (j + 1) * note_dur)
            tone_t = np.arange(ne - ns) / SR
            tone = sine(freq, tone_t) * 0.12 + sine(freq * 2, tone_t) * 0.03
            packet[ns:ne] += tone
        packet *= envelope(dur, attack=0.01, release=0.07)
        out_L[start:start+dur] += packet * (1 - pan)
        out_R[start:start+dur] += packet * pan

        echo_start = start + int(0.24 * SR)
        echo_dur = int(0.18 * SR)
        if echo_start + echo_dur < N:
            echo_t = np.arange(echo_dur) / SR
            echo = sine(arp_notes[-1], echo_t) * 0.04 + sine(arp_notes[-1] * 1.5, echo_t) * 0.016
            echo *= envelope(echo_dur, attack=0.008, release=0.07)
            out_L[echo_start:echo_start+echo_dur] += echo * 0.33
            out_R[echo_start:echo_start+echo_dur] += echo * 0.33

    fin_times = [45.0, 46.3, 47.6, 48.9]
    fin_freqs = [293.66, 220.0, 146.83, 73.42]
    for i, (fin_time, freq) in enumerate(zip(fin_times, fin_freqs)):
        start = int(fin_time * SR)
        dur = int(0.85 * SR)
        if start + dur >= N:
            continue
        t_local = np.arange(dur) / SR
        tone = fm_tone(freq, freq * 0.5, 0.9, t_local) * 0.09
        tone += sine(freq * 0.5, t_local) * 0.05
        tone *= envelope(dur, attack=0.04, release=0.3)
        pan = 0.25 if i % 2 == 0 else 0.75
        out_L[start:start+dur] += tone * (1 - pan)
        out_R[start:start+dur] += tone * pan
        fin_click = click_sound(180, int(freq * 6)) * 0.05
        out_L[start:start+180] += fin_click * (1 - pan)
        out_R[start:start+180] += fin_click * pan

    d1_start = int(49.5 * SR)
    d1_t = np.arange(N - d1_start) / SR
    d1 = sine(36.71, d1_t) * 0.12 + sine(73.42, d1_t) * 0.05
    d1 *= envelope(len(d1_t), attack=1.2, release=2.5)
    out_L[d1_start:] += d1
    out_R[d1_start:] += d1

    peak = max(np.max(np.abs(out_L)), np.max(np.abs(out_R)), 1e-6)
    if peak > 0.95:
        out_L *= 0.9 / peak
        out_R *= 0.9 / peak

    return np.stack([out_L, out_R], axis=1)


# ── Piece 2: DNS Resolution ──────────────────────────────────────────────────

def dns_resolution():
    """Recursive lookup paths, cache behavior, TTL decay, and final resolution."""
    print("Generating DNS Resolution...")
    duration = 50.0
    N = int(duration * SR)
    t = np.arange(N) / SR
    out_L = np.zeros(N)
    out_R = np.zeros(N)

    bed = (sine(55, t) * 0.016 + sine(110, t) * 0.024) * envelope(N, attack=1.5, release=2.0)
    out_L += bed
    out_R += bed

    query_times = [2.0, 7.2, 12.4, 17.8, 23.4, 29.0, 34.8, 40.6]
    query_types = ["miss", "hit", "miss", "mixed", "hit", "miss", "mixed_nxdomain", "hit"]
    ttl_values = [0.9, 0.75, 0.6, 0.45, 0.35, 0.85, 0.5, 0.25]

    for i, (query_time, qtype, ttl) in enumerate(zip(query_times, query_types, ttl_values)):
        start = int(query_time * SR)

        qdur = int(0.24 * SR)
        q_t = np.arange(qdur) / SR
        query = fm_tone(880, 55, 2.6, q_t) * 0.13
        query += sine(1760, q_t) * 0.025
        query *= envelope(qdur, attack=0.008, release=0.08)
        out_L[start:start+qdur] += query * 0.92
        out_R[start:start+qdur] += query * 0.08

        if qtype == "hit":
            rdur = int((0.28 + ttl * 0.5) * SR)
            rstart = start + int(0.16 * SR)
            if rstart + rdur >= N:
                continue
            r_t = np.arange(rdur) / SR
            response = sine(A_MAJOR[(i + 3) % len(A_MAJOR)], r_t) * 0.09
            response += fm_tone(880, 220, 1.2, r_t) * 0.04
            response += sine(440, r_t) * 0.03
            response *= np.exp(-r_t * (4.5 - 3.0 * ttl))
            response *= envelope(rdur, attack=0.005, release=0.12 + ttl * 0.28)
            out_L[rstart:rstart+rdur] += response * 0.42
            out_R[rstart:rstart+rdur] += response * 0.58
            continue

        if qtype == "miss":
            hop_freqs = [110.0, 220.0, 440.0]
            for hop_i, hop_freq in enumerate(hop_freqs):
                hstart = start + int((0.20 + hop_i * 0.24) * SR)
                hdur = int((0.34 + 0.08 * hop_i) * SR)
                if hstart + hdur >= N:
                    continue
                h_t = np.arange(hdur) / SR
                hop = fm_tone(hop_freq * (1 + i * 0.01), hop_freq * 2, 1.5 - hop_i * 0.2, h_t) * 0.07
                hop += sine(hop_freq, h_t) * (0.04 + hop_i * 0.01)
                hop *= envelope(hdur, attack=0.01, release=0.12)
                out_L[hstart:hstart+hdur] += hop * 0.5
                out_R[hstart:hstart+hdur] += hop * 0.5

            descent = [440.0, 220.0, 110.0, A_MAJOR[(i + 1) % len(A_MAJOR)]]
            for j, freq in enumerate(descent):
                dstart = start + int((1.02 + j * 0.16) * SR)
                ddur = int(0.26 * SR)
                if dstart + ddur >= N:
                    continue
                d_t = np.arange(ddur) / SR
                tone = sine(freq, d_t) * 0.08 + sine(freq * 1.5, d_t) * 0.028
                tone *= envelope(ddur, attack=0.01, release=0.08)
                out_L[dstart:dstart+ddur] += tone * 0.38
                out_R[dstart:dstart+ddur] += tone * 0.62
            continue

        if qtype == "mixed":
            cache_dur = int((0.22 + ttl * 0.35) * SR)
            cache_start = start + int(0.14 * SR)
            if cache_start + cache_dur < N:
                c_t = np.arange(cache_dur) / SR
                cache = sine(110.0, c_t) * 0.06 + sine(220.0, c_t) * 0.025
                cache *= np.exp(-c_t * (4.0 - 2.2 * ttl))
                cache *= envelope(cache_dur, attack=0.005, release=0.10 + ttl * 0.18)
                out_L[cache_start:cache_start+cache_dur] += cache * 0.48
                out_R[cache_start:cache_start+cache_dur] += cache * 0.52

            for hop_i, hop_freq in enumerate([220.0, 440.0]):
                hstart = start + int((0.42 + hop_i * 0.22) * SR)
                hdur = int(0.34 * SR)
                if hstart + hdur >= N:
                    continue
                h_t = np.arange(hdur) / SR
                hop = fm_tone(hop_freq, hop_freq * 1.8, 1.1, h_t) * 0.07
                hop += sine(hop_freq, h_t) * 0.04
                hop *= envelope(hdur, attack=0.01, release=0.10)
                out_L[hstart:hstart+hdur] += hop * 0.5
                out_R[hstart:hstart+hdur] += hop * 0.5

            answer_start = start + int(0.96 * SR)
            answer_dur = int(0.55 * SR)
            if answer_start + answer_dur < N:
                a_t = np.arange(answer_dur) / SR
                answer = sine(A_MAJOR[(i + 2) % len(A_MAJOR)], a_t) * 0.09
                answer += sine(220.0, a_t) * 0.04 + sine(440.0, a_t) * 0.03
                answer *= envelope(answer_dur, attack=0.02, release=0.18)
                out_L[answer_start:answer_start+answer_dur] += answer * 0.4
                out_R[answer_start:answer_start+answer_dur] += answer * 0.6
            continue

        if qtype == "mixed_nxdomain":
            cache_start = start + int(0.15 * SR)
            cache_dur = int(0.24 * SR)
            if cache_start + cache_dur < N:
                c_t = np.arange(cache_dur) / SR
                cache = sine(110.0, c_t) * 0.05 + sine(220.0, c_t) * 0.03
                cache *= envelope(cache_dur, attack=0.008, release=0.08)
                out_L[cache_start:cache_start+cache_dur] += cache * 0.5
                out_R[cache_start:cache_start+cache_dur] += cache * 0.5

            hop_start = start + int(0.42 * SR)
            hop_dur = int(0.34 * SR)
            if hop_start + hop_dur < N:
                h_t = np.arange(hop_dur) / SR
                hop = fm_tone(440.0, 660.0, 1.8, h_t) * 0.08 + sine(440.0, h_t) * 0.03
                hop *= envelope(hop_dur, attack=0.01, release=0.09)
                out_L[hop_start:hop_start+hop_dur] += hop * 0.5
                out_R[hop_start:hop_start+hop_dur] += hop * 0.5

            nx_start = start + int(0.86 * SR)
            nx_dur = int(0.26 * SR)
            if nx_start + nx_dur < N:
                nx_t = np.arange(nx_dur) / SR
                burst = fm_tone(311.13, 440.0, 3.2, nx_t) * 0.16
                click_len = min(nx_dur, 260)
                burst[:click_len] += click_sound(click_len, 1860) * 0.09
                burst *= np.exp(-nx_t * 8)
                out_L[nx_start:nx_start+nx_dur] += burst * 0.5
                out_R[nx_start:nx_start+nx_dur] += burst * 0.5

    final_start = int(44.0 * SR)
    final_dur = N - final_start
    final_t = np.arange(final_dur) / SR
    final = (
        sine(110.0, final_t) * 0.06
        + sine(138.59, final_t) * 0.05
        + sine(164.81, final_t) * 0.05
        + sine(220.0, final_t) * 0.04
        + sine(277.18, final_t) * 0.035
        + sine(329.63, final_t) * 0.03
    )
    final *= envelope(final_dur, attack=0.8, release=2.2)
    out_L[final_start:] += final * 0.55
    out_R[final_start:] += final * 0.55

    peak = max(np.max(np.abs(out_L)), np.max(np.abs(out_R)), 1e-6)
    if peak > 0.95:
        out_L *= 0.9 / peak
        out_R *= 0.9 / peak

    return np.stack([out_L, out_R], axis=1)


# ── Piece 3: ARP Broadcast ───────────────────────────────────────────────────

def arp_broadcast():
    """ARP discovery, storm, poisoning, and table recovery across a LAN."""
    print("Generating ARP Broadcast...")
    duration = 55.0
    N = int(duration * SR)
    t = np.arange(N) / SR
    out_L = np.zeros(N)
    out_R = np.zeros(N)
    rng = np.random.default_rng(126)

    host_pans = np.linspace(0.05, 0.95, len(HOST_FREQS))
    host_harmonics = [2 + (i % 5) for i in range(len(HOST_FREQS))]
    seen_hosts = set()

    def add_host_response(start_time, host_idx, amp=0.11, detune=1.0):
        start = int(start_time * SR)
        dur = int(0.30 * SR)
        if start + dur >= N:
            return
        t_local = np.arange(dur) / SR
        freq = HOST_FREQS[host_idx] * detune
        voice = np.zeros(dur)
        for h in range(1, host_harmonics[host_idx] + 1):
            if h % 2 == 0:
                voice += sine(freq * h, t_local) * (amp / (h * 1.5))
            else:
                voice += fm_tone(freq * h, freq * 0.45, 0.8 + 0.2 * h, t_local) * (amp / (h * 1.8))
        voice *= envelope(dur, attack=0.01, release=0.08)
        pan = host_pans[host_idx]
        out_L[start:start+dur] += voice * (1 - pan)
        out_R[start:start+dur] += voice * pan

    def add_broadcast(start_time, amp=0.12):
        start = int(start_time * SR)
        dur = int(0.16 * SR)
        if start + dur >= N:
            return
        t_local = np.arange(dur) / SR
        left = np.sin(2 * np.pi * 660.0 * t_local + 2.5 * np.sin(2 * np.pi * 110.0 * t_local))
        right = np.sin(2 * np.pi * 660.0 * t_local + np.pi / 3 + 2.5 * np.sin(2 * np.pi * 110.0 * t_local))
        left *= amp * np.exp(-t_local * 12)
        right *= amp * np.exp(-t_local * 12)
        out_L[start:start+dur] += left
        out_R[start:start+dur] += right

    # Phase 1 (0-20s): normal ARP discovery
    for burst_time in np.arange(0.8, 19.2, 2.8):
        add_broadcast(burst_time, amp=0.10)

    intro_times = np.linspace(1.4, 18.6, len(HOST_FREQS))
    for host_idx, intro_time in enumerate(intro_times):
        add_host_response(intro_time, host_idx, amp=0.11)
        if host_idx not in seen_hosts:
            seen_hosts.add(host_idx)
            tail_start = int((intro_time + 0.08) * SR)
            tail_len = N - tail_start
            if tail_len <= 0:
                continue
            tail_t = np.arange(tail_len) / SR
            tail = sine(HOST_FREQS[host_idx] * 0.5, tail_t) * 0.006
            tail += sine(HOST_FREQS[host_idx], tail_t) * 0.0025
            tail *= envelope(tail_len, attack=0.4, release=2.0)
            pan = host_pans[host_idx]
            out_L[tail_start:] += tail * (1 - pan)
            out_R[tail_start:] += tail * pan

    # Phase 2 (20-35s): ARP storm with dense overlap
    for burst_time in np.arange(20.0, 35.0, 0.42):
        add_broadcast(burst_time, amp=0.08)
        responders = rng.choice(len(HOST_FREQS), size=3, replace=False)
        for j, host_idx in enumerate(responders):
            add_host_response(burst_time + 0.04 + j * 0.05 + rng.uniform(0.0, 0.06), host_idx, amp=0.09)

    # Phase 3 (35-50s): ARP poisoning, attacker detuned by tritone
    attacker = 9
    legit_hosts = [0, 2, 4, 5, 7, 10, 11]
    for poison_time in np.arange(35.0, 50.0, 0.72):
        progress = (poison_time - 35.0) / 15.0
        add_broadcast(poison_time, amp=0.07)
        add_host_response(poison_time + 0.03, attacker, amp=0.16 * (1.0 - 0.25 * progress), detune=np.sqrt(2))
        tritone_click_start = int((poison_time + 0.03) * SR)
        tc = click_sound(220, 1320) * 0.08 * (1.0 - 0.35 * progress)
        out_L[tritone_click_start:tritone_click_start+220] += tc * 0.35
        out_R[tritone_click_start:tritone_click_start+220] += tc * 0.65
        responders = rng.choice(legit_hosts, size=2, replace=False)
        for j, host_idx in enumerate(responders):
            add_host_response(poison_time + 0.18 + j * 0.11, host_idx, amp=0.08 + 0.02 * progress)

    # Resolution (50-55s): legitimate hosts reassert, attacker fades
    for resolve_time in np.arange(50.0, 54.5, 0.55):
        responders = [0, 4, 7, 11]
        for j, host_idx in enumerate(responders):
            add_host_response(resolve_time + j * 0.06, host_idx, amp=0.09)

    fade_start = int(50.0 * SR)
    fade_len = N - fade_start
    fade_t = np.arange(fade_len) / SR
    attacker_fade = sine(HOST_FREQS[attacker] * np.sqrt(2) * 0.5, fade_t) * 0.02 * np.exp(-fade_t * 2.8)
    out_L[fade_start:] += attacker_fade * 0.3
    out_R[fade_start:] += attacker_fade * 0.7

    resolve_chord = (
        sine(HOST_FREQS[0] * 0.5, fade_t) * 0.025
        + sine(HOST_FREQS[4] * 0.5, fade_t) * 0.022
        + sine(HOST_FREQS[7] * 0.5, fade_t) * 0.022
        + sine(HOST_FREQS[11] * 0.5, fade_t) * 0.018
    )
    resolve_chord *= envelope(fade_len, attack=0.8, release=2.0)
    out_L[fade_start:] += resolve_chord * 0.55
    out_R[fade_start:] += resolve_chord * 0.55

    peak = max(np.max(np.abs(out_L)), np.max(np.abs(out_R)), 1e-6)
    if peak > 0.95:
        out_L *= 0.9 / peak
        out_R *= 0.9 / peak

    return np.stack([out_L, out_R], axis=1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("output", exist_ok=True)
    os.makedirs("docs/audio", exist_ok=True)

    pieces = [
        ("net_1_tcp_handshake", tcp_handshake),
        ("net_2_dns_resolution", dns_resolution),
        ("net_3_arp_broadcast", arp_broadcast),
    ]

    for name, fn in pieces:
        audio = fn()
        write_wav(f"output/{name}.wav", audio)

    print("\nPhase 26: Network Protocols — done.")


if __name__ == "__main__":
    main()
