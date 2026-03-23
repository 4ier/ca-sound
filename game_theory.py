#!/usr/bin/env python3
"""Phase 20: Game Theory -- cooperation, defection, evolution.

Three pieces:
1. Prisoner's Dilemma Tournament (60s, stereo) -- Iterated PD with classic strategies.
   Tit-for-Tat, Always-Cooperate, Always-Defect, Random, Grudger play round-robin.
   Cooperation = consonant intervals (thirds, fifths), defection = dissonant (tritone, minor second).
   Mutual cooperation = warm harmony. Mutual defection = harsh noise. Exploitation = asymmetric tension.
   Cumulative scores shape volume/richness over time.
2. Evolutionary Stable Strategy (55s, stereo) -- Population dynamics.
   Hawks and Doves in a population. Hawk-Hawk = costly fight (dissonance + volume).
   Hawk-Dove = hawk wins quietly. Dove-Dove = share peacefully (soft consonance).
   Population ratio drifts toward ESS equilibrium. Audio tracks the hawk/dove ratio
   as a timbral spectrum from aggressive to gentle.
3. Nash Equilibrium (55s, stereo) -- Two-player coordination game.
   Two players searching a 5x5 payoff matrix. Each player controls pitch (row/column).
   When both land on Nash equilibrium = perfect consonance. Near-miss = almost-consonant
   beating. Far from equilibrium = random dissonance. Best-response dynamics as melody.
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
    env = np.exp(-t_arr * 40)
    return env * np.sin(2 * np.pi * freq * t_arr)


print("helpers defined")


# ── Piece 1: Prisoner's Dilemma Tournament ─────────────────────────────

def prisoners_dilemma():
    """Iterated PD tournament: 5 strategies, round-robin, 200 rounds each matchup."""
    np.random.seed(20)
    dur = 60.0
    n = int(dur * SR)
    out = np.zeros((n, 2))
    t = np.arange(n) / SR

    # 55Hz drone
    drone = 0.07 * sine(55, t) * envelope(n, attack=2.0, release=3.0)
    out[:, 0] += drone
    out[:, 1] += drone

    # Strategies: TFT, AllC, AllD, Random, Grudger
    strat_names = ['TFT', 'AllC', 'AllD', 'Random', 'Grudger']
    strat_freqs = np.array([220.0, 261.63, 174.61, 196.0, 293.66])  # A3,C4,F3,G3,D4
    strat_pans = np.array([-0.6, -0.3, 0.0, 0.3, 0.6])

    # PD payoffs: (C,C)=3,3  (C,D)=0,5  (D,C)=5,0  (D,D)=1,1
    def play_strategy(name, history_mine, history_other, rng):
        if name == 'AllC':
            return 0  # cooperate
        elif name == 'AllD':
            return 1  # defect
        elif name == 'TFT':
            return history_other[-1] if len(history_other) > 0 else 0
        elif name == 'Random':
            return rng.randint(2)
        elif name == 'Grudger':
            return 1 if 1 in history_other else 0
        return 0

    # Generate all matchups
    matchups = []
    for i in range(5):
        for j in range(i+1, 5):
            matchups.append((i, j))

    rounds_per_match = 200
    total_matchups = len(matchups)  # 10
    time_per_matchup = dur / total_matchups  # 6s each

    rng = np.random.RandomState(42)
    scores = np.zeros(5)

    for m_idx, (si, sj) in enumerate(matchups):
        t_start = m_idx * time_per_matchup
        hist_i, hist_j = [], []

        for r in range(rounds_per_match):
            ai = play_strategy(strat_names[si], hist_i, hist_j, rng)
            aj = play_strategy(strat_names[sj], hist_i, hist_j, rng)
            hist_i.append(ai)
            hist_j.append(aj)

            # Payoffs
            if ai == 0 and aj == 0:   # C,C
                scores[si] += 3; scores[sj] += 3
            elif ai == 0 and aj == 1:  # C,D
                scores[si] += 0; scores[sj] += 5
            elif ai == 1 and aj == 0:  # D,C
                scores[si] += 5; scores[sj] += 0
            else:                       # D,D
                scores[si] += 1; scores[sj] += 1

            # Sound for each round
            round_time = t_start + (r / rounds_per_match) * time_per_matchup
            pos = int(round_time * SR)
            note_len = int(0.025 * SR)
            if pos + note_len >= n:
                continue

            nt = np.arange(note_len) / SR
            env = envelope(note_len, attack=0.002, release=0.01)

            # Cooperation = consonant (fifth), defection = dissonant (tritone)
            if ai == 0 and aj == 0:  # mutual cooperation: warm harmony
                tone_i = 0.08 * sine(strat_freqs[si], nt) * env
                tone_j = 0.08 * sine(strat_freqs[si] * 1.5, nt) * env  # perfect fifth
                vol = 0.12
            elif ai == 1 and aj == 1:  # mutual defection: harsh
                tone_i = 0.06 * fm_tone(strat_freqs[si], 7, 4, nt) * env
                tone_j = 0.06 * fm_tone(strat_freqs[si] * np.sqrt(2), 9, 5, nt) * env  # tritone
                vol = 0.10
            elif ai == 0 and aj == 1:  # i cooperates, j defects: tension
                tone_i = 0.04 * sine(strat_freqs[si], nt) * env  # quiet victim
                tone_j = 0.12 * fm_tone(strat_freqs[sj], 5, 3, nt) * env  # loud exploiter
                vol = 0.10
            else:  # i defects, j cooperates
                tone_i = 0.12 * fm_tone(strat_freqs[si], 5, 3, nt) * env
                tone_j = 0.04 * sine(strat_freqs[sj], nt) * env
                vol = 0.10

            pan_i_l = 0.5 * (1 - strat_pans[si])
            pan_i_r = 0.5 * (1 + strat_pans[si])
            pan_j_l = 0.5 * (1 - strat_pans[sj])
            pan_j_r = 0.5 * (1 + strat_pans[sj])

            out[pos:pos+note_len, 0] += tone_i * pan_i_l + tone_j * pan_j_l
            out[pos:pos+note_len, 1] += tone_i * pan_i_r + tone_j * pan_j_r

        # Matchup transition click
        click_pos = int((t_start + time_per_matchup - 0.1) * SR)
        if click_pos + 300 < n:
            c = click(300, 2000)
            out[click_pos:click_pos+300, 0] += 0.08 * c
            out[click_pos:click_pos+300, 1] += 0.08 * c

    # Coda: final score chord (last 3s) — winner loudest
    coda_s = int(57 * SR)
    coda_len = n - coda_s
    coda_t = np.arange(coda_len) / SR
    score_norm = scores / scores.max()
    for i in range(5):
        vol = 0.05 + 0.12 * score_norm[i]
        harmonics = 1 + int(score_norm[i] * 4)
        tone = np.zeros(coda_len)
        for h in range(1, harmonics + 1):
            tone += (1.0/h) * sine(strat_freqs[i] * h, coda_t)
        tone *= vol * envelope(coda_len, attack=0.5, release=1.5)
        pan_l = 0.5 * (1 - strat_pans[i])
        pan_r = 0.5 * (1 + strat_pans[i])
        out[coda_s:coda_s+coda_len, 0] += tone * pan_l
        out[coda_s:coda_s+coda_len, 1] += tone * pan_r

    out *= 0.7 / (np.max(np.abs(out)) + 1e-10)
    return out


# ── Piece 2: Evolutionary Stable Strategy ────────────────────────────

def evolutionary_ess():
    """Hawks and Doves: population dynamics converging to ESS."""
    np.random.seed(21)
    dur = 55.0
    n = int(dur * SR)
    out = np.zeros((n, 2))
    t = np.arange(n) / SR

    # Drone
    drone = 0.07 * sine(55, t) * envelope(n, attack=2.0, release=3.0)
    out[:, 0] += drone
    out[:, 1] += drone

    # Hawk-Dove game: V=4 (resource value), C=6 (fighting cost)
    # Hawk vs Hawk: (V-C)/2 = -1 each (costly fight)
    # Hawk vs Dove: V,0 (hawk gets all)
    # Dove vs Dove: V/2, V/2 (share)
    # ESS: p_hawk = V/C = 2/3
    V, C = 4.0, 6.0
    ess_hawk = V / C  # 2/3

    pop_size = 30
    steps = 500
    time_per_step = dur / steps

    # Initial population: 50% hawks
    hawk_ratio = 0.5
    hawk_freq_base = 174.61  # F3 (aggressive, low)
    dove_freq_base = 293.66  # D4 (gentle, higher)

    for step in range(steps):
        step_time = step * time_per_step
        pos = int(step_time * SR)

        # Fitness calculation
        p_h = hawk_ratio
        p_d = 1 - p_h
        # Hawk fitness: p_h*(V-C)/2 + p_d*V
        fit_h = p_h * (V - C) / 2 + p_d * V
        # Dove fitness: p_h*0 + p_d*V/2
        fit_d = p_d * V / 2

        # Replicator dynamics
        avg_fit = p_h * fit_h + p_d * fit_d + 1e-10
        hawk_ratio = np.clip(p_h + 0.003 * p_h * (fit_h - avg_fit) / abs(avg_fit), 0.01, 0.99)

        # Sound: hawk component (left-biased) and dove component (right-biased)
        note_len = max(int(time_per_step * SR), 100)
        if pos + note_len >= n:
            continue
        nt = np.arange(note_len) / SR
        env = envelope(note_len, attack=0.003, release=0.01)

        # Hawks: FM harsh tone, volume proportional to hawk_ratio
        hawk_vol = 0.15 * hawk_ratio
        # More hawks = more fighting = more dissonance (mod_depth increases)
        hawk_mod = 2 + 6 * hawk_ratio  # 2-8
        hawk_tone = hawk_vol * fm_tone(hawk_freq_base, 7, hawk_mod, nt) * env

        # Doves: pure sine gentle, volume proportional to dove_ratio
        dove_vol = 0.12 * (1 - hawk_ratio)
        dove_harmonics = sine(dove_freq_base, nt) + 0.3 * sine(dove_freq_base * 2, nt)
        dove_tone = dove_vol * dove_harmonics * env

        # Encounters: occasional fight/share events
        if np.random.random() < 0.15:
            encounter_type = np.random.random()
            enc_len = min(int(0.08 * SR), n - pos)
            enc_t = np.arange(enc_len) / SR
            enc_env = envelope(enc_len, attack=0.002, release=0.03)

            if encounter_type < hawk_ratio * hawk_ratio:
                # Hawk-Hawk: costly fight — loud dissonant burst
                fight = 0.2 * fm_tone(hawk_freq_base, 11, 8, enc_t) * enc_env
                fight += 0.15 * fm_tone(hawk_freq_base * np.sqrt(2), 13, 6, enc_t) * enc_env
                out[pos:pos+enc_len, 0] += fight * 0.7
                out[pos:pos+enc_len, 1] += fight * 0.3
            elif encounter_type < hawk_ratio:
                # Hawk-Dove: quick resolution — sharp click + quiet retreat
                c = click(min(400, enc_len), hawk_freq_base * 2)
                cl = len(c)
                if pos + cl < n:
                    out[pos:pos+cl, 0] += 0.10 * c * 0.6
                    out[pos:pos+cl, 1] += 0.10 * c * 0.4
            else:
                # Dove-Dove: gentle sharing — soft fifth
                share = 0.06 * sine(dove_freq_base, enc_t) * enc_env
                share += 0.06 * sine(dove_freq_base * 1.5, enc_t) * enc_env
                out[pos:pos+enc_len, 0] += share * 0.4
                out[pos:pos+enc_len, 1] += share * 0.6

        # Background population texture
        out[pos:pos+note_len, 0] += hawk_tone * 0.7 + dove_tone * 0.3
        out[pos:pos+note_len, 1] += hawk_tone * 0.3 + dove_tone * 0.7

    # Coda (last 4s): ESS equilibrium chord
    coda_s = int(51 * SR)
    coda_len = n - coda_s
    coda_t = np.arange(coda_len) / SR
    # Hawks at 2/3 intensity, doves at 1/3
    h_coda = 0.10 * fm_tone(hawk_freq_base, 5, 2, coda_t)  # mellowed hawk
    d_coda = 0.08 * (sine(dove_freq_base, coda_t) + 0.5 * sine(dove_freq_base * 1.5, coda_t))
    coda_env = envelope(coda_len, attack=0.5, release=2.0)
    out[coda_s:, 0] += (h_coda * 0.67 + d_coda * 0.33) * coda_env
    out[coda_s:, 1] += (h_coda * 0.33 + d_coda * 0.67) * coda_env

    out *= 0.7 / (np.max(np.abs(out)) + 1e-10)
    return out


# ── Piece 3: Nash Equilibrium ────────────────────────────────────────

def nash_equilibrium():
    """Two players search a 5x5 payoff matrix via best-response dynamics."""
    np.random.seed(22)
    dur = 55.0
    n = int(dur * SR)
    out = np.zeros((n, 2))
    t = np.arange(n) / SR

    # Drone
    drone = 0.07 * sine(55, t) * envelope(n, attack=2.0, release=3.0)
    out[:, 0] += drone
    out[:, 1] += drone

    # 5x5 payoff matrix (coordination game with unique Nash eq at (2,2))
    # Row player payoffs
    payoff_r = np.array([
        [3, 1, 0, 1, 0],
        [1, 4, 2, 1, 0],
        [0, 2, 6, 2, 1],  # row 2 col 2 = Nash (6,6)
        [1, 1, 2, 3, 1],
        [0, 0, 1, 1, 2],
    ], dtype=float)
    payoff_c = payoff_r.T  # symmetric game

    # Frequencies for row/col choices (pentatonic A minor)
    row_freqs = np.array([220.0, 246.94, 261.63, 293.66, 329.63])  # A3,B3,C4,D4,E4
    col_freqs = np.array([329.63, 369.99, 392.00, 440.00, 493.88])  # E4,F#4,G4,A4,B4

    # Nash equilibrium: row=2, col=2 => 261.63 + 392.00 (C4+G4 = perfect fifth!)

    steps = 400
    time_per_step = dur / steps

    # Start at random positions
    rng = np.random.RandomState(22)
    row_pos = 0.0  # continuous position [0,4]
    col_pos = 4.0

    for step in range(steps):
        step_time = step * time_per_step
        pos = int(step_time * SR)
        note_len = max(int(time_per_step * SR), 200)
        if pos + note_len >= n:
            continue

        # Current discrete choices
        r_idx = int(np.clip(np.round(row_pos), 0, 4))
        c_idx = int(np.clip(np.round(col_pos), 0, 4))

        # Best response with noise
        # Row player's best response to col_idx
        br_r = np.argmax(payoff_r[:, c_idx])
        # Col player's best response to r_idx
        br_c = np.argmax(payoff_c[r_idx, :])

        # Smooth movement toward best response (with inertia + noise)
        row_pos += 0.08 * (br_r - row_pos) + rng.normal(0, 0.15)
        col_pos += 0.08 * (br_c - col_pos) + rng.normal(0, 0.15)
        row_pos = np.clip(row_pos, 0, 4)
        col_pos = np.clip(col_pos, 0, 4)

        # Current payoff as quality metric
        payoff = payoff_r[r_idx, c_idx]
        max_payoff = 6.0
        quality = payoff / max_payoff  # 0 to 1

        nt = np.arange(note_len) / SR
        env = envelope(note_len, attack=0.003, release=0.015)

        # Row player (left channel): frequency from position
        r_freq = row_freqs[r_idx]
        # Col player (right channel): frequency from position  
        c_freq = col_freqs[c_idx]

        # High quality = consonant, pure tones
        # Low quality = dissonant, FM noise
        if quality > 0.8:  # Near Nash equilibrium
            # Pure consonance — both tones clear
            harmonics_r = 1 + int(quality * 4)
            tone_r = np.zeros(note_len)
            for h in range(1, harmonics_r + 1):
                tone_r += (1.0/h) * sine(r_freq * h, nt)
            tone_r *= 0.10 * env

            tone_c = np.zeros(note_len)
            for h in range(1, harmonics_r + 1):
                tone_c += (1.0/h) * sine(c_freq * h, nt)
            tone_c *= 0.10 * env
        elif quality > 0.3:  # Moderate
            tone_r = 0.08 * sine(r_freq, nt) * env
            # slight FM for uncertainty
            tone_c = 0.08 * fm_tone(c_freq, 3, 1.5 * (1 - quality), nt) * env
        else:  # Poor — dissonant
            tone_r = 0.06 * fm_tone(r_freq, 7, 3, nt) * env
            tone_c = 0.06 * fm_tone(c_freq, 9, 4, nt) * env

        out[pos:pos+note_len, 0] += tone_r * 0.8 + tone_c * 0.2
        out[pos:pos+note_len, 1] += tone_r * 0.2 + tone_c * 0.8

        # Click when strategy changes
        if step > 0 and (r_idx != int(np.clip(np.round(row_pos - 0.08 * (br_r - row_pos)), 0, 4))):
            c_click = click(200, 2000)
            cl = len(c_click)
            if pos + cl < n:
                out[pos:pos+cl, 0] += 0.05 * c_click
                out[pos:pos+cl, 1] += 0.05 * c_click

    # Coda: Nash equilibrium chord (last 4s)
    coda_s = int(51 * SR)
    coda_len = n - coda_s
    coda_t = np.arange(coda_len) / SR
    # C4 + G4 = perfect fifth (Nash equilibrium row=2, col=2)
    nash_r = row_freqs[2]  # C4 = 261.63
    nash_c = col_freqs[2]  # G4 = 392.00
    coda_tone = np.zeros(coda_len)
    for h in range(1, 6):
        coda_tone += (1.0/h) * sine(nash_r * h, coda_t)
        coda_tone += (0.8/h) * sine(nash_c * h, coda_t)
    coda_tone *= 0.10 * envelope(coda_len, attack=0.8, release=2.0)
    out[coda_s:, 0] += coda_tone * 0.6
    out[coda_s:, 1] += coda_tone * 0.6

    out *= 0.7 / (np.max(np.abs(out)) + 1e-10)
    return out


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    print("Phase 20: Game Theory")
    print("=" * 50)

    print("\n1. Prisoner's Dilemma Tournament")
    audio = prisoners_dilemma()
    write_wav("output/game_1_prisoners_dilemma.wav", audio)

    print("\n2. Evolutionary Stable Strategy")
    audio = evolutionary_ess()
    write_wav("output/game_2_evolutionary_ess.wav", audio)

    print("\n3. Nash Equilibrium")
    audio = nash_equilibrium()
    write_wav("output/game_3_nash_equilibrium.wav", audio)

    print("\nDone! 3 tracks generated.")
