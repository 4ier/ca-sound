"""
Busy Beaver — The Sound of Undecidability

Three pieces exploring Turing machines through sound:

1. Zoo (60s) — BB(2), BB(3), BB(4) + non-halting machines race in parallel.
2. BB(2) Portrait (30s) — The 6-step journey expanded into musical phrases.
3. The Wall (50s) — BB(4) left channel vs never-halting cycler right channel.
"""

import struct
import math
import os
import numpy as np

RATE = 44100

def write_wav(filename, data, sr=RATE, channels=1):
    """Write numpy float array to WAV (16-bit)."""
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    data = np.clip(data, -1.0, 1.0)
    pcm = (data * 32767).astype(np.int16)
    raw = pcm.tobytes()
    num_frames = len(data) // channels
    with open(filename, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(raw)))
        f.write(b'WAVEfmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, channels, sr, sr*channels*2, channels*2, 16))
        f.write(b'data')
        f.write(struct.pack('<I', len(raw)))
        f.write(raw)


# --- Turing Machine Simulator ---

class TM:
    def __init__(self, name, transitions, start='A'):
        self.name = name
        self.trans = transitions
        self.state = start
        self.tape = {}
        self.head = 0
        self.steps = 0
        self.halted = False
        self.history = []

    def step(self):
        if self.halted: return False
        r = self.tape.get(self.head, 0)
        key = (self.state, r)
        if key not in self.trans:
            self.halted = True; return False
        w, d, ns = self.trans[key]
        self.tape[self.head] = w
        self.head += 1 if d == 'R' else -1
        self.state = ns
        self.steps += 1
        if ns == 'H': self.halted = True
        ones = sum(1 for v in self.tape.values() if v)
        self.history.append((self.state, self.head, ones))
        return not self.halted

    def run(self, n=10000):
        while self.steps < n and not self.halted: self.step()
        return self.history


def bb2():
    return TM("BB(2)", {('A',0):(1,'R','B'),('A',1):(1,'L','B'),('B',0):(1,'L','A'),('B',1):(1,'R','H')})

def bb3():
    return TM("BB(3)", {('A',0):(1,'R','B'),('A',1):(1,'R','H'),('B',0):(0,'R','C'),('B',1):(1,'R','B'),('C',0):(1,'L','C'),('C',1):(1,'L','A')})

def bb4():
    return TM("BB(4)", {('A',0):(1,'R','B'),('A',1):(1,'L','B'),('B',0):(1,'L','A'),('B',1):(0,'L','C'),('C',0):(1,'R','H'),('C',1):(1,'L','D'),('D',0):(1,'R','D'),('D',1):(0,'R','A')})

def non_halter_cycler():
    return TM("Cycler", {('A',0):(1,'R','B'),('A',1):(0,'R','A'),('B',0):(1,'L','A'),('B',1):(1,'R','B')})

def non_halter_sweeper():
    return TM("Sweeper", {('A',0):(1,'R','B'),('A',1):(1,'R','A'),('B',0):(0,'R','A'),('B',1):(1,'R','B')})

def non_halter_counter():
    return TM("Counter", {('A',0):(1,'R','B'),('A',1):(0,'L','C'),('B',0):(0,'R','A'),('B',1):(1,'R','B'),('C',0):(1,'R','A'),('C',1):(1,'L','C')})


STATE_FREQ = {'A':220.0, 'B':277.18, 'C':329.63, 'D':392.0, 'H':73.42}

def np_rich(freq, t_arr, harmonics=5):
    s = np.zeros_like(t_arr)
    for n in range(1, harmonics+1):
        s += (1.0/n) * np.sin(2*np.pi*freq*n*t_arr)
    return s / 1.5

def np_adsr(t_arr, dur, a=0.01, d=0.05, sus=0.7, r=0.1):
    env = np.zeros_like(t_arr)
    for i, t in enumerate(t_arr):
        if t < a: env[i] = t/a
        elif t < a+d: env[i] = 1.0 - (1.0-sus)*(t-a)/d
        elif t < dur-r: env[i] = sus
        else: env[i] = sus * max(0, (dur-t)/r)
    return env


# --- Piece 1: Zoo ---
def generate_zoo(dur=60):
    print("  Generating Zoo...")
    machines = [bb2(), bb3(), bb4(), non_halter_cycler(), non_halter_sweeper(), non_halter_counter()]
    for m in machines: m.run(2000)
    pans = [-0.8, -0.4, 0.0, 0.3, 0.6, 0.9]
    N = int(dur * RATE)
    L = np.zeros(N); R = np.zeros(N)
    t_global = np.arange(N) / RATE

    for mi, m in enumerate(machines):
        if not m.history: continue
        ns = len(m.history)
        if m.halted:
            sd = min(dur*0.6/max(ns,1), 2.0)
        else:
            sd = dur / ns
        pan = pans[mi]
        lg = math.sqrt(0.5*(1-pan)); rg = math.sqrt(0.5*(1+pan))

        for si, (state, head, ones) in enumerate(m.history):
            ts = si * sd
            if ts >= dur: break
            te = min(ts + sd, dur)
            freq = STATE_FREQ.get(state, 220.0) * (1 + head*0.005)
            freq = max(50, min(freq, 4000))
            harm = min(1+ones, 8)
            i0 = int(ts*RATE); i1 = min(int(te*RATE), N)
            seg = (i1-i0)/RATE
            t_local = np.arange(i1-i0)/RATE
            env = np_adsr(t_local, seg, a=0.005, r=min(0.05, seg*0.3))
            s = np_rich(freq, t_global[i0:i1], harm) * env * 0.15/6
            L[i0:i1] += s * lg
            R[i0:i1] += s * rg

        if m.halted:
            mtime = ns * sd
            if mtime < dur:
                fs = int(mtime*RATE); fd = min(3.0, dur-mtime)
                fe = min(int((mtime+fd)*RATE), N)
                t_f = np.arange(fe-fs)/RATE
                env_f = 0.1*(1.0 - t_f/fd)
                s = env_f * np.sin(2*np.pi*73.42*t_global[fs:fe])
                L[fs:fe] += s*lg; R[fs:fe] += s*rg

    out = np.empty(N*2)
    out[0::2] = L; out[1::2] = R
    write_wav('output/beaver_1_zoo.wav', out, channels=2)
    print("    → output/beaver_1_zoo.wav")


# --- Piece 2: BB(2) Portrait ---
def generate_bb2_portrait(dur=30):
    print("  Generating BB(2) Portrait...")
    m = bb2()
    steps_data = []
    while not m.halted:
        st, hd, rd = m.state, m.head, m.tape.get(m.head, 0)
        m.step()
        steps_data.append((st, hd, rd))
    ns = len(steps_data)
    phrase = (dur-5)/ns
    N = int(dur*RATE)
    out = np.zeros(N)
    t_g = np.arange(N)/RATE

    for si, (state, hd, rd) in enumerate(steps_data):
        i0 = int(si*phrase*RATE); i1 = min(int((si+1)*phrase*RATE), N)
        t_l = np.arange(i1-i0)/RATE
        freq = STATE_FREQ.get(state, 220.0)
        env = np_adsr(t_l, phrase, a=0.1, d=0.2, sus=0.6, r=0.5)
        if state == 'A':
            vib = 1.0 + 0.01*np.sin(2*np.pi*5.5*t_g[i0:i1])
            s = 0.3*env*np.sin(2*np.pi*freq*vib*t_g[i0:i1])
        else:
            s = np_rich(freq, t_g[i0:i1], 5)*0.15*env + np_rich(freq*1.003, t_g[i0:i1], 4)*0.15*env
        out[i0:i1] += s

    # Coda
    cs = int(ns*phrase*RATE)
    cd = (N-cs)/RATE
    t_c = np.arange(N-cs)/RATE
    env_c = 0.25*(1.0-t_c/cd)**2
    out[cs:] += env_c*np.sin(2*np.pi*36.71*t_g[cs:])
    out[cs:] += env_c*0.3*np.sin(2*np.pi*73.42*t_g[cs:])

    write_wav('output/beaver_2_bb2_portrait.wav', out)
    print("    → output/beaver_2_bb2_portrait.wav")


# --- Piece 3: The Wall ---
def generate_wall(dur=50):
    print("  Generating The Wall...")
    bb = bb4(); bb.run(200)
    nh = non_halter_cycler(); nh.run(2000)

    N = int(dur*RATE)
    L = np.zeros(N); R = np.zeros(N)
    t_g = np.arange(N)/RATE

    # BB(4) accelerating schedule
    bbs = len(bb.history)
    weights = np.array([(1-(i/bbs))**0.5+0.1 for i in range(bbs)])
    times = weights/weights.sum()*35.0

    cur = 0.0
    for si, (state, head, ones) in enumerate(bb.history):
        sd = times[si]
        i0 = int(cur*RATE); i1 = min(int((cur+sd)*RATE), N)
        if i1 <= i0: cur += sd; continue
        freq = 110+ones*30
        harm = min(2+(ord(state)-ord('A')), 8)
        t_l = np.arange(i1-i0)/RATE
        env = np_adsr(t_l, sd, a=0.003, d=0.02, sus=0.8, r=min(0.05, sd*0.3))
        L[i0:i1] += np_rich(freq, t_g[i0:i1], harm)*0.3*env
        cur += sd

    # Post-halt fade
    hs = int(cur*RATE); fd = 3.0; he = min(hs+int(fd*RATE), N)
    if hs < N:
        t_h = np.arange(he-hs)/RATE
        L[hs:he] += 0.15*(1-t_h/fd)**2*np.sin(2*np.pi*73.42*t_g[hs:he])

    # Right: non-halter drone
    nhs = len(nh.history)
    sd_nh = dur/nhs
    for si, (state, head, ones) in enumerate(nh.history):
        ts = si*sd_nh
        if ts >= dur: break
        i0 = int(ts*RATE); i1 = min(int((ts+sd_nh)*RATE), N)
        freq = 55.0*(1+ones*0.02)
        mod = 0.8+0.2*np.sin(2*np.pi*0.1*t_g[i0:i1])
        R[i0:i1] += 0.2*mod*np.sin(2*np.pi*freq*t_g[i0:i1])
        R[i0:i1] += 0.06*mod*np.sin(2*np.pi*freq*1.5*t_g[i0:i1])

    out = np.empty(N*2)
    out[0::2] = L; out[1::2] = R
    write_wav('output/beaver_3_wall.wav', out, channels=2)
    print("    → output/beaver_3_wall.wav")


if __name__ == '__main__':
    print("Busy Beaver — The Sound of Undecidability")
    print("="*50)
    generate_zoo()
    generate_bb2_portrait()
    generate_wall()
    print("\nDone.")
