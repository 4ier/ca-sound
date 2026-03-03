"""
Fractals — Computational Aesthetics Phase 5

Three pieces exploring the Mandelbrot set and Julia sets as sound:

1. Escape Orbits (55s, stereo)
   Scan a line through the Mandelbrot set (Im=0, Re=-2→0.5).
   Each point's escape time → note duration; final |z| → pitch;
   orbit trajectory → stereo wobble. Interior points (non-escaping) = deep drone.

2. Julia Morphs (50s, stereo)  
   Four Julia sets with c traveling along the main cardioid boundary.
   Each Julia set rendered as a granular texture: escape times of a grid
   become grain densities and pitches. The four sets crossfade as c moves
   from stable (c=0) through the cardioid edge to chaotic (c=-0.75+0.1j).

3. Mandelbrot Zoom (45s, stereo)
   Zoom into the Mandelbrot set near the Misiurewicz point (-0.77568377, 0.13646737).
   Each zoom level: the escape-time histogram becomes a spectral profile.
   As we zoom deeper, micro-structure self-similarity = recurring harmonic motifs.
"""

import numpy as np
import struct
import os

SR = 44100
OUTPUT = "output"
os.makedirs(OUTPUT, exist_ok=True)


def save_wav(path: str, data: np.ndarray, sr: int = SR):
    """Save float64 array as 16-bit WAV (mono or stereo)."""
    if data.ndim == 1:
        channels = 1
    else:
        channels = data.shape[1]
    # Normalize
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * 0.9
    pcm = (data * 32767).astype(np.int16)
    raw = pcm.tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(raw)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, channels, sr, sr * channels * 2, channels * 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(raw)))
        f.write(raw)
    print(f"  Saved {path} ({len(data)/sr:.1f}s)")


def mandelbrot_escape(c, max_iter=256):
    """Return (escape_time, final_abs, orbit) for a complex point c."""
    z = 0 + 0j
    orbit = [z]
    for i in range(max_iter):
        z = z * z + c
        orbit.append(z)
        if abs(z) > 2.0:
            return i + 1, abs(z), orbit
    return max_iter, abs(z), orbit


def julia_escape(z0, c, max_iter=256):
    """Return escape time for Julia set at z0 with parameter c."""
    z = z0
    for i in range(max_iter):
        z = z * z + c
        if abs(z) > 2.0:
            return i + 1
    return max_iter


# === Piece 1: Escape Orbits ===
def make_escape_orbits():
    print("Piece 1: Escape Orbits")
    duration = 55.0
    n_points = 200  # points along the real axis scan
    re_vals = np.linspace(-2.0, 0.5, n_points)
    max_iter = 256

    # Compute escapes
    escapes = []
    for re in re_vals:
        c = complex(re, 0.0)
        etime, fabs, orbit = mandelbrot_escape(c, max_iter)
        escapes.append((etime, fabs, orbit, re))

    # Each point gets a time slice
    slice_dur = duration / n_points
    total_samples = int(duration * SR)
    out = np.zeros((total_samples, 2))

    for idx, (etime, fabs, orbit, re) in enumerate(escapes):
        t0 = idx * slice_dur
        # Duration of this note proportional to escape time
        note_dur = min(slice_dur * 0.95, slice_dur * (etime / max_iter) * 3)
        note_dur = max(note_dur, 0.02)
        n_samp = int(note_dur * SR)
        t = np.arange(n_samp) / SR

        if etime >= max_iter:
            # Interior point: deep drone
            freq = 55.0  # A1
            env = np.ones(n_samp) * 0.3
            # Fade in/out
            fade = min(n_samp // 4, int(0.05 * SR))
            env[:fade] *= np.linspace(0, 1, fade)
            env[-fade:] *= np.linspace(1, 0, fade)
            wave = np.sin(2 * np.pi * freq * t) * env
            pan_l, pan_r = 0.5, 0.5
        else:
            # Escaping point: pitch from log(escape_time)
            freq = 110 * 2 ** (np.log2(etime + 1) / 2)  # spread across octaves
            freq = np.clip(freq, 80, 4000)

            # Harmonics based on final |z|
            n_harmonics = min(int(fabs), 8)
            wave = np.zeros(n_samp)
            for h in range(1, n_harmonics + 2):
                wave += np.sin(2 * np.pi * freq * h * t) / h

            # Envelope: attack-decay
            env = np.exp(-t / (note_dur * 0.4))
            fade_in = min(int(0.005 * SR), n_samp // 4)
            env[:fade_in] *= np.linspace(0, 1, fade_in)
            wave *= env * 0.4

            # Stereo from orbit: average real part of orbit → pan
            orbit_re = np.mean([o.real for o in orbit[:etime + 1]])
            pan = np.clip((orbit_re + 2) / 4, 0, 1)  # map [-2,2] → [0,1]
            pan_l = np.sqrt(1 - pan)
            pan_r = np.sqrt(pan)

        start = int(t0 * SR)
        end = min(start + n_samp, total_samples)
        actual = end - start
        out[start:end, 0] += wave[:actual] * pan_l
        out[start:end, 1] += wave[:actual] * pan_r

    save_wav(os.path.join(OUTPUT, "fractal_1_escape_orbits.wav"), out)


# === Piece 2: Julia Morphs ===
def make_julia_morphs():
    print("Piece 2: Julia Morphs")
    duration = 50.0
    total_samples = int(duration * SR)
    out = np.zeros((total_samples, 2))

    # Four c values along the cardioid boundary and beyond
    c_values = [
        0.0 + 0.0j,          # center, fully connected
        -0.4 + 0.6j,         # near cardioid, interesting dendrites
        -0.75 + 0.0j,        # period-2 bulb boundary
        -0.12 + 0.75j,       # dendritic, near Siegel disk
    ]

    grid_n = 40  # 40x40 grid per Julia set
    max_iter = 128
    section_dur = duration / len(c_values)

    for ci, c in enumerate(c_values):
        print(f"  Julia c={c}")
        t_start = ci * section_dur

        # Sample Julia set on [-1.5,1.5]^2
        xs = np.linspace(-1.5, 1.5, grid_n)
        ys = np.linspace(-1.5, 1.5, grid_n)
        etimes = []
        for y in ys:
            for x in xs:
                et = julia_escape(complex(x, y), c, max_iter)
                etimes.append(et)
        etimes = np.array(etimes, dtype=float)

        # Histogram of escape times → spectral profile
        bins = np.linspace(1, max_iter, 32)
        hist, _ = np.histogram(etimes, bins=bins)
        hist = hist / (hist.max() + 1e-8)

        # Synthesize: each bin = a frequency band
        n_samp = int(section_dur * SR)
        t = np.arange(n_samp) / SR
        section = np.zeros((n_samp, 2))

        base_freq = 80.0
        for bi in range(len(hist)):
            if hist[bi] < 0.05:
                continue
            freq = base_freq * (1.5 ** bi)  # spread in fifths
            if freq > 6000:
                break
            amp = hist[bi] * 0.15

            # Slight detuning for stereo width
            detune = 1.002 if bi % 2 == 0 else 0.998
            section[:, 0] += amp * np.sin(2 * np.pi * freq * t)
            section[:, 1] += amp * np.sin(2 * np.pi * freq * detune * t)

        # Add granular texture from individual escape times
        n_grains = 80
        for gi in range(n_grains):
            et = etimes[gi * len(etimes) // n_grains]
            grain_freq = 200 * 2 ** (et / max_iter * 3)  # 200-1600 Hz
            grain_dur = 0.02 + (et / max_iter) * 0.08
            grain_samp = int(grain_dur * SR)
            grain_t = np.arange(grain_samp) / SR

            # Hann window grain
            window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(grain_samp) / grain_samp))
            grain = np.sin(2 * np.pi * grain_freq * grain_t) * window * 0.1

            # Place grain
            pos = int(gi / n_grains * n_samp)
            end = min(pos + grain_samp, n_samp)
            actual = end - pos
            pan = (gi % grid_n) / grid_n
            section[pos:end, 0] += grain[:actual] * np.sqrt(1 - pan)
            section[pos:end, 1] += grain[:actual] * np.sqrt(pan)

        # Crossfade envelope
        fade = int(0.3 * SR)
        env = np.ones(n_samp)
        env[:fade] *= np.linspace(0, 1, fade)
        env[-fade:] *= np.linspace(1, 0, fade)
        section[:, 0] *= env
        section[:, 1] *= env

        start = int(t_start * SR)
        end = min(start + n_samp, total_samples)
        actual = end - start
        out[start:end] += section[:actual]

    save_wav(os.path.join(OUTPUT, "fractal_2_julia_morphs.wav"), out)


# === Piece 3: Mandelbrot Zoom ===
def make_mandelbrot_zoom():
    print("Piece 3: Mandelbrot Zoom")
    duration = 45.0
    total_samples = int(duration * SR)
    out = np.zeros((total_samples, 2))

    # Zoom center: Misiurewicz point
    center = -0.77568377 + 0.13646737j
    n_levels = 30
    max_iter = 300
    grid_n = 50  # 50x50 grid per zoom level

    level_dur = duration / n_levels

    for li in range(n_levels):
        zoom = 2.0 / (1.5 ** li)  # each level zooms 1.5x
        print(f"  Zoom level {li}: window size {zoom:.6f}")

        # Sample grid
        xs = np.linspace(center.real - zoom, center.real + zoom, grid_n)
        ys = np.linspace(center.imag - zoom, center.imag + zoom, grid_n)
        etimes = []
        for y in ys:
            for x in xs:
                et, _, _ = mandelbrot_escape(complex(x, y), max_iter)
                etimes.append(et)
        etimes = np.array(etimes, dtype=float)

        # Escape time histogram → spectral fingerprint
        bins = np.linspace(1, max_iter, 24)
        hist, _ = np.histogram(etimes, bins=bins)
        hist = hist / (hist.max() + 1e-8)

        # Interior ratio → drone amplitude
        interior_ratio = np.sum(etimes >= max_iter) / len(etimes)

        n_samp = int(level_dur * SR)
        t = np.arange(n_samp) / SR
        section = np.zeros((n_samp, 2))

        # Spectral synthesis from histogram
        base_freq = 65.0  # C2
        for bi in range(len(hist)):
            if hist[bi] < 0.03:
                continue
            # Map bin to harmonic series with slight stretch
            freq = base_freq * (bi + 1) * (1.0 + li * 0.005)
            if freq > 8000:
                break
            amp = hist[bi] * 0.12

            # Detune for depth
            section[:, 0] += amp * np.sin(2 * np.pi * freq * t)
            section[:, 1] += amp * np.sin(2 * np.pi * freq * 1.001 * t)

        # Interior drone
        if interior_ratio > 0.05:
            drone_freq = 55.0 * (1 + li * 0.02)
            drone = np.sin(2 * np.pi * drone_freq * t) * interior_ratio * 0.3
            section[:, 0] += drone
            section[:, 1] += drone

        # Envelope
        fade = int(0.2 * SR)
        env = np.ones(n_samp)
        env[:fade] *= np.linspace(0, 1, fade)
        env[-fade:] *= np.linspace(1, 0, fade)
        section[:, 0] *= env
        section[:, 1] *= env

        start = int(li * level_dur * SR)
        end = min(start + n_samp, total_samples)
        actual = end - start
        out[start:end] += section[:actual]

    save_wav(os.path.join(OUTPUT, "fractal_3_mandelbrot_zoom.wav"), out)


if __name__ == "__main__":
    make_escape_orbits()
    make_julia_morphs()
    make_mandelbrot_zoom()
    print("Done — all fractal pieces generated.")
