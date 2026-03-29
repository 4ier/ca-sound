#!/usr/bin/env python3
"""Phase 22: Signal Processing -- DFT, convolution, wavelets.

Three pieces:
1. DFT Decomposition (55s, stereo) -- A complex tone is decomposed into its Fourier
   components one by one. Starts as a rich buzzy timbre; each DFT bin extraction
   removes one frequency and places it in the stereo field by its frequency.
   Low bins left, high bins right. The original dissolves as its spectrum is laid bare.

2. Convolution (50s, stereo) -- An impulse (click) meets various kernels.
   Dry signal left, wet signal right. Three kernels morph: identity -> low-pass blur ->
   resonant bandpass -> comb/echo. Each convolution reshapes the timbre in real-time.
   The impulse response IS the sound.

3. Wavelet Transform (55s, stereo) -- A chirp signal analyzed at multiple scales.
   Low-frequency wavelets = slow pulsing bass (left), high-frequency wavelets =
   rapid treble shimmer (right). Scale increases over time, revealing the chirp's
   time-frequency structure. Morlet wavelets as musical grains.
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


def crossfade(a, b, n):
    """Crossfade last n samples of a with first n of b."""
    fade_out = np.linspace(1, 0, n)
    fade_in = np.linspace(0, 1, n)
    result = np.zeros(len(a) + len(b) - n)
    result[:len(a)] = a
    result[len(a) - n:len(a)] *= fade_out
    result[len(a) - n:len(a)] += b[:n] * fade_in
    result[len(a):] = b[n:]
    return result


# ── Piece 1: DFT Decomposition ──────────────────────────────────────────────

def dft_decomposition():
    """A rich tone decomposed into its Fourier spectrum, one bin at a time."""
    print("Generating DFT Decomposition...")
    duration = 55.0
    N = int(duration * SR)
    t = np.arange(N) / SR
    out_L = np.zeros(N)
    out_R = np.zeros(N)

    # Build the original complex tone: fundamental + 15 harmonics
    fundamental = 110.0  # A2
    n_harmonics = 16
    # Harmonic amplitudes: 1/k with slight randomness for organic feel
    rng = np.random.RandomState(42)
    harm_amps = np.array([1.0 / (k + 1) ** 0.7 for k in range(n_harmonics)])
    harm_amps *= rng.uniform(0.7, 1.0, n_harmonics)
    harm_phases = rng.uniform(0, 2 * np.pi, n_harmonics)

    # Phase 1 (0-8s): Full complex tone, stereo center
    # Phase 2 (8-48s): Extract harmonics one by one, highest first
    # Phase 3 (48-55s): Only fundamental remains, then fade

    # Section boundaries
    t_intro_end = 8.0
    t_extract_end = 48.0

    # Global drone: 55 Hz (A1) very low
    drone = sine(55.0, t) * 0.06 * envelope(N, attack=2.0, release=3.0)

    # Build the full complex tone
    full_tone = np.zeros(N)
    for k in range(n_harmonics):
        freq = fundamental * (k + 1)
        full_tone += harm_amps[k] * sine(freq, t, harm_phases[k])
    full_tone /= np.max(np.abs(full_tone)) + 1e-9
    full_tone *= 0.5

    # Extraction schedule: remove harmonics 15 down to 1 (keep fundamental last)
    extract_order = list(range(n_harmonics - 1, 0, -1))  # [15, 14, ..., 1]
    extract_duration = t_extract_end - t_intro_end
    time_per_extract = extract_duration / len(extract_order)

    # Track which harmonics are still in the "residual"
    residual_mask = np.ones((n_harmonics, N))  # 1 = present, 0 = extracted

    for i, k in enumerate(extract_order):
        ext_start = t_intro_end + i * time_per_extract
        ext_mid = ext_start + time_per_extract * 0.3
        ext_end = ext_start + time_per_extract

        s_start = int(ext_start * SR)
        s_mid = int(ext_mid * SR)
        s_end = min(int(ext_end * SR), N)

        # Fade out this harmonic from residual
        fade_len = s_mid - s_start
        if fade_len > 0:
            residual_mask[k, s_start:s_mid] = np.linspace(1, 0, fade_len)
        residual_mask[k, s_mid:] = 0

        # The extracted harmonic plays as a separate voice
        freq = fundamental * (k + 1)
        # Pan by frequency: low = left, high = right
        pan = k / (n_harmonics - 1)  # 0=left, 1=right
        # Extracted voice amplitude envelope
        voice_len = s_end - s_mid
        if voice_len > 0:
            voice_env = np.zeros(N)
            voice_env[s_mid:s_end] = envelope(voice_len, attack=0.1, release=0.5)
            # Also sustain at low level after extraction
            sustain_level = 0.15
            voice_env[s_end:] = sustain_level
            # Final fadeout
            fade_start = int((duration - 3.0) * SR)
            if fade_start < N:
                fade_n = N - fade_start
                voice_env[fade_start:] *= np.linspace(1, 0, fade_n)

            voice = harm_amps[k] * sine(freq, t, harm_phases[k]) * voice_env * 0.3
            out_L += voice * (1 - pan)
            out_R += voice * pan

        # Click marker when extraction happens
        c = click(300, freq * 0.5)
        if s_mid + len(c) < N:
            out_L[s_mid:s_mid + len(c)] += c * 0.15 * (1 - pan)
            out_R[s_mid:s_mid + len(c)] += c * 0.15 * pan

    # Build the residual (remaining harmonics) playing center
    residual = np.zeros(N)
    for k in range(n_harmonics):
        freq = fundamental * (k + 1)
        residual += harm_amps[k] * sine(freq, t, harm_phases[k]) * residual_mask[k]
    residual /= np.max(np.abs(residual)) + 1e-9
    residual *= 0.4

    # Intro envelope for residual
    intro_env = np.ones(N)
    att_n = int(1.5 * SR)
    intro_env[:att_n] = np.linspace(0, 1, att_n)

    out_L += residual * intro_env * 0.5 + drone
    out_R += residual * intro_env * 0.5 + drone

    # Final fadeout
    fade_n = int(4.0 * SR)
    out_L[-fade_n:] *= np.linspace(1, 0, fade_n)
    out_R[-fade_n:] *= np.linspace(1, 0, fade_n)

    # Normalize
    out = np.stack([out_L, out_R], axis=1)
    peak = np.max(np.abs(out))
    if peak > 0:
        out *= 0.85 / peak
    return out


# ── Piece 2: Convolution ────────────────────────────────────────────────────

def convolution():
    """An impulse train convolved with morphing kernels."""
    print("Generating Convolution...")
    duration = 50.0
    N = int(duration * SR)
    t = np.arange(N) / SR
    out_L = np.zeros(N)
    out_R = np.zeros(N)

    # Dry signal: rhythmic impulse train with varying intervals
    # Use a musical pattern: impulses at subdivision of 120 BPM
    bpm = 120.0
    beat_samples = int(60.0 / bpm * SR)
    subdivisions = [1.0, 0.5, 0.75, 0.25, 1.0, 0.5, 0.5, 0.25, 0.25, 0.5]

    # Build impulse positions
    impulse_positions = []
    pos = 0
    while pos < N:
        impulse_positions.append(pos)
        subdiv = subdivisions[len(impulse_positions) % len(subdivisions)]
        pos += int(beat_samples * subdiv)

    # Create dry impulse signal (short clicks at 440 Hz)
    dry = np.zeros(N)
    click_len = int(0.003 * SR)  # 3ms click
    click_sig = np.sin(2 * np.pi * 440 * np.arange(click_len) / SR)
    click_sig *= np.exp(-np.arange(click_len) / SR * 300)

    for p in impulse_positions:
        end = min(p + click_len, N)
        dry[p:end] += click_sig[:end - p]

    # Four kernel phases with crossfades:
    # 1. Identity (0-10s): output ≈ input
    # 2. Low-pass blur (10-22s): smooth the clicks into gentle bumps
    # 3. Resonant bandpass (22-36s): ring at a specific frequency
    # 4. Comb filter / echo (36-50s): rhythmic echoes

    def apply_kernel_segment(signal, kernel, start_s, end_s):
        """Convolve a segment with a kernel, return full-length result."""
        seg = np.zeros(len(signal))
        seg[start_s:end_s] = signal[start_s:end_s]
        # Use overlap-save for efficiency
        conv = np.convolve(seg, kernel, mode='full')[:len(signal)]
        return conv

    # Kernel 1: near-identity (very short)
    k1 = np.zeros(5)
    k1[2] = 1.0

    # Kernel 2: Gaussian low-pass
    k2_len = int(0.02 * SR)  # 20ms
    k2_x = np.linspace(-3, 3, k2_len)
    k2 = np.exp(-k2_x ** 2 / 2)
    k2 /= np.sum(k2)

    # Kernel 3: Resonant bandpass (damped sinusoid)
    k3_len = int(0.05 * SR)  # 50ms ring
    k3_t = np.arange(k3_len) / SR
    k3_freq = 660.0  # E5
    k3 = np.sin(2 * np.pi * k3_freq * k3_t) * np.exp(-k3_t * 60)
    k3 /= np.max(np.abs(k3)) + 1e-9
    k3 *= 0.3

    # Kernel 4: Comb filter (echo at ~125ms intervals)
    k4_len = int(0.5 * SR)
    k4 = np.zeros(k4_len)
    echo_delay = int(0.125 * SR)
    for i in range(4):
        idx = i * echo_delay
        if idx < k4_len:
            k4[idx] = 0.7 ** i
    k4 /= np.max(np.abs(k4)) + 1e-9

    # Phase boundaries
    phases = [(0, 10), (10, 22), (22, 36), (36, 50)]
    kernels = [k1, k2, k3, k4]
    xfade = int(1.5 * SR)  # 1.5s crossfade

    wet = np.zeros(N)
    for pi, ((start_t, end_t), kernel) in enumerate(zip(phases, kernels)):
        s = int(start_t * SR)
        e = min(int(end_t * SR), N)

        # Convolve dry signal with this kernel for this segment
        seg_wet = np.convolve(dry, kernel, mode='full')[:N]

        # Apply windowed segment with crossfade
        seg_env = np.zeros(N)
        seg_env[s:e] = 1.0

        # Fade in
        fi = min(xfade, e - s)
        seg_env[s:s + fi] = np.linspace(0, 1, fi)
        # Fade out
        fo = min(xfade, e - s)
        seg_env[e - fo:e] = np.linspace(1, 0, fo)

        wet += seg_wet * seg_env

    # Normalize wet
    wet_peak = np.max(np.abs(wet))
    if wet_peak > 0:
        wet /= wet_peak
    wet *= 0.5

    dry_norm = dry / (np.max(np.abs(dry)) + 1e-9) * 0.4

    # Dry = left, Wet = right, with some bleed
    out_L = dry_norm * 0.7 + wet * 0.3
    out_R = wet * 0.7 + dry_norm * 0.3

    # 55Hz drone
    drone = sine(55.0, t) * 0.05 * envelope(N, attack=2.0, release=2.0)
    out_L += drone
    out_R += drone

    # Kernel transition clicks at phase boundaries
    for start_t, end_t in phases[1:]:
        s = int(start_t * SR)
        c = click(400, 1500)
        if s + len(c) < N:
            out_L[s:s + len(c)] += c * 0.1
            out_R[s:s + len(c)] += c * 0.1

    # Global envelope
    env = envelope(N, attack=1.0, release=3.0)
    out_L *= env
    out_R *= env

    out = np.stack([out_L, out_R], axis=1)
    peak = np.max(np.abs(out))
    if peak > 0:
        out *= 0.85 / peak
    return out


# ── Piece 3: Wavelet Transform ──────────────────────────────────────────────

def wavelet_transform():
    """Chirp signal analyzed by Morlet wavelets at multiple scales."""
    print("Generating Wavelet Transform...")
    duration = 55.0
    N = int(duration * SR)
    t = np.arange(N) / SR
    out_L = np.zeros(N)
    out_R = np.zeros(N)

    # Source signal: chirp from 110 Hz to 880 Hz over the duration
    # This gives the wavelet analysis something to track
    chirp_f0 = 110.0
    chirp_f1 = 880.0
    # Exponential chirp
    chirp_phase = 2 * np.pi * chirp_f0 * duration / np.log(chirp_f1 / chirp_f0) * (
        np.exp(t / duration * np.log(chirp_f1 / chirp_f0)) - 1
    )
    chirp = np.sin(chirp_phase) * 0.3

    # Add the chirp as a quiet center reference
    out_L += chirp * 0.15
    out_R += chirp * 0.15

    # Wavelet analysis: Morlet wavelets at different scales
    # Each scale produces a time-varying coefficient that modulates a tone
    n_scales = 12
    center_freqs = np.geomspace(110, 1760, n_scales)  # A2 to A6

    # For each scale, compute wavelet coefficient magnitude over time
    # using a simplified CWT (sliding inner product with Morlet wavelet)
    hop = int(0.02 * SR)  # 20ms hop
    n_frames = N // hop

    for si, cf in enumerate(center_freqs):
        # Morlet wavelet parameters
        sigma = 5.0 / (2 * np.pi * cf)  # Window width inversely proportional to freq
        wavelet_len = int(min(6 * sigma * SR, N // 4))
        if wavelet_len < 10:
            wavelet_len = 10

        wt = np.arange(wavelet_len) / SR - wavelet_len / SR / 2
        wavelet = np.exp(-wt ** 2 / (2 * sigma ** 2)) * np.cos(2 * np.pi * cf * wt)
        wavelet /= np.sqrt(np.sum(wavelet ** 2) + 1e-9)

        # Compute magnitude of wavelet coefficients
        coeffs = np.convolve(chirp, wavelet, mode='same')
        coeff_mag = np.abs(coeffs)

        # Smooth the coefficient magnitude
        smooth_len = int(0.05 * SR)
        if smooth_len > 1:
            kernel = np.ones(smooth_len) / smooth_len
            coeff_mag = np.convolve(coeff_mag, kernel, mode='same')

        # Normalize
        peak = np.max(coeff_mag)
        if peak > 0:
            coeff_mag /= peak

        # Synthesize: each scale generates a tone at its center frequency
        # modulated by the wavelet coefficient magnitude
        # Pan: low frequencies left, high frequencies right
        pan = si / (n_scales - 1)

        # Tone for this scale
        tone = sine(cf, t) * coeff_mag * 0.25

        # Add harmonics based on coefficient strength (richer when strong)
        tone += sine(cf * 2, t) * coeff_mag ** 2 * 0.08
        tone += sine(cf * 3, t) * coeff_mag ** 3 * 0.04

        # Apply scale-specific vibrato (wider for lower scales)
        vib_rate = 4.0 + si * 0.5
        vib_depth = (0.003 - si * 0.0002) * cf
        vib = sine(cf + vib_depth * np.sin(2 * np.pi * vib_rate * t), t)
        tone *= (0.8 + 0.2 * np.abs(vib))

        out_L += tone * (1 - pan)
        out_R += tone * pan

    # 55Hz drone
    drone = sine(55.0, t) * 0.05 * envelope(N, attack=2.0, release=3.0)
    out_L += drone
    out_R += drone

    # Scale marker clicks at the start: identify each scale with a brief ping
    for si, cf in enumerate(center_freqs):
        marker_t = 1.0 + si * 0.4  # stagger over first 6 seconds
        s = int(marker_t * SR)
        ping_len = int(0.1 * SR)
        if s + ping_len < N:
            ping_t = np.arange(ping_len) / SR
            ping = np.sin(2 * np.pi * cf * ping_t) * np.exp(-ping_t * 30) * 0.2
            pan = si / (n_scales - 1)
            out_L[s:s + ping_len] += ping * (1 - pan)
            out_R[s:s + ping_len] += ping * pan

    # Global envelope
    env = envelope(N, attack=1.5, release=4.0)
    out_L *= env
    out_R *= env

    out = np.stack([out_L, out_R], axis=1)
    peak = np.max(np.abs(out))
    if peak > 0:
        out *= 0.85 / peak
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    pieces = [
        ("sig_1_dft_decomposition", dft_decomposition),
        ("sig_2_convolution", convolution),
        ("sig_3_wavelet_transform", wavelet_transform),
    ]
    for name, fn in pieces:
        data = fn()
        write_wav(f"output/{name}.wav", data)
    print("\nDone! All signal processing pieces generated.")
