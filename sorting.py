"""
Sorting — The Sound of Order Emerging from Chaos

Three pieces exploring how different algorithms impose order,
each with a radically different personality:

1. Bubble Sort Meditation (50s, stereo)
   The most patient algorithm. Adjacent swaps ripple through like
   waves on a pond. Each pass, the largest unsorted element "bubbles"
   to its place — audible as a descending pitch settling at the top.
   Increasingly consonant as order emerges.

2. Quicksort Drama (45s, stereo)
   Recursive partitioning as musical drama. The pivot divides the
   stereo field — elements less than pivot drift left, greater drift
   right. Each recursion level is a frequency band. The stack depth
   becomes reverb depth. Dramatic because partition sizes are unequal.

3. Merge Sort Counterpoint (50s, stereo)
   Bottom-up merging as two-voice counterpoint. Pairs merge into
   fours merge into eights — each merge is two melodic lines
   converging into one. The doubling structure creates a natural
   crescendo. Bach would approve.
"""

import numpy as np
import wave
import os

SAMPLE_RATE = 44100
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_wav(filename, samples, sr=SAMPLE_RATE):
    samples = samples / (np.max(np.abs(samples)) + 1e-8) * 0.85
    samples_int = (samples * 32767).astype(np.int16)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples_int.tobytes())


def save_wav_stereo(filename, left, right, sr=SAMPLE_RATE):
    mx = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-8)
    left = (left / mx * 0.85 * 32767).astype(np.int16)
    right = (right / mx * 0.85 * 32767).astype(np.int16)
    stereo = np.column_stack((left, right)).flatten()
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())


# ── Tone generation ──

def sine(freq, duration, sr=SAMPLE_RATE):
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t)


def tone_with_harmonics(freq, duration, n_harmonics=4, decay=0.6, sr=SAMPLE_RATE):
    """A richer tone with decaying harmonics."""
    t = np.arange(int(sr * duration)) / sr
    sig = np.zeros_like(t)
    for h in range(1, n_harmonics + 1):
        sig += (decay ** (h - 1)) * np.sin(2 * np.pi * freq * h * t)
    return sig


def apply_envelope(sig, attack=0.01, release=0.05, sr=SAMPLE_RATE):
    """Simple AR envelope."""
    n = len(sig)
    env = np.ones(n)
    att_samples = int(attack * sr)
    rel_samples = int(release * sr)
    if att_samples > 0:
        env[:att_samples] = np.linspace(0, 1, att_samples)
    if rel_samples > 0:
        env[-rel_samples:] = np.linspace(1, 0, rel_samples)
    return sig * env


def value_to_freq(val, min_val, max_val, min_freq=130.0, max_freq=1200.0):
    """Map array value to frequency (log scale)."""
    if max_val == min_val:
        return (min_freq + max_freq) / 2
    t = (val - min_val) / (max_val - min_val)
    return min_freq * (max_freq / min_freq) ** t


def value_to_pan(val, min_val, max_val):
    """Map value to stereo position 0..1 (0=left, 1=right)."""
    if max_val == min_val:
        return 0.5
    return (val - min_val) / (max_val - min_val)


# ── Sorting trackers ──

def bubble_sort_trace(arr):
    """Bubble sort returning list of (swap_i, swap_j, array_snapshot)."""
    a = arr.copy()
    trace = []
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                trace.append((j, j + 1, a.copy()))
                swapped = True
        if not swapped:
            break
    return trace


def quicksort_trace(arr):
    """Quicksort returning list of (pivot_val, partition_left, partition_right, depth, array_snapshot)."""
    a = arr.copy()
    trace = []

    def qs(lo, hi, depth):
        if lo >= hi:
            return
        pivot = a[hi]
        i = lo
        for j in range(lo, hi):
            if a[j] <= pivot:
                a[i], a[j] = a[j], a[i]
                i += 1
        a[i], a[hi] = a[hi], a[i]
        trace.append((pivot, lo, hi, depth, a.copy()))
        qs(lo, i - 1, depth + 1)
        qs(i + 1, hi, depth + 1)

    qs(0, len(a) - 1, 0)
    return trace


def merge_sort_trace(arr):
    """Bottom-up merge sort returning list of (merge_start, merge_mid, merge_end, array_snapshot)."""
    a = arr.copy()
    n = len(a)
    trace = []
    width = 1
    while width < n:
        for i in range(0, n, 2 * width):
            lo = i
            mid = min(i + width, n)
            hi = min(i + 2 * width, n)
            if mid < hi:
                merged = []
                l, r = lo, mid
                while l < mid and r < hi:
                    if a[l] <= a[r]:
                        merged.append(a[l]); l += 1
                    else:
                        merged.append(a[r]); r += 1
                merged.extend(a[l:mid])
                merged.extend(a[r:hi])
                a[lo:hi] = merged
                trace.append((lo, mid, hi, a.copy()))
        width *= 2
    return trace


# ── Piece 1: Bubble Sort Meditation ──

def bubble_sort_meditation():
    print("  Generating Bubble Sort Meditation...")
    duration = 50.0
    n_elements = 32
    np.random.seed(42)
    arr = np.random.permutation(n_elements).astype(float)
    min_v, max_v = 0, n_elements - 1

    trace = bubble_sort_trace(arr)
    n_swaps = len(trace)
    time_per_swap = duration / (n_swaps + 1)

    total_samples = int(duration * SAMPLE_RATE)
    left = np.zeros(total_samples)
    right = np.zeros(total_samples)

    # Each swap produces a two-note figure: the two swapped values
    note_dur = min(time_per_swap * 0.8, 0.15)

    for idx, (i, j, snapshot) in enumerate(trace):
        t_start = idx * time_per_swap
        sample_start = int(t_start * SAMPLE_RATE)

        # Swapped values (after swap, so snapshot[i] was moved from j, etc.)
        val_a = snapshot[i]
        val_b = snapshot[j]

        freq_a = value_to_freq(val_a, min_v, max_v, 200, 1000)
        freq_b = value_to_freq(val_b, min_v, max_v, 200, 1000)

        # Position in array → stereo
        pan_a = i / (n_elements - 1)
        pan_b = j / (n_elements - 1)

        # Progress → more harmonics (sound becomes richer as order emerges)
        progress = idx / n_swaps
        n_harm = 2 + int(progress * 4)

        for freq, pan in [(freq_a, pan_a), (freq_b, pan_b)]:
            note = tone_with_harmonics(freq, note_dur, n_harmonics=n_harm, decay=0.5)
            note = apply_envelope(note, attack=0.005, release=note_dur * 0.3)
            end = min(sample_start + len(note), total_samples)
            seg_len = end - sample_start
            if seg_len > 0:
                left[sample_start:end] += note[:seg_len] * (1 - pan)
                right[sample_start:end] += note[:seg_len] * pan

    # Final chord: the sorted array as a sustained harmony
    chord_start = int((duration - 4.0) * SAMPLE_RATE)
    chord_dur = 3.5
    # Pick 8 evenly spaced values from sorted array for chord
    sorted_arr = np.arange(n_elements, dtype=float)
    chord_indices = np.linspace(0, n_elements - 1, 8).astype(int)
    for ci in chord_indices:
        freq = value_to_freq(sorted_arr[ci], min_v, max_v, 200, 1000)
        note = tone_with_harmonics(freq, chord_dur, n_harmonics=6, decay=0.4)
        note = apply_envelope(note, attack=0.3, release=1.5)
        pan = ci / (n_elements - 1)
        end = min(chord_start + len(note), total_samples)
        seg_len = end - chord_start
        if seg_len > 0:
            left[chord_start:end] += note[:seg_len] * (1 - pan) * 0.3
            right[chord_start:end] += note[:seg_len] * pan * 0.3

    save_wav_stereo(os.path.join(OUTPUT_DIR, "sort_1_bubble_meditation.wav"), left, right)
    print("    ✓ sort_1_bubble_meditation.wav")


# ── Piece 2: Quicksort Drama ──

def quicksort_drama():
    print("  Generating Quicksort Drama...")
    duration = 45.0
    n_elements = 48
    np.random.seed(7)
    arr = np.random.permutation(n_elements).astype(float)
    min_v, max_v = 0, n_elements - 1

    trace = quicksort_trace(arr)
    n_events = len(trace)
    time_per_event = duration / (n_events + 1)

    total_samples = int(duration * SAMPLE_RATE)
    left = np.zeros(total_samples)
    right = np.zeros(total_samples)

    max_depth = max(t[3] for t in trace) if trace else 1

    for idx, (pivot_val, lo, hi, depth, snapshot) in enumerate(trace):
        t_start = idx * time_per_event
        sample_start = int(t_start * SAMPLE_RATE)

        # Pivot tone — central, sustained
        pivot_freq = value_to_freq(pivot_val, min_v, max_v, 150, 1200)
        pivot_dur = min(time_per_event * 0.7, 0.5)

        # Depth → affects timbre (deeper = more harmonics, like going into detail)
        depth_ratio = depth / (max_depth + 1)
        n_harm = 3 + int(depth_ratio * 5)

        # Partition size → loudness
        part_size = hi - lo + 1
        loudness = 0.3 + 0.7 * (part_size / n_elements)

        # Pivot tone centered in the partition's stereo range
        center_pan = ((lo + hi) / 2) / (n_elements - 1)

        pivot_tone = tone_with_harmonics(pivot_freq, pivot_dur, n_harmonics=n_harm, decay=0.45)
        pivot_tone = apply_envelope(pivot_tone, attack=0.01, release=pivot_dur * 0.4) * loudness

        end = min(sample_start + len(pivot_tone), total_samples)
        seg_len = end - sample_start
        if seg_len > 0:
            left[sample_start:end] += pivot_tone[:seg_len] * (1 - center_pan)
            right[sample_start:end] += pivot_tone[:seg_len] * center_pan

        # Scatter tones for elements in partition (quick arpeggio)
        scatter_dur = min(time_per_event * 0.3, 0.08)
        for k in range(lo, min(hi + 1, lo + 12)):  # cap to avoid too many
            val = snapshot[k]
            freq = value_to_freq(val, min_v, max_v, 150, 1200)
            pan = k / (n_elements - 1)
            t_offset = (k - lo) * scatter_dur * 0.3
            s_start = int((t_start + pivot_dur * 0.2 + t_offset) * SAMPLE_RATE)

            grain = sine(freq, scatter_dur)
            grain = apply_envelope(grain, attack=0.003, release=scatter_dur * 0.5) * 0.15

            s_end = min(s_start + len(grain), total_samples)
            s_len = s_end - s_start
            if s_len > 0 and s_start >= 0:
                left[s_start:s_end] += grain[:s_len] * (1 - pan)
                right[s_start:s_end] += grain[:s_len] * pan

    save_wav_stereo(os.path.join(OUTPUT_DIR, "sort_2_quicksort_drama.wav"), left, right)
    print("    ✓ sort_2_quicksort_drama.wav")


# ── Piece 3: Merge Sort Counterpoint ──

def merge_sort_counterpoint():
    print("  Generating Merge Sort Counterpoint...")
    duration = 50.0
    n_elements = 32
    np.random.seed(13)
    arr = np.random.permutation(n_elements).astype(float)
    min_v, max_v = 0, n_elements - 1

    trace = merge_sort_trace(arr)
    n_merges = len(trace)
    time_per_merge = duration / (n_merges + 1)

    total_samples = int(duration * SAMPLE_RATE)
    left = np.zeros(total_samples)
    right = np.zeros(total_samples)

    # Group merges by level (width doubling)
    # Each merge event has (lo, mid, hi, snapshot)
    # Width can be inferred from (mid - lo)
    max_width = max(t[1] - t[0] for t in trace) if trace else 1

    for idx, (lo, mid, hi, snapshot) in enumerate(trace):
        t_start = idx * time_per_merge
        merge_width = mid - lo  # current merge level

        # Level → base note duration and timbre
        level = np.log2(max(merge_width, 1)) + 1
        max_level = np.log2(max(max_width, 1)) + 1
        level_ratio = level / max_level

        note_dur = 0.08 + level_ratio * 0.25  # wider merges get longer notes

        # Left half melody (lo..mid)
        for k in range(lo, mid):
            val = snapshot[k]
            freq = value_to_freq(val, min_v, max_v, 180, 900)
            t_note = t_start + (k - lo) * note_dur * 0.4
            s_start = int(t_note * SAMPLE_RATE)

            # Left half → slightly left pan, warm timbre
            n_harm = 3 + int(level_ratio * 3)
            note = tone_with_harmonics(freq, note_dur, n_harmonics=n_harm, decay=0.5)
            note = apply_envelope(note, attack=0.01, release=note_dur * 0.4) * 0.25

            # Pan: left-ish, position within merge range
            pan = 0.2 + 0.2 * ((k - lo) / max(mid - lo - 1, 1))

            s_end = min(s_start + len(note), total_samples)
            s_len = s_end - s_start
            if s_len > 0 and s_start >= 0:
                left[s_start:s_end] += note[:s_len] * (1 - pan)
                right[s_start:s_end] += note[:s_len] * pan

        # Right half melody (mid..hi)
        for k in range(mid, hi):
            val = snapshot[k]
            freq = value_to_freq(val, min_v, max_v, 180, 900)
            t_note = t_start + (k - mid) * note_dur * 0.4
            s_start = int(t_note * SAMPLE_RATE)

            n_harm = 3 + int(level_ratio * 3)
            note = tone_with_harmonics(freq, note_dur, n_harmonics=n_harm, decay=0.5)
            note = apply_envelope(note, attack=0.01, release=note_dur * 0.4) * 0.25

            # Right-ish pan
            pan = 0.6 + 0.2 * ((k - mid) / max(hi - mid - 1, 1))

            s_end = min(s_start + len(note), total_samples)
            s_len = s_end - s_start
            if s_len > 0 and s_start >= 0:
                left[s_start:s_end] += note[:s_len] * (1 - pan)
                right[s_start:s_end] += note[:s_len] * pan

        # Merged result: brief chord showing the combined sorted subsequence
        chord_t = t_start + max(mid - lo, hi - mid) * note_dur * 0.4 + 0.05
        chord_s = int(chord_t * SAMPLE_RATE)
        chord_dur = note_dur * 1.5

        # Pick up to 6 notes from merged range for chord
        merged_vals = snapshot[lo:hi]
        step = max(1, len(merged_vals) // 6)
        chord_vals = merged_vals[::step][:6]
        for cv in chord_vals:
            freq = value_to_freq(cv, min_v, max_v, 180, 900)
            note = tone_with_harmonics(freq, chord_dur, n_harmonics=4, decay=0.4)
            note = apply_envelope(note, attack=0.02, release=chord_dur * 0.5) * 0.12
            center_pan = ((lo + hi) / 2) / (n_elements - 1)
            c_end = min(chord_s + len(note), total_samples)
            c_len = c_end - chord_s
            if c_len > 0 and chord_s >= 0:
                left[chord_s:c_end] += note[:c_len] * (1 - center_pan)
                right[chord_s:c_end] += note[:c_len] * center_pan

    # Final: fully sorted array as ascending scale
    scale_start_t = duration - 5.0
    scale_note_dur = 0.3
    sorted_arr = np.sort(arr)
    # Play every 4th element as a clean ascending scale
    scale_vals = sorted_arr[::4]
    for si, val in enumerate(scale_vals):
        freq = value_to_freq(val, min_v, max_v, 180, 900)
        t_note = scale_start_t + si * 0.4
        s_start = int(t_note * SAMPLE_RATE)
        note = tone_with_harmonics(freq, scale_note_dur, n_harmonics=6, decay=0.35)
        note = apply_envelope(note, attack=0.02, release=0.15) * 0.3
        pan = si / max(len(scale_vals) - 1, 1)
        s_end = min(s_start + len(note), total_samples)
        s_len = s_end - s_start
        if s_len > 0 and s_start >= 0:
            left[s_start:s_end] += note[:s_len] * (1 - pan)
            right[s_start:s_end] += note[:s_len] * pan

    save_wav_stereo(os.path.join(OUTPUT_DIR, "sort_3_merge_counterpoint.wav"), left, right)
    print("    ✓ sort_3_merge_counterpoint.wav")


# ── Main ──

if __name__ == "__main__":
    print("Phase 7: Sorting — The Sound of Order Emerging from Chaos")
    bubble_sort_meditation()
    quicksort_drama()
    merge_sort_counterpoint()
    print("Done.")
