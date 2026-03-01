"""
Render Halting Music visualizations as video frames, pipe to ffmpeg with audio.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess, os, sys, math

SAMPLE_RATE = 44100
FPS = 30
WIDTH, HEIGHT = 1280, 720
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

# Colors
BG = (10, 10, 15)
WARM = (255, 107, 53)
COOL = (74, 111, 165)
WHITE = (224, 224, 224)
DIM = (80, 80, 80)
DIMMER = (40, 40, 40)

PALETTE = [
    (255,107,53),(247,197,159),(239,239,208),(137,176,174),(74,111,165),
    (230,57,70),(69,123,157),(168,218,220),(244,162,97),(42,157,143),
    (231,111,81),(38,70,83),(233,196,106),(96,108,56),(221,161,94),
    (188,108,37),(142,202,230)
]

def collatz(n):
    seq = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        seq.append(n)
    return seq

def alpha_blend(bg, fg, alpha):
    return tuple(int(b * (1-alpha) + f * alpha) for b, f in zip(bg, fg))

def lerp(a, b, t):
    return a + (b - a) * t

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
    FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 9)
    FONT_LG = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
except:
    FONT = FONT_SM = FONT_LG = ImageFont.load_default()

def pipe_to_ffmpeg(audio_path, output_path, duration):
    n_frames = int(duration * FPS)
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{WIDTH}x{HEIGHT}', '-pix_fmt', 'rgb24', '-r', str(FPS),
        '-i', '-',
        '-i', audio_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
        '-c:a', 'aac', '-b:a', '192k',
        '-pix_fmt', 'yuv420p',
        '-shortest',
        output_path
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

# ═══════════════ I. ENSEMBLE ═══════════════
def render_ensemble():
    starters = [3,5,7,10,16,27,54,73,97,113,327,649,871,1161,6171,77031,837799]
    seqs_raw = [(n, collatz(n)) for n in starters]
    seqs_raw.sort(key=lambda x: len(x[1]))
    
    # Precompute normalized log values
    seqs = []
    for n, seq in seqs_raw:
        log_seq = [math.log2(max(v, 1)) for v in seq]
        log_max = max(log_seq) or 1
        norm = [v / log_max for v in log_seq]
        seqs.append({'n': n, 'seq': seq, 'norm': norm, 'len': len(seq)})
    
    global_max_steps = seqs[-1]['len']
    duration = 40.0
    audio_path = os.path.join(OUTDIR, 'halting_1_ensemble_stereo.wav')
    video_path = os.path.join(OUTDIR, 'halting_1_ensemble.mp4')
    
    print(f"  Rendering {int(duration*FPS)} frames...")
    proc = pipe_to_ffmpeg(audio_path, video_path, duration)
    
    margin = {'top': 50, 'bottom': 70, 'left': 50, 'right': 30}
    plot_w = WIDTH - margin['left'] - margin['right']
    plot_h = HEIGHT - margin['top'] - margin['bottom']
    N = len(seqs)
    lane_h = plot_h / N
    entry_window = 0.6
    
    for frame_i in range(int(duration * FPS)):
        t = frame_i / FPS
        progress = t / duration
        
        img = Image.new('RGB', (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(img)
        
        # Grid lines
        for s in range(0, global_max_steps + 1, 50):
            x = margin['left'] + plot_w * s / global_max_steps
            for y in range(margin['top'], HEIGHT - margin['bottom'], 4):
                draw.point((int(x), y), fill=DIMMER)
        
        # Title
        draw.text((margin['left'], 10), "I. Collatz Ensemble", fill=DIM, font=FONT_LG)
        
        for idx, item in enumerate(seqs):
            entry_frac = idx / N
            entry_time = entry_frac * entry_window
            if progress < entry_time:
                continue
            
            local_progress = min((progress - entry_time) / (1 - entry_time) * 1.5, 1.0)
            max_steps = min(int(local_progress * item['len']), item['len'])
            
            lane_y = margin['top'] + idx * lane_h
            lane_center = lane_y + lane_h / 2
            amplitude = lane_h * 0.42
            
            color = PALETTE[idx % len(PALETTE)]
            
            # Label
            draw.text((4, int(lane_center) - 4), str(item['n']), fill=DIMMER, font=FONT_SM)
            
            # Draw trace
            points = []
            for i in range(max_steps):
                x = margin['left'] + plot_w * (i / global_max_steps)
                y = lane_center + amplitude - item['norm'][i] * amplitude * 2
                points.append((int(x), int(y)))
            
            if len(points) > 1:
                draw.line(points, fill=color, width=1)
            
            # Head dot
            if max_steps > 0 and max_steps <= item['len']:
                ci = max_steps - 1
                cx = int(margin['left'] + plot_w * (ci / global_max_steps))
                cy = int(lane_center + amplitude - item['norm'][ci] * amplitude * 2)
                # Glow
                for r in range(8, 0, -1):
                    a = 0.15 * (1 - r/8)
                    gc = alpha_blend(BG, color, a)
                    draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=gc)
                draw.ellipse((cx-2, cy-2, cx+2, cy+2), fill=color)
        
        # Drone bar at bottom
        drone_alpha = 0.12 + min(progress / entry_window, 1) * 0.2
        bar_w = int(plot_w * progress)
        bar_color = alpha_blend(BG, WARM, drone_alpha)
        draw.rectangle((margin['left'], HEIGHT - margin['bottom'] + 2,
                        margin['left'] + bar_w, HEIGHT - margin['bottom'] + 4), fill=bar_color)
        
        # Bottom text
        draw.text((margin['left'], HEIGHT - 30),
                  f"17 voices · step 0-{global_max_steps} · D1 drone = the unproven conjecture",
                  fill=DIMMER, font=FONT_SM)
        
        proc.stdin.write(np.array(img).tobytes())
        
        if frame_i % 100 == 0:
            print(f"    frame {frame_i}/{int(duration*FPS)}")
    
    proc.stdin.close()
    proc.wait()
    print(f"  ✓ {video_path}")

# ═══════════════ II. JOURNEY ═══════════════
def render_journey():
    seq = collatz(27)
    max_val = max(seq)
    log_max = math.log2(max_val)
    duration_s = len(seq) * 0.22 + 2.0  # match audio
    audio_path = os.path.join(OUTDIR, 'halting_2_journey_27.wav')
    video_path = os.path.join(OUTDIR, 'halting_2_journey.mp4')
    
    # Get actual audio duration
    import wave
    with wave.open(audio_path) as wf:
        duration_s = wf.getnframes() / wf.getframerate()
    
    print(f"  Rendering {int(duration_s*FPS)} frames...")
    proc = pipe_to_ffmpeg(audio_path, video_path, duration_s)
    
    m = {'top': 50, 'bottom': 70, 'left': 60, 'right': 40}
    pw = WIDTH - m['left'] - m['right']
    ph = HEIGHT - m['top'] - m['bottom']
    
    grid_vals = [1, 4, 27, 100, 1000, 9232]
    
    for frame_i in range(int(duration_s * FPS)):
        t = frame_i / FPS
        progress = t / duration_s
        max_steps = min(int(progress * len(seq) * 1.3), len(seq))
        
        img = Image.new('RGB', (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(img)
        
        # Title
        draw.text((m['left'], 10), "II. Single Journey — n=27", fill=DIM, font=FONT_LG)
        
        # Grid
        for v in grid_vals:
            lv = math.log2(max(v, 1))
            y = int(m['top'] + ph * (1 - lv / log_max))
            for x in range(m['left'], WIDTH - m['right'], 6):
                draw.point((x, y), fill=(25, 25, 30))
            draw.text((4, y - 5), f"{v:,}", fill=DIMMER, font=FONT_SM)
        
        # Path with color segments
        for i in range(1, max_steps):
            x0 = int(m['left'] + pw * ((i-1) / (len(seq)-1)))
            x1 = int(m['left'] + pw * (i / (len(seq)-1)))
            y0 = int(m['top'] + ph * (1 - math.log2(max(seq[i-1],1)) / log_max))
            y1 = int(m['top'] + ph * (1 - math.log2(max(seq[i],1)) / log_max))
            is_up = seq[i] > seq[i-1]
            color = WARM if is_up else COOL
            draw.line((x0, y0, x1, y1), fill=color, width=2)
        
        # Peak annotation
        peak_idx = seq.index(max_val)
        if peak_idx < max_steps:
            px = int(m['left'] + pw * (peak_idx / (len(seq)-1)))
            draw.text((px + 6, m['top'] + 4), "↑ 9,232", fill=alpha_blend(BG, WARM, 0.4), font=FONT)
        
        # Head dot
        if 0 < max_steps <= len(seq):
            ci = max_steps - 1
            cx = int(m['left'] + pw * (ci / (len(seq)-1)))
            cy = int(m['top'] + ph * (1 - math.log2(max(seq[ci],1)) / log_max))
            
            pulse = 1 + math.sin(t * 5) * 0.25
            r = int(14 * pulse)
            for rr in range(r, 0, -1):
                a = 0.3 * (1 - rr/r)
                gc = alpha_blend(BG, WARM, a)
                draw.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=gc)
            draw.ellipse((cx-3, cy-3, cx+3, cy+3), fill=(255, 255, 255))
            
            draw.text((cx + 16, cy - 8), f"{seq[ci]:,}  [{ci}]", fill=DIM, font=FONT)
        
        # Bottom legend
        draw.text((m['left'], HEIGHT - 30),
                  f"111 steps · peak 9,232 · step {min(max_steps, len(seq))-1}/{len(seq)-1}",
                  fill=DIMMER, font=FONT_SM)
        # Color legend
        draw.rectangle((m['left'] + 300, HEIGHT - 32, m['left'] + 310, HEIGHT - 24), fill=WARM)
        draw.text((m['left'] + 314, HEIGHT - 30), "3n+1 (up)", fill=DIMMER, font=FONT_SM)
        draw.rectangle((m['left'] + 400, HEIGHT - 32, m['left'] + 410, HEIGHT - 24), fill=COOL)
        draw.text((m['left'] + 414, HEIGHT - 30), "n/2 (down)", fill=DIMMER, font=FONT_SM)
        
        proc.stdin.write(np.array(img).tobytes())
        if frame_i % 100 == 0:
            print(f"    frame {frame_i}/{int(duration_s*FPS)}")
    
    proc.stdin.close()
    proc.wait()
    print(f"  ✓ {video_path}")

# ═══════════════ III. DENSITY ═══════════════
def render_density():
    seqs = [collatz(n) for n in range(100, 600)]
    max_len = max(len(s) for s in seqs)
    N = len(seqs)
    
    # Precompute stats
    stats = []
    for step in range(max_len):
        vals = [s[step] for s in seqs if step < len(s)]
        count = len(vals)
        mean = sum(vals) / count if count else 0
        std = (sum((v - mean)**2 for v in vals) / count) ** 0.5 if count > 1 else 0
        stats.append({'count': count, 'mean': mean, 'std': std})
    
    max_mean = max(s['mean'] for s in stats) or 1
    
    duration = 30.0
    audio_path = os.path.join(OUTDIR, 'halting_3_density.wav')
    video_path = os.path.join(OUTDIR, 'halting_3_density.mp4')
    
    print(f"  Rendering {int(duration*FPS)} frames...")
    proc = pipe_to_ffmpeg(audio_path, video_path, duration)
    
    m = {'top': 50, 'bottom': 70, 'left': 50, 'right': 50}
    pw = WIDTH - m['left'] - m['right']
    ph = HEIGHT - m['top'] - m['bottom']
    max_log = 18  # display cap
    
    # Previous frame buffer for trail effect
    prev_img = None
    
    for frame_i in range(int(duration * FPS)):
        t = frame_i / FPS
        progress = t / duration
        current_step = min(int(progress * max_len), max_len - 1)
        stat = stats[current_step]
        density = stat['count'] / N
        
        # Trail: blend with previous frame
        img = Image.new('RGB', (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(img)
        
        if prev_img is not None:
            # Darken previous
            prev_arr = np.array(prev_img, dtype=np.float32)
            prev_arr *= 0.88
            bg_arr = np.array(img, dtype=np.float32)
            blended = np.clip(np.maximum(prev_arr, bg_arr), 0, 255).astype(np.uint8)
            img = Image.fromarray(blended)
            draw = ImageDraw.Draw(img)
        
        # Title
        draw.text((m['left'], 10), "III. Density", fill=DIM, font=FONT_LG)
        
        # Particles
        for i in range(0, N, 1):
            seq = seqs[i]
            if current_step >= len(seq):
                continue
            val = seq[current_step]
            log_val = math.log2(max(val, 1))
            y = int(m['top'] + ph * (1 - min(log_val / max_log, 1)))
            x = int(m['left'] + pw * (i / N))
            
            is_above = val > stat['mean']
            color = WARM if is_above else COOL
            alpha = 0.3 + density * 0.5
            pc = alpha_blend(BG, color, min(alpha, 1))
            draw.rectangle((x, y, x+2, y+2), fill=pc)
        
        # Mean line
        if stat['mean'] > 0:
            mean_y = int(m['top'] + ph * (1 - min(math.log2(max(stat['mean'], 1)) / max_log, 1)))
            line_alpha = 0.08 + density * 0.2
            lc = alpha_blend(BG, WHITE, min(line_alpha, 1))
            draw.line((m['left'], mean_y, WIDTH - m['right'], mean_y), fill=lc, width=1)
        
        # Stats
        sx, sy = WIDTH - 170, 55
        draw.text((sx, sy), f"active  {stat['count']}/{N}", fill=DIM, font=FONT)
        draw.text((sx, sy+16), f"mean    {stat['mean']:.0f}", fill=DIM, font=FONT)
        draw.text((sx, sy+32), f"σ       {stat['std']:.0f}", fill=DIM, font=FONT)
        draw.text((sx, sy+48), f"step    {current_step}/{max_len}", fill=DIM, font=FONT)
        
        # Progress bar
        bar_color = alpha_blend(BG, WARM, 0.2)
        draw.rectangle((0, HEIGHT-3, int(WIDTH*progress), HEIGHT), fill=bar_color)
        
        # Bottom
        draw.text((m['left'], HEIGHT - 30),
                  f"500 sequences (n=100..599) as particle cloud · noise: logistic map r=3.99",
                  fill=DIMMER, font=FONT_SM)
        
        prev_img = img.copy()
        proc.stdin.write(np.array(img).tobytes())
        if frame_i % 100 == 0:
            print(f"    frame {frame_i}/{int(duration*FPS)}")
    
    proc.stdin.close()
    proc.wait()
    print(f"  ✓ {video_path}")


if __name__ == '__main__':
    print("=== Halting Music — Video Render ===\n")
    
    print("I. Collatz Ensemble")
    render_ensemble()
    
    print("\nII. Single Journey — n=27")
    render_journey()
    
    print("\nIII. Density")
    render_density()
    
    print("\n=== Done ===")
