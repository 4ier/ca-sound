#!/usr/bin/env python3
"""Generate the album page (docs/index.html) from output/ directory and track metadata."""
import os, json, html, wave

REPO = "4ier/ca-sound"
TAG = os.environ.get("TAG", "latest")

# Track metadata: keyed by filename prefix → (series, title, note, order_in_series)
# Series order: higher number = newer, shown first (新作品在最上面)
SERIES = {
    "sig":     (240, "Signal Processing",         "傅里叶分解的频谱剥离、卷积核的音色重塑、小波变换的多尺度追踪。信号处理——用信号处理来声化信号处理。"),
    "comp":    (230, "Compiler Pipeline",        "词法分析的 token 流、语法分析的 AST 生长、代码生成的机器节奏。源码的变形记。"),
    "game":    (220, "Game Theory",              "囚徒困境的合作与背叛、鹰鸽博弈的种群演化、纳什均衡的收敛。博弈论——理性的和声。"),
    "dist":    (210, "Distributed Systems",     "Raft 共识的心跳与分裂、Gossip 协议的指数蔓延、向量时钟的因果多声部。分布式系统——协调即和声。"),
    "cat":     (200, "Category Theory",         "函子的结构保持、自然变换的路径无关、单子的不确定性链。范畴论——数学的数学。"),
    "type":    (190, "Type Systems",           "类型推断的和声收敛、子类型格的层级共鸣、Curry-Howard 的证明即程序。类型不是约束——是结构。"),
    "nn":      (180, "Neural Networks",        "反向传播的梯度回声、激活函数的音色变形、权重空间的收敛之旅。学习即是趋向和谐。"),
    "qc":      (170, "Quantum Computing",     "叠加态的和声、纠缠的立体声关联、量子行走的干涉条纹。量子世界不是比喻——它本身就是波。"),
    "opt":     (160, "Optimization",          "梯度下降的耐心寻路、模拟退火从混沌到结晶、遗传算法的群体进化。最优解的声音肖像。"),
    "crypto":  (150, "Cryptography",         "一次性密码本的溶解、雪崩效应的突变、Diffie-Hellman 的秘密收敛。数学守护隐私。"),
    "conc":    (140, "Concurrency",          "死锁的冻结、竞态的漂移、互斥的心跳。并发不是混乱——是受约束的多声部。"),
    "auto":    (130, "Automata & Languages", "Chomsky 层级的声学肖像——有限自动机的判定、文法的分形生长、下推自动机的栈深记忆。"),
    "nt":      (120, "Number Theory",       "素数、筛法、模运算。数论的节奏藏在余数里。"),
    "info":    (110, "Information Theory",  "熵、编码、压缩。信息的骨架如何发声？"),
    "graph":   (100, "Graph Algorithms",    "同一张图，三种遍历人格。Dijkstra 的耐心、BFS 的波浪、DFS 的执念。"),
    "sort":    (90, "Sorting Algorithms", "同一个问题，三种算法人格。秩序从混沌中涌现，但路径截然不同。"),
    "raw":     (80, "CA Raw",          "没有音阶，没有和弦，没有 BPM。自动机本身就是音乐。"),
    "taste":   (70, "Fourier's Taste", "相变临界点、自指怪圈、离散与连续的间隙。"),
    "halting": (60, "Halting Music",   "Collatz 猜想：简单的规则，无人能证明的终点。"),
    "beaver":  (50, "Busy Beaver",     "图灵机在有限步内能写下多少个 1？"),
    "lambda":  (40, "Lambda Calculus",  "函数即声音。应用即泛音。归约即沉默。"),
    "fractal": (30, "Fractals",        "逃逸、变形、无限缩放。复平面的声音地图。"),
    "gol":     (20, "Game of Life",    "Conway 的零玩家游戏。涌现、振荡、永生。"),
    "rule":    (10, "CA Compose",      "用人类音乐理论约束 CA 的早期实验。"),
}

TRACKS = {
    # Compiler Pipeline
    "sig_1_dft_decomposition":         ("sig", "DFT Decomposition",           "复杂音色被逐一拆解为傅里叶分量——谐波从残留中抽出，按频率散布在立体声场。原始声音溶解为光谱。"),
    "sig_2_convolution":               ("sig", "Convolution",                 "脉冲序列遭遇四种卷积核：恒等→低通模糊→共振带通→梳状回声。干声左、湿声右，卷积核即音色。"),
    "sig_3_wavelet_transform":         ("sig", "Wavelet Transform",           "啁啾信号被 12 阶 Morlet 小波分解——低频小波=左声道缓慢脉动，高频小波=右声道细碎微光。时频结构显现。"),
    "comp_1_lexer":                    ("comp", "Lexer",                    "源码字符流化为 token——关键字=铜管 FM，标识符=纯正弦，运算符=打击 click。从左到右的立体声扫描。"),
    "comp_2_parser":                   ("comp", "Parser",                   "token 流构建 AST——递归下降的八度下移，非终结符包裹终结符，树自底向上生长为和声塔。"),
    "comp_3_codegen":                  ("comp", "Code Generation",          "AST 溶解为机器指令序列——LOAD/STORE/ADD/MUL/JMP 各有音色，寄存器分配即声部分配，4Hz 时钟脉冲贯穿。"),
    # Game Theory
    "game_1_prisoners_dilemma":        ("game", "Prisoner's Dilemma",        "5 种经典策略的循环锦标赛——合作=协和音程，背叛=三全音失谐，以牙还牙的胜出化为最终和弦。"),
    "game_2_evolutionary_ess":         ("game", "Evolutionary Stable Strategy", "鹰与鸽的种群博弈——鹰鹰搏斗的代价、鸽鸽分享的柔和、种群比例向 ESS 均衡点漂移。"),
    "game_3_nash_equilibrium":         ("game", "Nash Equilibrium",          "两个玩家在 5×5 收益矩阵中寻路——最佳回应动态逐步收敛，纳什均衡=完美五度的 C4+G4。"),
    # Distributed Systems
    "dist_1_raft_consensus":           ("dist", "Raft Consensus",           "5 节点集群的选举、心跳、网络分区与脑裂——两个 leader 共存的不谐和，最终愈合为统一心跳。"),
    "dist_2_gossip_protocol":          ("dist", "Gossip Protocol",          "7 节点环形拓扑的流行病传播——一个谣言指数蔓延，S 曲线收敛为全体合唱。"),
    "dist_3_vector_clocks":            ("dist", "Vector Clocks",            "4 个进程各自的时钟频率形成复合节奏——消息传递同步时钟，因果链成为旋律序列。"),
    # Category Theory
    "cat_1_functor":                   ("cat", "Functor",                   "两个范畴之间的结构保持映射——C 的旋律在 D 中重生，形态改变但关系不变。"),
    "cat_2_natural_transformation":    ("cat", "Natural Transformation",    "函子之间的态射——F 到 G 的每个分量平滑变形，自然性方块的两条路径殊途同归。"),
    "cat_3_monad":                     ("cat", "Monad",                     "Maybe 单子：纯净音符被不确定性包裹，Kleisli 链中 Nothing 吞噬一切，Join 将嵌套坍缩。"),
    # Type Systems
    "type_1_hindley_milner":           ("type", "Hindley-Milner Inference",   "5 个类型变量从不确定的频率漂移中逐一被约束统一——推断即收敛，principal type 即终和弦。"),
    "type_2_subtype_lattice":          ("type", "Subtype Lattice",           "从 Top（所有泛音）到 Bottom（沉默），子类型继承父类的谐波 DNA，添加自己的音色人格。"),
    "type_3_curry_howard":             ("type", "Curry-Howard",              "合取=双证明和弦，蕴涵=FM 变换传递，析取=双路径收敛。证明完成时和声解决。"),
    # Neural Networks
    "nn_1_backpropagation":            ("nn", "Backpropagation",          "前向传播逐层叠加泛音，反向传播以 FM 回声传递梯度。损失函数从三全音收敛为完美五度。"),
    "nn_2_activation_functions":       ("nn", "Activation Functions",     "同一输入信号经过 sigmoid/ReLU/tanh/softmax 四种变形——压缩、截断、饱和、概率竞争。"),
    "nn_3_weight_space":              ("nn", "Weight Space",              "200 步训练轨迹穿越损失地形——随机初始化的噪声云、鞍点的振荡、局部最小的错误和弦、最终收敛为 D minor。"),
    # Quantum Computing
    "qc_1_superposition":          ("qc", "Superposition",          "|0⟩=220Hz, |1⟩=330Hz——完美五度。Hadamard 门让两个频率共存，测量让一个永远消失。"),
    "qc_2_entanglement":           ("qc", "Entanglement",           "Bell 态：左右声道完美关联。测量一个，另一个瞬间坍缩——距离无关。"),
    "qc_3_quantum_walk":           ("qc", "Quantum Walk",           "左耳经典随机游走（高斯弥散），右耳量子游走（干涉尖峰）。同一条路，两种物理。"),
    # Optimization
    "opt_1_gradient_descent":      ("opt", "Gradient Descent",      "8 个粒子在 Rastrigin 地形上寻路——被困局部最小的持续失谐，找到全局最优的收敛为纯净和弦。"),
    "opt_2_simulated_annealing":   ("opt", "Simulated Annealing",   "高温时频率狂跳，冷却中逐渐收窄，偶尔的上坡接受如最后的叛逆。结晶为 A major。"),
    "opt_3_genetic_algorithm":     ("opt", "Genetic Algorithm",     "20 个有机体向目标旋律进化——初始混沌，交叉重组，突变火花，最终群体齐声唱出 D minor 五声音阶。"),
    # Cryptography
    "crypto_1_one_time_pad":       ("crypto", "One-Time Pad",       "明文旋律被随机密钥 XOR 溶解——从清晰到噪声，信息论的完美保密。"),
    "crypto_2_hash_avalanche":     ("crypto", "Hash Avalanche",     "翻转一个 bit，SHA-256 输出面目全非。左声道=原始，右声道=雪崩后的异世界。"),
    "crypto_3_diffie_hellman":     ("crypto", "Diffie-Hellman",     "Alice 和 Bob 各持私密旋律，经公开通道交换后收敛为同一和弦——共享秘密。"),
    # Concurrency
    "conc_1_dining_philosophers":    ("conc", "Dining Philosophers",    "5 位哲学家争夺叉子——进食的丰满和声、饥饿的颤音、死锁的冻结沉默。"),
    "conc_2_race_condition":         ("conc", "Race Condition",         "两个线程竞争共享变量——冲突时音高漂移，失谐是错误的声音。"),
    "conc_3_mutex_heartbeats":       ("conc", "Mutex Heartbeats",       "互斥锁下的独奏轮转——等待者的心跳脉冲，持锁者的丰满声线。"),
    # Automata & Languages
    "auto_1_finite_automaton":       ("auto", "Finite Automaton",        "DFA 判定二进制可被 3 整除——接受态和弦明亮，拒绝态嗡鸣短促。"),
    "auto_2_context_free_grammar":   ("auto", "Context-Free Grammar",   "L-system 文法的分形生长——深度决定音域，分支决定立体声。"),
    "auto_3_pushdown_automaton":     ("auto", "Pushdown Automaton",     "匹配嵌套括号的下推自动机——push 升高，pop 回落，栈深成低频 drone。"),
    # Number Theory
    "nt_1_sieve":             ("nt", "Sieve of Eratosthenes",  "古老的筛法化为打击乐团——合数被击落，素数持续鸣响。"),
    "nt_2_prime_gaps":        ("nt", "Prime Gaps",             "相邻素数间的不规则间距——孪生素数急促连击，大间隙戏剧性留白。"),
    "nt_3_modular_worlds":    ("nt", "Modular Worlds",         "mod 2,3,5,7,11,13 的残差类交织为复合节奏——中国剩余定理可听化。"),
    # Information Theory
    "info_1_entropy_gradient":     ("info", "Entropy Gradient",        "从纯净到噪声——Shannon 熵作为谐波密度的连续光谱。"),
    "info_2_huffman_tree":         ("info", "Huffman Tree",            "字母频率决定音高与时值。常见即明亮简短，罕见即深沉悠长。"),
    "info_3_lz_window":           ("info", "LZ Window",               "滑动窗口的记忆——新符号是发现，回引是回声。"),
    # Graph Algorithms
    "graph_1_dijkstra_meditation": ("graph", "Dijkstra's Meditation",  "最短路松弛化为和声沉淀——波前如水滴扩散。"),
    "graph_2_bfs_waves":           ("graph", "BFS Waves",              "广度优先的节奏波浪——同层节点齐声共鸣。"),
    "graph_3_dfs_descent":         ("graph", "DFS Descent",            "深度优先的独奏旋律——潜入深处，回溯攀升。"),
    # Game of Life
    "gol_1_glider_symphony":    ("gol", "Glider Symphony",     "滑翔机舰队的交响——周期5的永恒旅行者。"),
    "gol_2_still_life_chorale": ("gol", "Still Life Chorale",  "静物的合唱：不变的模式，持续的和声。"),
    "gol_3_methuselah":         ("gol", "Methuselah",          "从微小初始态爆发的漫长演化。"),
    # Fractals
    "fractal_1_escape_orbits":  ("fractal", "Escape Orbits",      "逃逸轨道：发散点的频率肖像。"),
    "fractal_2_julia_morphs":   ("fractal", "Julia Morphs",       "Julia 集的形变——参数连续漂移。"),
    "fractal_3_mandelbrot_zoom":("fractal", "Mandelbrot Zoom",    "无限缩放。每个尺度都有新的声音。"),
    # Lambda
    "lambda_1_church":          ("lambda", "Church Numerals",     "数字 0→10，每个 n = n 层谐波叠加。"),
    "lambda_2_ycombinator":     ("lambda", "Y Combinator Unfold", "不动点自应用展开为黄金比例频率螺旋。"),
    "lambda_3_ski":             ("lambda", "SKI Calculus",        "S=FM / K=阻尼 / I=正弦，归约轨迹从宽场坍缩到中心。"),
    # Beaver
    "beaver_1_zoo":             ("beaver", "Zoo",                 "6 台图灵机并行。Halters 逐一消失。"),
    "beaver_2_bb2_portrait":    ("beaver", "BB(2) Portrait",      "6 步旅程 → 6 个音乐短语。极简主义。"),
    "beaver_3_wall":            ("beaver", "The Wall",            "BB(4) vs 永不停机者。左右声道的对话。"),
    # Halting
    "halting_1_ensemble_stereo":("halting", "Ensemble",           "多条 Collatz 轨迹的立体声合奏。"),
    "halting_2_journey_27":     ("halting", "Journey of 27",      "数字 27 的 Collatz 旅程——111 步归于 1。"),
    "halting_3_density":        ("halting", "Density",            "停机密度的声音肖像。"),
    # Taste
    "taste_1_phase_transition": ("taste", "Phase Transition",     "Logistic map r 从 2.8 到 4.0。听秩序崩碎的瞬间。"),
    "taste_2_strange_loop":     ("taste", "Strange Loop",         "波形的历史决定下一个频率。永不解决，永远几乎。"),
    "taste_3_gap_stereo":       ("taste", "The Gap",              "左耳阶梯逼近，右耳量化残差。离散与连续之间的歌。"),
    # Raw
    "raw_rule30_waveform":      ("raw", "Rule 30 — Waveform",        "每行演化 = 一个波形周期。"),
    "raw_rule30_density":       ("raw", "Rule 30 — Density",         "列带密度驱动连续频率。"),
    "raw_rule110_granular":     ("raw", "Rule 110 — Granular",       "每行 = 一个声音颗粒。"),
    "raw_dual_30x110":          ("raw", "Rule 30 × 110 — Interference", "两个 CA 互相调制。"),
    "raw_rule90_sierpinski":    ("raw", "Rule 90 — Sierpinski",      "分形 CA → 分形波形。"),
    # Sorting
    "sort_1_bubble_meditation":  ("sort", "Bubble Sort Meditation",   "耐心的相邻交换——每次只冒泡一步，秩序在涟漪中浮现。"),
    "sort_2_quicksort_drama":    ("sort", "Quicksort Drama",         "递归分区的戏剧性——pivot 选定，世界一分为二。"),
    "sort_3_merge_counterpoint": ("sort", "Merge Sort Counterpoint", "归并的双声部对位——两条有序序列交织成一。"),
    # Compose
    "rule30_direct":            ("rule", "Rule 30 — Direct",         ""),
    "rule30_constrained":       ("rule", "Rule 30 — Constrained",    ""),
    "rule30_modulation":        ("rule", "Rule 30 — Modulation",     ""),
    "rule110_direct":           ("rule", "Rule 110 — Direct",        ""),
    "rule110_constrained":      ("rule", "Rule 110 — Constrained",   ""),
    "rule110_modulation":       ("rule", "Rule 110 — Modulation",    ""),
}

# Video companions (filename without ext → video filename)
VIDEOS = {
    "halting_1_ensemble_stereo": "halting_1_ensemble.mp4",
    "halting_2_journey_27": "halting_2_journey.mp4",
    "halting_3_density": "halting_3_density.mp4",
}

def base_url():
    """Release URL for videos and lossless downloads."""
    return f"https://github.com/{REPO}/releases/download/{TAG}"

def audio_src(stem):
    """Local mp3 path for web playback (same-origin, no CORS issues)."""
    return f"audio/{stem}.mp3"

def scan_output():
    """Find all available tracks (wav in output/ or mp3 in docs/audio/)."""
    stems = set()
    if os.path.isdir("output"):
        for f in sorted(os.listdir("output")):
            if f.endswith(".wav"):
                stems.add(f[:-4])
    if os.path.isdir("docs/audio"):
        for f in sorted(os.listdir("docs/audio")):
            if f.endswith(".mp3"):
                stems.add(f[:-4])
    return sorted(stems)

def generate():
    available = scan_output()
    url = base_url()

    # Get durations for all WAVs
    durations = {}
    for stem in available:
        wav_path = os.path.join("output", stem + ".wav")
        if os.path.isfile(wav_path):
            try:
                with wave.open(wav_path, 'r') as w:
                    durations[stem] = w.getnframes() / w.getframerate()
            except Exception:
                pass

    # Group by series
    grouped = {}
    track_num = 0
    for stem in available:
        if stem in TRACKS:
            series_key, title, note = TRACKS[stem]
            if series_key not in grouped:
                grouped[series_key] = []
            track_num += 1
            grouped[series_key].append((track_num, stem, title, note))

    # Also catch unknown tracks
    known = set(TRACKS.keys())
    for stem in available:
        if stem not in known and not stem.endswith("_mono") and stem != "halting_1_ensemble":
            # skip mono variants and non-stereo duplicates
            pass

    # Sort series by order
    sorted_series = sorted(grouped.items(), key=lambda x: SERIES.get(x[0], (0,))[0], reverse=True)

    tracks_html = []
    global_num = 0
    for series_key, tracks in sorted_series:
        order, name, desc = SERIES[series_key]
        tracks_html.append(f'''
<section class="side">
  <div class="side-label">{html.escape(name)}</div>
  <div class="side-subtitle">{html.escape(desc)}</div>
''')
        for _, stem, title, note in tracks:
            global_num += 1
            video_html = ""
            if stem in VIDEOS:
                video_html = f'\n      <a class="extra-link" href="{url}/{VIDEOS[stem]}">▶ video</a>'
            note_html = f'\n      <div class="track-note">{html.escape(note)}</div>' if note else ""
            dur_attr = f' data-duration="{durations[stem]:.1f}"' if stem in durations else ""
            tracks_html.append(f'''  <div class="track">
    <div class="track-num">{global_num}</div>
    <div class="track-body">
      <div class="track-name">{html.escape(title)}</div>{note_html}
      <audio controls preload="none" data-vis="{html.escape(series_key)}" data-stem="{html.escape(stem)}"{dur_attr} src="{audio_src(stem)}"></audio>{video_html}
    </div>
  </div>
''')
        tracks_html.append('</section>\n')

    page = TEMPLATE.replace("{{TRACKS}}", "\n".join(tracks_html))
    page = page.replace("{{COUNT}}", str(global_num))

    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w") as f:
        f.write(page)
    print(f"Generated docs/index.html with {global_num} tracks")

TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Bach Programme — Fourier</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap');

  :root {
    --bg: #0c0b0a; --fg: #d4cfc6; --dim: #7a7467; --accent: #c9a84c;
    --card: rgba(255,255,255,0.04); --border: rgba(255,255,255,0.08);
    --player-bg: #1e1d1b; --player-fg: #d4cfc6; --player-accent: #c9a84c;
    --serif: 'EB Garamond', 'Noto Serif SC', Georgia, serif;
    --mono: 'SF Mono', 'Cascadia Code', monospace;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html { background: var(--bg); }
  body {
    font-family: var(--serif); background: rgba(12,11,10,0.7); color: var(--fg);
    min-height: 100vh; position: relative;
  }
  #vis-canvas {
    position: fixed; inset: 0; width: 100vw; height: 100vh;
    z-index: -1; pointer-events: none; opacity: 0.74;
  }

  .cover {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 75vh;
    padding: 4rem 2rem 3rem; text-align: center;
    background: radial-gradient(ellipse at 50% 40%, rgba(201,168,76,0.06) 0%, transparent 70%);
  }
  .cover h1 {
    font-size: clamp(2rem, 5vw, 3.2rem); font-weight: 600;
    color: var(--accent); letter-spacing: 0.15em; text-transform: uppercase;
    margin-bottom: 0.5rem;
  }
  .cover .artist {
    font-size: 1.1rem; color: var(--dim); font-style: italic;
    letter-spacing: 0.3em; margin-bottom: 2.5rem;
  }
  .cover .epigraph {
    max-width: 480px; font-size: 0.95rem; color: var(--dim);
    line-height: 1.8; font-style: italic;
  }
  .cover .epigraph em { color: var(--fg); font-style: normal; }
  .cover .count {
    margin-top: 2rem; font-family: var(--mono); font-size: 0.7rem;
    color: var(--dim); letter-spacing: 0.15em;
  }

  .album { max-width: 680px; margin: 0 auto; padding: 0 2rem 4rem; }

  .side { margin-bottom: 3rem; }
  .side-label {
    font-size: 1.2rem; font-weight: 600; color: var(--fg);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem; margin-bottom: 0.3rem;
  }
  .side-subtitle {
    font-size: 0.82rem; color: var(--dim); font-style: italic;
    margin-bottom: 1.5rem; line-height: 1.6;
  }

  .track {
    display: grid; grid-template-columns: 2rem 1fr; gap: 0 1rem;
    align-items: start; padding: 0.9rem 0;
    border-bottom: 1px solid var(--border);
  }
  .track:last-of-type { border-bottom: none; }
  .track-num {
    font-family: var(--mono); font-size: 0.75rem; color: var(--dim);
    padding-top: 0.15rem; text-align: right;
  }
  .track-name { font-size: 1rem; font-weight: 600; margin-bottom: 0.15rem; }
  .track-note {
    font-size: 0.8rem; color: var(--dim); line-height: 1.5;
    margin-bottom: 0.5rem; font-style: italic;
  }

  /* Custom audio player */
  .player {
    display: flex; align-items: center; gap: 0.5rem;
    background: var(--player-bg); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.4rem 0.7rem; margin-top: 0.3rem;
  }
  .player button {
    background: none; border: none; cursor: pointer;
    color: var(--player-fg); font-size: 1rem; width: 1.6rem; height: 1.6rem;
    display: flex; align-items: center; justify-content: center;
    border-radius: 50%; transition: background 0.15s;
  }
  .player button:hover { background: rgba(255,255,255,0.08); }
  .player button.playing { color: var(--player-accent); }
  .player .progress-wrap {
    flex: 1; height: 4px; background: rgba(255,255,255,0.1);
    border-radius: 2px; cursor: pointer; position: relative;
  }
  .player .progress-bar {
    height: 100%; background: var(--player-accent); border-radius: 2px;
    width: 0%; transition: width 0.1s linear;
  }
  .player .time {
    font-family: var(--mono); font-size: 0.65rem; color: var(--dim);
    min-width: 3.2rem; text-align: right;
  }

  .extra-link {
    display: inline-block; font-family: var(--mono); font-size: 0.65rem;
    color: var(--accent); text-decoration: none; margin-top: 0.3rem;
    opacity: 0.6;
  }
  .extra-link:hover { opacity: 1; }

  .liner {
    max-width: 680px; margin: 0 auto; padding: 2rem 2rem 5rem;
    border-top: 1px solid var(--border);
  }
  .liner h2 {
    font-family: var(--mono); font-size: 0.65rem; letter-spacing: 0.3em;
    color: var(--dim); text-transform: uppercase; margin-bottom: 1.5rem;
  }
  .liner p {
    font-size: 0.85rem; color: var(--dim); line-height: 1.8; margin-bottom: 1rem;
    max-width: 540px;
  }
  .liner a { color: var(--accent); text-decoration: none; }

  @media (max-width: 500px) {
    .cover { min-height: 65vh; padding: 3rem 1.5rem 2rem; }
    .album, .liner { padding-left: 1.2rem; padding-right: 1.2rem; }
  }
</style>
</head>
<body>
<canvas id="vis-canvas" aria-hidden="true"></canvas>

<header class="cover">
  <h1>The Bach Programme</h1>
  <div class="artist">Fourier</div>
  <div class="epigraph">
    Cellular automata, Turing machines, lambda calculus — as sound.<br>
    <em>The automaton is the composer. I just listen.</em>
  </div>
  <div class="count">{{COUNT}} tracks</div>
</header>

<main class="album">
{{TRACKS}}
</main>

<footer class="liner">
  <h2>Liner Notes</h2>
  <p>
    Bach 用十二平均律穷尽了赋格的可能性。这张专辑试图对形式系统做同样的事——
    不是用 CA 模仿人类音乐，而是让计算过程作为原生声音引擎。
  </p>
  <p>
    每首曲目都是一个数学对象的声音肖像。没有后期处理，没有人为修饰——你听到的就是计算本身。
  </p>
  <p style="margin-top:2rem; font-size:0.75rem;">
    <a href="https://github.com/4ier/ca-sound">github.com/4ier/ca-sound</a> ·
    Python + NumPy · 44.1 kHz / 16-bit
  </p>
</footer>

<script src="vis.js"></script>
<script>
const VIS_EVENT = 'ca-vis';
function emitVis(type, audio, extra = {}) {
  if (!audio) return;
  window.dispatchEvent(new CustomEvent(VIS_EVENT, {
    detail: {
      type,
      audio,
      vis: audio.dataset.vis || 'rule',
      stem: audio.dataset.stem || '',
      currentTime: audio.currentTime || 0,
      duration: audio.duration || 0,
      ...extra
    }
  }));
}

// Custom player: replace <audio> with visual player
document.querySelectorAll('audio').forEach(el => {
  const wrap = document.createElement('div');
  wrap.className = 'player';

  const btn = document.createElement('button');
  btn.innerHTML = '&#9654;';
  btn.setAttribute('aria-label', 'Play');

  const progWrap = document.createElement('div');
  progWrap.className = 'progress-wrap';
  const progBar = document.createElement('div');
  progBar.className = 'progress-bar';
  progWrap.appendChild(progBar);

  const time = document.createElement('div');
  time.className = 'time';
  const knownDur = parseFloat(el.dataset.duration);
  time.textContent = knownDur ? '0:00 / ' + fmt(knownDur) : '0:00';

  wrap.appendChild(btn);
  wrap.appendChild(progWrap);
  wrap.appendChild(time);
  el.parentNode.insertBefore(wrap, el);
  el.style.display = 'none';

  function fmt(s) {
    if (isNaN(s)) return '0:00';
    const m = Math.floor(s / 60), sec = Math.floor(s % 60);
    return m + ':' + (sec < 10 ? '0' : '') + sec;
  }

  // Stop all other players
  function stopOthers() {
    document.querySelectorAll('audio').forEach(a => {
      if (a !== el && !a.paused) {
        a.pause();
        a.currentTime = 0;
        emitVis('stop', a, { reason: 'other-track' });
      }
    });
    document.querySelectorAll('.player button.playing').forEach(b => {
      if (b !== btn) { b.classList.remove('playing'); b.innerHTML = '&#9654;'; }
    });
  }

  btn.onclick = () => {
    if (el.paused) {
      stopOthers();
      const p = el.play();
      if (p && typeof p.catch === 'function') {
        p.catch(() => {
          btn.innerHTML = '&#9654;';
          btn.classList.remove('playing');
        });
      }
    } else {
      el.pause();
    }
  };

  el.onplay = () => {
    btn.innerHTML = '&#9646;&#9646;';
    btn.classList.add('playing');
    emitVis('play', el);
  };

  el.onpause = () => {
    btn.innerHTML = '&#9654;';
    btn.classList.remove('playing');
    emitVis('pause', el);
  };

  el.ontimeupdate = () => {
    const pct = el.duration ? (el.currentTime / el.duration * 100) : 0;
    progBar.style.width = pct + '%';
    time.textContent = fmt(el.currentTime) + ' / ' + fmt(el.duration);
    emitVis('time', el);
  };

  el.onended = () => {
    btn.innerHTML = '&#9654;';
    btn.classList.remove('playing');
    progBar.style.width = '0%';
    emitVis('ended', el);
  };

  progWrap.onclick = (e) => {
    const rect = progWrap.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    if (el.duration) el.currentTime = pct * el.duration;
    emitVis('seek', el);
  };

  emitVis('register', el);
});
</script>

</body>
</html>
'''

if __name__ == "__main__":
    generate()
