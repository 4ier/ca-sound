(function () {
  'use strict';
  const BG = '#0c0b0a';
  const GOLD = [201, 168, 76];
  const TAU = Math.PI * 2;
  const canvas = document.getElementById('vis-canvas') || createCanvas();
  const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
  if (!ctx) return;
  const state = {
    w: 0,
    h: 0,
    dpr: 1,
    running: false,
    raf: 0,
    lastTs: performance.now(),
    activeAudio: null,
    activeVis: 'rule',
    activeStem: '',
    playhead: 0,
    duration: 0,
    intensity: 0,
    targetIntensity: 0,
    audioReady: false,
    audioFailed: false,
    fallbackOnly: false,
    freqData: new Uint8Array(1024),
    timeDomainData: new Uint8Array(2048),
    prevFreqData: new Uint8Array(1024),
    metrics: {
      bass: 0,
      mid: 0,
      treble: 0,
      energy: 0,
      rms: 0,
      flux: 0,
      transient: 0,
      beat: false,
      peak: 0,
      nowMs: 0,
      beatGateMs: 0,
      lastBass: 0
    }
  };
  const analysis = {
    frequencyData: state.freqData,
    timeDomainData: state.timeDomainData,
    bass: 0,
    mid: 0,
    treble: 0,
    energy: 0,
    rms: 0,
    flux: 0,
    transient: 0,
    beat: false
  };
  const sourceCache = new WeakMap();
  let audioCtx = null;
  let analyser = null;
  let outputGain = null;
  const sketchGameOfLife = createGameOfLifeSketch();
  const sketchFractal = createFractalSketch();
  const sketchLambda = createLambdaSketch();
  const sketchBeaver = createBeaverSketch();
  const sketchHalting = createHaltingSketch();
  const sketchTaste = createTasteSketch();
  const sketchSort = createSortSketch();
  const sketchQC = createQuantumSketch();
  const sketchNN = createNeuralNetSketch();
  const sketchOpt = createOptimizationSketch();
  const sketchNT = createNumberTheorySketch();
  const sketchInfo = createInformationSketch();
  const sketchGraph = createGraphSketch();
  const sketchCrypto = createCryptoSketch();
  const sketchConc = createConcurrencySketch();
  const sketchAuto = createAutomataSketch();
  const sketchType = createTypeSystemsSketch();
  const sketchDist = createDistributedSketch();
  const sketchCat = createCategorySketch();
  const sketchRaw = createRawSketch();
  const sketchCompose = createComposeSketch();
  const sketchAmbient = createAmbientSketch();
  const sketchMap = {
    gol: sketchGameOfLife,
    fractal: sketchFractal,
    lambda: sketchLambda,
    beaver: sketchBeaver,
    halting: sketchHalting,
    taste: sketchTaste,
    sort: sketchSort,
    qc: sketchQC,
    nn: sketchNN,
    opt: sketchOpt,
    nt: sketchNT,
    info: sketchInfo,
    graph: sketchGraph,
    crypto: sketchCrypto,
    conc: sketchConc,
    auto: sketchAuto,
    type: sketchType,
    cat: sketchCat,
    dist: sketchDist,
    raw: sketchRaw,
    rule: sketchCompose
  };
  window.caVis = {
    analysis,
    get activeVis() {
      return state.activeVis;
    },
    get activeStem() {
      return state.activeStem;
    },
    resume() {
      ensureAudioEngine();
      if (audioCtx && audioCtx.state === 'suspended') {
        return audioCtx.resume();
      }
      return Promise.resolve();
    }
  };
  init();
  function init() {
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas, { passive: true });
    window.addEventListener('orientationchange', resizeCanvas, { passive: true });
    bindActivationEvents();
    bindAudioElements();
    window.addEventListener('ca-vis', onVisEvent);
    startLoop();
  }
  function createCanvas() {
    const el = document.createElement('canvas');
    el.id = 'vis-canvas';
    el.setAttribute('aria-hidden', 'true');
    document.body.insertBefore(el, document.body.firstChild);
    return el;
  }
  function bindActivationEvents() {
    const activate = () => {
      ensureAudioEngine();
      if (audioCtx && audioCtx.state === 'suspended') {
        audioCtx.resume().catch(() => {});
      }
    };
    ['pointerdown', 'touchstart', 'mousedown', 'keydown'].forEach((ev) => {
      window.addEventListener(ev, activate, { passive: true });
    });
  }
  function bindAudioElements() {
    document.querySelectorAll('audio').forEach((audio) => {
      if (!audio.dataset.vis) {
        const stem = audio.dataset.stem || parseStem(audio.currentSrc || audio.src);
        audio.dataset.vis = visFromStem(stem);
      }
      if (!audio.dataset.stem) {
        audio.dataset.stem = parseStem(audio.currentSrc || audio.src);
      }
      audio.addEventListener('play', () => {
        activateTrack(audio, audio.dataset.vis || 'rule', audio.dataset.stem || '');
      });
      audio.addEventListener('pause', () => {
        if (state.activeAudio === audio) {
          state.targetIntensity = 0;
        }
      });
      audio.addEventListener('ended', () => {
        if (state.activeAudio === audio) {
          state.activeAudio = null;
          state.targetIntensity = 0;
        }
      });
    });
  }
  function onVisEvent(evt) {
    const detail = evt && evt.detail;
    if (!detail || !detail.type) return;
    const audio = detail.audio || null;
    const vis = detail.vis || (audio && audio.dataset ? audio.dataset.vis : 'rule') || 'rule';
    const stem = detail.stem || (audio && audio.dataset ? audio.dataset.stem : '') || '';
    if (detail.type === 'register') {
      if (audio) {
        if (!audio.dataset.vis) audio.dataset.vis = visFromStem(stem || parseStem(audio.src || ''));
        if (!audio.dataset.stem) audio.dataset.stem = stem || parseStem(audio.src || '');
      }
      return;
    }
    if (detail.type === 'play') {
      activateTrack(audio, vis, stem);
      return;
    }
    if (detail.type === 'time') {
      if (audio && state.activeAudio === audio) {
        state.playhead = Number.isFinite(detail.currentTime) ? detail.currentTime : audio.currentTime || 0;
        state.duration = Number.isFinite(detail.duration) ? detail.duration : audio.duration || 0;
      }
      return;
    }
    if ((detail.type === 'pause' || detail.type === 'ended' || detail.type === 'stop') && state.activeAudio === audio) {
      if (detail.type === 'ended' || detail.type === 'stop') {
        state.activeAudio = null;
      }
      state.targetIntensity = 0;
    }
    if (detail.type === 'seek' && audio && state.activeAudio === audio) {
      state.playhead = audio.currentTime || 0;
      state.duration = audio.duration || 0;
    }
  }
  function activateTrack(audio, vis, stem) {
    if (!audio) return;
    ensureAudioEngine();
    if (audioCtx && audioCtx.state === 'suspended') {
      audioCtx.resume().catch(() => {});
    }
    if (state.audioReady) {
      connectSource(audio);
    }
    state.activeAudio = audio;
    state.activeVis = vis || visFromStem(stem || parseStem(audio.src || ''));
    state.activeStem = stem || parseStem(audio.currentSrc || audio.src || '');
    state.playhead = audio.currentTime || 0;
    state.duration = audio.duration || 0;
    state.targetIntensity = 1;
    startLoop();
  }
  function ensureAudioEngine() {
    if (state.audioReady || state.audioFailed) return state.audioReady;
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) {
      state.audioFailed = true;
      state.fallbackOnly = true;
      return false;
    }
    try {
      audioCtx = new AudioCtx();
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.84;
      outputGain = audioCtx.createGain();
      outputGain.gain.value = 1;
      analyser.connect(outputGain);
      outputGain.connect(audioCtx.destination);
      state.freqData = new Uint8Array(analyser.frequencyBinCount);
      state.prevFreqData = new Uint8Array(analyser.frequencyBinCount);
      state.timeDomainData = new Uint8Array(analyser.fftSize);
      analysis.frequencyData = state.freqData;
      analysis.timeDomainData = state.timeDomainData;
      state.audioReady = true;
      document.querySelectorAll('audio').forEach(connectSource);
      return true;
    } catch (err) {
      console.warn('[ca-vis] AudioContext unavailable, switching to fallback animation.', err);
      state.audioFailed = true;
      state.fallbackOnly = true;
      return false;
    }
  }
  function connectSource(audio) {
    if (!audio || !state.audioReady || !audioCtx || sourceCache.has(audio)) return;
    try {
      const src = audioCtx.createMediaElementSource(audio);
      src.connect(analyser);
      sourceCache.set(audio, src);
    } catch (err) {
      // Already-connected element or unsupported element: keep page functional.
      console.warn('[ca-vis] could not connect media source', err);
    }
  }
  function resizeCanvas() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    state.w = w;
    state.h = h;
    state.dpr = dpr;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);
    [
      sketchGameOfLife,
      sketchFractal,
      sketchLambda,
      sketchBeaver,
      sketchHalting,
      sketchTaste,
      sketchSort,
      sketchOpt,
      sketchNT,
      sketchInfo,
      sketchGraph,
      sketchCrypto,
      sketchConc,
      sketchAuto,
      sketchRaw,
      sketchCompose,
      sketchAmbient
    ].forEach((s) => {
      if (typeof s.resize === 'function') s.resize(w, h);
    });
  }
  function startLoop() {
    if (state.running) return;
    state.running = true;
    state.lastTs = performance.now();
    state.raf = requestAnimationFrame(loop);
  }
  function loop(ts) {
    const dt = Math.min(0.05, Math.max(0.001, (ts - state.lastTs) / 1000));
    state.lastTs = ts;
    if (state.activeAudio) {
      state.playhead = state.activeAudio.currentTime || state.playhead;
      state.duration = state.activeAudio.duration || state.duration;
      state.targetIntensity = state.activeAudio.paused ? 0 : 1;
    }
    updateAudioMetrics(ts);
    state.intensity += (state.targetIntensity - state.intensity) * Math.min(1, dt * 4.2);
    const fade = state.targetIntensity > 0 ? 0.11 : 0.25;
    ctx.fillStyle = `rgba(12,11,10,${fade})`;
    ctx.fillRect(0, 0, state.w, state.h);
    const api = {
      ctx,
      w: state.w,
      h: state.h,
      dt,
      ts,
      audio: analysis,
      intensity: state.intensity,
      vis: state.activeVis,
      stem: state.activeStem,
      playhead: state.playhead,
      duration: state.duration,
      hasTrack: !!state.activeAudio && !state.activeAudio.paused
    };
    const sketch = resolveSketch(state.activeVis, state.activeStem);
    if (api.hasTrack) {
      // Use time-based fallback when audio analysis returns zero (CORS blocked)
      if (api.audio.energy < 0.01 && state.activeAudio && !state.activeAudio.paused) {
        const t = (ts % 4000) / 4000;
        const fakeIntensity = 0.3 + 0.2 * Math.sin(t * Math.PI * 2);
        api.audio.bass = fakeIntensity * 0.8;
        api.audio.mid = fakeIntensity * 0.6;
        api.audio.treble = fakeIntensity * 0.4;
        api.audio.energy = fakeIntensity;
        api.audio.rms = fakeIntensity * 0.5;
      }
      sketch.draw(api);
    } else if (state.fallbackOnly || !state.audioReady) {
      sketchAmbient.draw(api);
    }
    state.raf = requestAnimationFrame(loop);
  }
  function resolveSketch(vis, stem) {
    if (vis && sketchMap[vis]) return sketchMap[vis];
    if ((stem || '').startsWith('rule')) return sketchCompose;
    return sketchCompose;
  }
  function updateAudioMetrics(ts) {
    const m = state.metrics;
    m.nowMs = ts;
    const isPlaying = state.activeAudio && !state.activeAudio.paused;
    if (!state.audioReady || !analyser || !isPlaying) {
      if (isPlaying) {
        // Audio playing but no analyser (CORS blocked) — use time-based fallback
        const t = (ts % 3000) / 3000;
        const pulse = 0.3 + 0.25 * Math.sin(t * Math.PI * 2);
        const slow = 0.2 + 0.15 * Math.sin(t * Math.PI * 0.7);
        m.bass = pulse;
        m.mid = slow;
        m.treble = 0.15 + 0.1 * Math.sin(t * Math.PI * 3.1);
        m.energy = pulse * 0.6 + slow * 0.4;
        m.rms = pulse * 0.5;
        m.flux = 0.03 + 0.02 * Math.sin(t * Math.PI * 5);
        m.transient = 0;
        m.beat = Math.sin(t * Math.PI * 2) > 0.95;
        syncAnalysis();
        return;
      }
      m.bass *= 0.92;
      m.mid *= 0.92;
      m.treble *= 0.92;
      m.energy *= 0.9;
      m.rms *= 0.9;
      m.flux *= 0.86;
      m.transient *= 0.82;
      m.beat = false;
      syncAnalysis();
      return;
    }
    analyser.getByteFrequencyData(state.freqData);
    analyser.getByteTimeDomainData(state.timeDomainData);
    // Detect CORS-tainted source (all frequency data is zero)
    let allZero = true;
    for (let i = 0; i < state.freqData.length; i += 16) {
      if (state.freqData[i] > 0) { allZero = false; break; }
    }
    if (allZero && isPlaying) {
      // CORS blocked — switch to time-based fallback permanently
      state.audioReady = false;
      updateAudioMetrics(ts);
      return;
    }
    const bass = bandEnergy(20, 140);
    const mid = bandEnergy(140, 2200);
    const treble = bandEnergy(2200, 12000);
    let rmsAcc = 0;
    for (let i = 0; i < state.timeDomainData.length; i += 2) {
      const v = (state.timeDomainData[i] - 128) / 128;
      rmsAcc += v * v;
    }
    const rms = Math.sqrt(rmsAcc / (state.timeDomainData.length / 2));
    let fluxAcc = 0;
    for (let i = 0; i < state.freqData.length; i += 2) {
      const cur = state.freqData[i];
      const prev = state.prevFreqData[i] || 0;
      const diff = cur - prev;
      if (diff > 0) fluxAcc += diff;
      state.prevFreqData[i] = cur;
    }
    const flux = fluxAcc / ((state.freqData.length / 2) * 255);
    const transient = Math.max(0, flux * 2.3 + (bass - m.lastBass) * 1.3 - 0.08);
    m.lastBass = bass;
    const beatReady = ts - m.beatGateMs > 170;
    const beat = beatReady && bass > 0.22 && (flux > 0.06 || transient > 0.09);
    if (beat) m.beatGateMs = ts;
    m.bass = bass;
    m.mid = mid;
    m.treble = treble;
    m.energy = (bass * 0.44 + mid * 0.36 + treble * 0.2);
    m.rms = rms;
    m.flux = flux;
    m.transient = transient;
    m.beat = beat;
    m.peak = Math.max(0, transient * 1.4 + bass * 0.35);
    syncAnalysis();
  }
  function syncAnalysis() {
    const m = state.metrics;
    analysis.bass = m.bass;
    analysis.mid = m.mid;
    analysis.treble = m.treble;
    analysis.energy = m.energy;
    analysis.rms = m.rms;
    analysis.flux = m.flux;
    analysis.transient = m.transient;
    analysis.beat = m.beat;
  }
  function bandEnergy(lowHz, highHz) {
    const bins = state.freqData;
    if (!bins || bins.length === 0 || !audioCtx) return 0;
    const nyquist = audioCtx.sampleRate / 2;
    const i0 = clamp(Math.floor((lowHz / nyquist) * bins.length), 0, bins.length - 1);
    const i1 = clamp(Math.floor((highHz / nyquist) * bins.length), i0 + 1, bins.length);
    let sum = 0;
    for (let i = i0; i < i1; i++) sum += bins[i];
    return (sum / (i1 - i0)) / 255;
  }
  function visFromStem(stem) {
    if (!stem) return 'rule';
    if (stem.startsWith('gol_')) return 'gol';
    if (stem.startsWith('fractal_')) return 'fractal';
    if (stem.startsWith('lambda_')) return 'lambda';
    if (stem.startsWith('beaver_')) return 'beaver';
    if (stem.startsWith('halting_')) return 'halting';
    if (stem.startsWith('taste_')) return 'taste';
    if (stem.startsWith('sort_')) return 'sort';
    if (stem.startsWith('opt_')) return 'opt';
    if (stem.startsWith('nt_')) return 'nt';
    if (stem.startsWith('info_')) return 'info';
    if (stem.startsWith('graph_')) return 'graph';
    if (stem.startsWith('crypto_')) return 'crypto';
    if (stem.startsWith('conc_')) return 'conc';
    if (stem.startsWith('auto_')) return 'auto';
    if (stem.startsWith('raw_')) return 'raw';
    if (stem.startsWith('rule')) return 'rule';
    return 'rule';
  }
  function parseStem(src) {
    const path = (src || '').split('?')[0].split('#')[0];
    const name = path.substring(path.lastIndexOf('/') + 1);
    return name.replace(/\.wav$/i, '');
  }
  function createAmbientSketch() {
    const particles = [];
    let seed = 1;
    function resize(w, h) {
      const count = Math.max(24, Math.floor((w * h) / 38000));
      while (particles.length < count) {
        particles.push({
          x: Math.random() * w,
          y: Math.random() * h,
          vx: (Math.random() - 0.5) * 12,
          vy: (Math.random() - 0.5) * 8,
          r: 1 + Math.random() * 2,
          a: 0.08 + Math.random() * 0.08
        });
      }
      particles.length = count;
      seed = 1;
    }
    function draw(api) {
      const { ctx: c, w, h, dt } = api;
      const alpha = 0.12 + api.intensity * 0.08;
      c.save();
      c.globalCompositeOperation = 'lighter';
      c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${alpha})`;
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.vx += (hash(seed + i * 11.3) - 0.5) * 2.2 * dt;
        p.vy += (hash(seed + i * 7.1) - 0.5) * 2.2 * dt;
        p.vx *= 0.995;
        p.vy *= 0.995;
        if (p.x < -20) p.x = w + 10;
        if (p.x > w + 20) p.x = -10;
        if (p.y < -20) p.y = h + 10;
        if (p.y > h + 20) p.y = -10;
        c.globalAlpha = p.a;
        c.beginPath();
        c.arc(p.x, p.y, p.r, 0, TAU);
        c.fill();
      }
      c.restore();
      seed += dt;
    }
    return { resize, draw };
  }
  function createGameOfLifeSketch() {
    const s = {
      cols: 0,
      rows: 0,
      cell: 10,
      grid: null,
      next: null,
      trail: null,
      accum: 0,
      lastStem: ''
    };
    function resize(w, h) {
      const targetCells = 7600;
      s.cell = clamp(Math.sqrt((w * h) / targetCells), 8, 16);
      s.cols = Math.max(24, Math.floor(w / s.cell));
      s.rows = Math.max(18, Math.floor(h / s.cell));
      const size = s.cols * s.rows;
      s.grid = new Uint8Array(size);
      s.next = new Uint8Array(size);
      s.trail = new Float32Array(size);
      s.accum = 0;
      for (let i = 0; i < size; i++) {
        s.grid[i] = Math.random() < 0.15 ? 1 : 0;
        s.trail[i] = 0;
      }
    }
    function draw(api) {
      if (!s.grid || api.stem !== s.lastStem) {
        resize(api.w, api.h);
        s.lastStem = api.stem;
      }
      const a = api.audio;
      const speed = 2.3 + a.energy * 16 + a.rms * 8;
      s.accum += api.dt * speed;
      while (s.accum >= 1) {
        step(a.bass);
        s.accum -= 1;
      }
      const c = api.ctx;
      const liveAlpha = 0.25 + a.energy * 0.45;
      c.save();
      c.globalCompositeOperation = 'lighter';
      for (let i = 0; i < s.trail.length; i++) {
        s.trail[i] = Math.max(0, s.trail[i] - api.dt / 2);
      }
      c.fillStyle = `rgba(92,214,177,0.17)`;
      for (let y = 0; y < s.rows; y++) {
        const py = (y + 0.5) * s.cell;
        for (let x = 0; x < s.cols; x++) {
          const idx = y * s.cols + x;
          const t = s.trail[idx];
          if (t < 0.02) continue;
          c.globalAlpha = t * (0.22 + a.mid * 0.5);
          c.fillRect((x + 0.5) * s.cell - 2, py - 2, 4, 4);
        }
      }
      c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${liveAlpha})`;
      c.globalAlpha = 1;
      for (let y = 0; y < s.rows; y++) {
        const py = (y + 0.5) * s.cell;
        for (let x = 0; x < s.cols; x++) {
          const idx = y * s.cols + x;
          if (!s.grid[idx]) continue;
          const px = (x + 0.5) * s.cell;
          c.fillRect(px - 1.4, py - 1.4, 2.8, 2.8);
        }
      }
      c.restore();
    }
    function step(bass) {
      const cols = s.cols;
      const rows = s.rows;
      const birthBoost = bass * 0.05;
      for (let y = 0; y < rows; y++) {
        const yUp = (y - 1 + rows) % rows;
        const yDn = (y + 1) % rows;
        for (let x = 0; x < cols; x++) {
          const xLt = (x - 1 + cols) % cols;
          const xRt = (x + 1) % cols;
          const idx = y * cols + x;
          let n = 0;
          n += s.grid[yUp * cols + xLt];
          n += s.grid[yUp * cols + x];
          n += s.grid[yUp * cols + xRt];
          n += s.grid[y * cols + xLt];
          n += s.grid[y * cols + xRt];
          n += s.grid[yDn * cols + xLt];
          n += s.grid[yDn * cols + x];
          n += s.grid[yDn * cols + xRt];
          const alive = s.grid[idx] === 1;
          let next = 0;
          if (alive) {
            next = n === 2 || n === 3 ? 1 : 0;
            if (!next) s.trail[idx] = 1;
          } else {
            if (n === 3 || (n === 2 && Math.random() < birthBoost)) {
              next = 1;
            }
          }
          s.next[idx] = next;
        }
      }
      if (bass > 0.35) {
        for (let i = 0; i < 8; i++) {
          const idx = (Math.random() * s.grid.length) | 0;
          s.next[idx] = 1;
        }
      }
      const tmp = s.grid;
      s.grid = s.next;
      s.next = tmp;
    }
    return { resize, draw };
  }
  function createBeaverSketch() {
    const s = {
      tapes: [],
      cellW: 11,
      cellH: 10,
      spacing: 26,
      viewOffset: 0,
      stepAcc: 0,
      lastStem: ''
    };
    function resize(w, h) {
      const tapeCount = clamp(Math.floor(h / 170), 3, 6);
      const cells = 220;
      s.cellW = clamp(Math.floor(w / 72), 7, 14);
      s.cellH = clamp(Math.floor(h / 95), 7, 13);
      s.spacing = Math.max(20, Math.floor(h / (tapeCount + 5)));
      s.tapes = [];
      for (let t = 0; t < tapeCount; t++) {
        const row = {
          cells: new Uint8Array(cells),
          flash: new Float32Array(cells),
          head: Math.floor(cells / 2),
          dir: Math.random() < 0.5 ? -1 : 1,
          pulse: 0
        };
        for (let i = 0; i < cells; i++) {
          row.cells[i] = Math.random() < 0.04 ? 1 : 0;
        }
        s.tapes.push(row);
      }
      s.stepAcc = 0;
      s.viewOffset = cells / 2;
    }
    function draw(api) {
      if (!s.tapes.length || api.stem !== s.lastStem) {
        resize(api.w, api.h);
        s.lastStem = api.stem;
      }
      const a = api.audio;
      const stepsPerSec = 3 + a.energy * 11;
      s.stepAcc += api.dt * stepsPerSec;
      while (s.stepAcc >= 1) {
        machineStep(a, false);
        s.stepAcc -= 1;
      }
      if (a.beat) machineStep(a, true);
      let headAvg = 0;
      for (let i = 0; i < s.tapes.length; i++) headAvg += s.tapes[i].head;
      headAvg /= s.tapes.length;
      const visibleCells = Math.ceil(api.w / s.cellW);
      const targetOffset = headAvg - visibleCells * 0.5;
      s.viewOffset = lerp(s.viewOffset, targetOffset, 0.08);
      const c = api.ctx;
      const baseY = api.h * 0.5 - ((s.tapes.length - 1) * s.spacing) * 0.5;
      c.save();
      c.globalCompositeOperation = 'lighter';
      for (let ti = 0; ti < s.tapes.length; ti++) {
        const tape = s.tapes[ti];
        const y = baseY + ti * s.spacing;
        c.globalAlpha = 0.18;
        c.fillStyle = 'rgba(255,255,255,0.12)';
        c.fillRect(0, y - s.cellH * 0.6, api.w, s.cellH * 1.2);
        const start = Math.floor(s.viewOffset);
        for (let i = 0; i <= visibleCells; i++) {
          const idx = start + i;
          if (idx < 0 || idx >= tape.cells.length) continue;
          const x = (idx - s.viewOffset) * s.cellW;
          const bit = tape.cells[idx];
          const flash = tape.flash[idx];
          if (bit) {
            c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.22 + flash * 0.55})`;
            c.globalAlpha = 0.7;
            const w = s.cellW * (1.0 + flash * 0.7);
            const h = s.cellH * (1.0 + flash * 0.6);
            c.fillRect(x - (w - s.cellW) * 0.5, y - h * 0.5, w, h);
          } else {
            c.fillStyle = 'rgba(210,198,170,0.08)';
            c.globalAlpha = 0.45;
            c.fillRect(x, y - s.cellH * 0.35, s.cellW - 1, s.cellH * 0.7);
          }
          if (flash > 0.01) {
            c.fillStyle = `rgba(255,255,255,${flash * 0.5})`;
            c.globalAlpha = 1;
            c.fillRect(x - 1, y - s.cellH * 0.7, s.cellW + 1, s.cellH * 1.4);
          }
          tape.flash[idx] = Math.max(0, flash - api.dt * 1.9);
        }
        const hx = (tape.head - s.viewOffset + 0.5) * s.cellW;
        tape.pulse = Math.max(0, tape.pulse - api.dt * 2.3);
        c.fillStyle = `rgba(255,255,255,${0.45 + tape.pulse * 0.5})`;
        c.beginPath();
        c.arc(hx, y, s.cellH * (0.42 + tape.pulse * 0.65), 0, TAU);
        c.fill();
      }
      c.restore();
    }
    function machineStep(a, forceBeat) {
      for (let ti = 0; ti < s.tapes.length; ti++) {
        const tape = s.tapes[ti];
        if (forceBeat || Math.random() < 0.08 + a.transient * 1.8) {
          tape.dir = Math.random() < 0.5 ? -1 : 1;
        }
        tape.head += tape.dir;
        if (tape.head <= 1) {
          tape.head = 1;
          tape.dir = 1;
        } else if (tape.head >= tape.cells.length - 2) {
          tape.head = tape.cells.length - 2;
          tape.dir = -1;
        }
        const writeOne = Math.random() < (0.28 + a.bass * 0.6);
        if (writeOne) {
          tape.cells[tape.head] = 1;
          tape.flash[tape.head] = 1;
          tape.pulse = 1;
        } else if (Math.random() < 0.15) {
          tape.cells[tape.head] = 0;
          tape.flash[tape.head] = Math.max(tape.flash[tape.head], 0.2);
        }
      }
    }
    return { resize, draw };
  }
  function createHaltingSketch() {
    const s = {
      particles: [],
      spawnCooldown: 0,
      lastStem: ''
    };
    function resize() {
      // Keep particles on resize; the motion naturally re-centers.
    }
    function draw(api) {
      if (api.stem !== s.lastStem) {
        s.particles = [];
        s.spawnCooldown = 0;
        s.lastStem = api.stem;
      }
      const a = api.audio;
      s.spawnCooldown -= api.dt;
      if (s.spawnCooldown <= 0 && (a.transient > 0.09 || a.beat || Math.random() < api.dt * (0.4 + a.energy))) {
        spawnParticle(api.w, api.h, a.energy);
        s.spawnCooldown = 0.04 + Math.random() * 0.12;
      }
      for (let i = s.particles.length - 1; i >= 0; i--) {
        const p = s.particles[i];
        updateParticle(p, api);
        if (p.life <= 0 || p.y > api.h + 30) {
          s.particles.splice(i, 1);
        }
      }
      const c = api.ctx;
      c.save();
      c.globalCompositeOperation = 'lighter';
      for (let i = 0; i < s.particles.length; i++) {
        const p = s.particles[i];
        drawTrail(c, p, a.mid);
        const hot = p.rising ? `rgba(244,211,132,${0.32 + p.life * 0.22})` : `rgba(111,148,228,${0.24 + p.life * 0.16})`;
        c.fillStyle = hot;
        c.beginPath();
        c.arc(p.x, p.y, 1.5 + p.life * 1.4, 0, TAU);
        c.fill();
      }
      c.fillStyle = 'rgba(255,225,175,0.2)';
      c.beginPath();
      c.arc(api.w * 0.5, api.h * 0.9, 4 + a.bass * 14, 0, TAU);
      c.fill();
      c.restore();
    }
    function spawnParticle(w, h, energy) {
      const n0 = 2 + ((Math.random() * (5000 + energy * 30000)) | 0);
      const p = {
        n: n0,
        prevN: n0,
        x: w * (0.25 + Math.random() * 0.5),
        y: h * (0.75 + Math.random() * 0.15),
        stepTimer: 0,
        life: 1,
        rising: false,
        converging: false,
        trail: []
      };
      s.particles.push(p);
      if (s.particles.length > 140) s.particles.shift();
    }
    function updateParticle(p, api) {
      const a = api.audio;
      p.stepTimer -= api.dt;
      if (!p.converging && p.stepTimer <= 0) {
        p.prevN = p.n;
        p.n = nextCollatz(p.n);
        p.stepTimer = 0.045 + (1 - a.energy) * 0.08;
        p.rising = p.n > p.prevN;
        if (p.rising) {
          p.y -= 7 + a.treble * 12;
        } else {
          p.y += 5 + a.mid * 9;
        }
        const logN = Math.log2(p.n + 1);
        p.x += (logN - 7) * (0.24 + a.energy * 0.35) + (Math.random() - 0.5) * 3.2;
        if (p.n === 1) {
          p.converging = true;
        }
      }
      if (p.converging) {
        p.x += (api.w * 0.5 - p.x) * api.dt * 3.3;
        p.y += (api.h * 0.9 - p.y) * api.dt * 3.3;
        p.life -= api.dt * 0.55;
      } else {
        p.life -= api.dt * (0.12 + (p.n > 50000 ? 0.25 : 0));
      }
      p.x = clamp(p.x, -30, api.w + 30);
      p.y = clamp(p.y, -30, api.h + 40);
      p.trail.push({
        x: p.x,
        y: p.y,
        rising: p.rising,
        a: 0.75
      });
      const persist = 0.5 + a.mid * 2.2;
      const decay = api.dt / persist;
      for (let i = p.trail.length - 1; i >= 0; i--) {
        p.trail[i].a -= decay;
        if (p.trail[i].a <= 0) p.trail.splice(i, 1);
      }
      if (p.trail.length > 48) p.trail.splice(0, p.trail.length - 48);
    }
    function drawTrail(c, p, sustain) {
      if (p.trail.length < 2) return;
      c.lineWidth = 1 + sustain * 1.2;
      for (let i = 1; i < p.trail.length; i++) {
        const prev = p.trail[i - 1];
        const cur = p.trail[i];
        const alpha = Math.min(prev.a, cur.a) * 0.32;
        c.strokeStyle = cur.rising ? `rgba(240,202,118,${alpha})` : `rgba(102,130,204,${alpha})`;
        c.beginPath();
        c.moveTo(prev.x, prev.y);
        c.lineTo(cur.x, cur.y);
        c.stroke();
      }
    }
    function nextCollatz(n) {
      if (n <= 1) return 1;
      if ((n & 1) === 0) return n / 2;
      return 3 * n + 1;
    }
    return { resize, draw };
  }
  function createFractalSketch() {
    const s = {
      offscreen: document.createElement('canvas'),
      octx: null,
      image: null,
      bw: 0,
      bh: 0,
      row: 0,
      centerX: -0.743643887037151,
      centerY: 0.13182590420533,
      zoom: 2.2,
      targetZoom: 2.2,
      mode: 'mandelbrot',
      lastJumpMs: 0,
      lastStem: ''
    };
    s.octx = s.offscreen.getContext('2d', { alpha: false });
    const interesting = [
      [-0.7436438870, 0.1318259042],
      [-0.1011, 0.9563],
      [-0.7435, 0.11],
      [-1.25066, 0.02012],
      [0.285, 0.01],
      [-0.70176, -0.3842],
      [-0.835, -0.2321]
    ];
    function resize(w, h) {
      s.bw = clamp(Math.floor(w / 3.2), 140, 460);
      s.bh = clamp(Math.floor(h / 3.2), 90, 300);
      s.offscreen.width = s.bw;
      s.offscreen.height = s.bh;
      s.image = s.octx.createImageData(s.bw, s.bh);
      s.row = 0;
      for (let i = 0; i < s.image.data.length; i += 4) {
        s.image.data[i] = 12;
        s.image.data[i + 1] = 11;
        s.image.data[i + 2] = 10;
        s.image.data[i + 3] = 255;
      }
      s.octx.putImageData(s.image, 0, 0);
    }
    function draw(api) {
      if (!s.image || api.stem !== s.lastStem) {
        resize(api.w, api.h);
        s.mode = /julia/i.test(api.stem) ? 'julia' : 'mandelbrot';
        s.lastStem = api.stem;
      }
      const a = api.audio;
      const t = api.ts * 0.001;
      s.targetZoom = 0.7 + (1 - a.energy) * 1.9 + 0.22 * Math.sin(t * 0.23);
      s.zoom = lerp(s.zoom, s.targetZoom, 0.03 + a.energy * 0.04);
      s.centerX += Math.sin(t * 0.17) * 0.00006 * (0.3 + a.energy);
      s.centerY += Math.cos(t * 0.13) * 0.00005 * (0.3 + a.mid);
      if (a.bass > 0.34 && a.beat && api.ts - s.lastJumpMs > 850) {
        const pick = interesting[(Math.random() * interesting.length) | 0];
        s.centerX = pick[0] + (Math.random() - 0.5) * 0.02;
        s.centerY = pick[1] + (Math.random() - 0.5) * 0.02;
        s.lastJumpMs = api.ts;
      }
      const rowsPerFrame = 8 + ((a.energy * 24) | 0);
      for (let i = 0; i < rowsPerFrame; i++) {
        renderRow(s.row, api);
        s.row = (s.row + 1) % s.bh;
      }
      s.octx.putImageData(s.image, 0, 0);
      const c = api.ctx;
      c.save();
      c.globalAlpha = 0.7;
      c.imageSmoothingEnabled = true;
      c.drawImage(s.offscreen, 0, 0, api.w, api.h);
      c.restore();
    }
    function renderRow(row, api) {
      const data = s.image.data;
      const a = api.audio;
      const maxIter = 44 + ((a.treble + a.energy) * 34) | 0;
      const aspect = s.bw / s.bh;
      const juliaCx = -0.7 + Math.sin(api.ts * 0.00019) * (0.16 + a.mid * 0.2);
      const juliaCy = 0.27 + Math.cos(api.ts * 0.00021) * (0.16 + a.treble * 0.18);
      for (let x = 0; x < s.bw; x++) {
        const nx = (x / s.bw - 0.5) * s.zoom * aspect + s.centerX;
        const ny = (row / s.bh - 0.5) * s.zoom + s.centerY;
        let zx;
        let zy;
        let cx;
        let cy;
        if (s.mode === 'julia') {
          zx = nx;
          zy = ny;
          cx = juliaCx;
          cy = juliaCy;
        } else {
          zx = 0;
          zy = 0;
          cx = nx;
          cy = ny;
        }
        let iter = 0;
        let zx2 = 0;
        let zy2 = 0;
        while (iter < maxIter && zx2 + zy2 <= 4) {
          zy = 2 * zx * zy + cy;
          zx = zx2 - zy2 + cx;
          zx2 = zx * zx;
          zy2 = zy * zy;
          iter++;
        }
        const idx = (row * s.bw + x) * 4;
        if (iter >= maxIter) {
          data[idx] = 10;
          data[idx + 1] = 9;
          data[idx + 2] = 8;
          data[idx + 3] = 255;
          continue;
        }
        const mu = iter + 1 - Math.log2(Math.log2(Math.max(4.0001, zx2 + zy2)));
        const n = clamp(mu / maxIter, 0, 1);
        const hue = 28 + a.treble * 140 + n * 35;
        const sat = 0.55 + a.mid * 0.3;
        const val = Math.pow(n, 0.7) * (0.35 + a.energy * 0.8);
        const rgb = hsvToRgb(hue / 360, sat, val);
        data[idx] = rgb[0];
        data[idx + 1] = rgb[1];
        data[idx + 2] = rgb[2];
        data[idx + 3] = 255;
      }
    }
    return { resize, draw };
  }
  function createLambdaSketch() {
    const s = {
      nodes: [],
      edges: [],
      nextId: 1,
      spawnAcc: 0,
      reduceCooldown: 0,
      lastStem: ''
    };
    function resize() {
      // Keep dynamic graph; no hard reset on resize.
    }
    function draw(api) {
      if (api.stem !== s.lastStem) {
        s.nodes = [];
        s.edges = [];
        s.nextId = 1;
        s.spawnAcc = 0;
        s.reduceCooldown = 0;
        s.lastStem = api.stem;
      }
      const a = api.audio;
      s.spawnAcc += api.dt * (0.8 + a.energy * 2.5);
      while (s.spawnAcc >= 1) {
        spawnNode(api.w);
        s.spawnAcc -= 1;
      }
      s.reduceCooldown -= api.dt;
      if ((a.transient > 0.08 || a.beat) && s.reduceCooldown <= 0 && s.nodes.length >= 2) {
        reducePair(api);
        s.reduceCooldown = 0.12 + Math.random() * 0.18;
      }
      const byId = new Map();
      for (let i = s.nodes.length - 1; i >= 0; i--) {
        const n = s.nodes[i];
        if (n.alive) {
          n.x += n.vx * api.dt;
          n.y += (22 + a.mid * 50) * api.dt;
          n.glow = Math.max(0, n.glow - api.dt * 1.8);
          n.alpha = Math.min(1, n.alpha + api.dt * 2.4);
        } else {
          n.alpha -= api.dt * 2.2;
        }
        if (n.alpha <= 0 || n.y > api.h + 40) {
          s.nodes.splice(i, 1);
          continue;
        }
        byId.set(n.id, n);
      }
      for (let i = s.edges.length - 1; i >= 0; i--) {
        const e = s.edges[i];
        e.life -= api.dt * 0.16;
        if (e.life <= 0 || !byId.has(e.a) || !byId.has(e.b)) {
          s.edges.splice(i, 1);
        }
      }
      const c = api.ctx;
      c.save();
      c.globalCompositeOperation = 'lighter';
      c.lineWidth = 1;
      for (let i = 0; i < s.edges.length; i++) {
        const e = s.edges[i];
        const aNode = byId.get(e.a);
        const bNode = byId.get(e.b);
        if (!aNode || !bNode) continue;
        const alpha = Math.min(aNode.alpha, bNode.alpha) * 0.18 * e.life;
        c.strokeStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${alpha})`;
        c.beginPath();
        c.moveTo(aNode.x, aNode.y);
        c.lineTo(bNode.x, bNode.y);
        c.stroke();
      }
      let reducedNode = null;
      let aliveCount = 0;
      for (let i = 0; i < s.nodes.length; i++) {
        const n = s.nodes[i];
        if (!n.alive) continue;
        aliveCount++;
        if (!reducedNode || n.y > reducedNode.y) reducedNode = n;
      }
      for (let i = 0; i < s.nodes.length; i++) {
        const n = s.nodes[i];
        const glow = n.glow + (reducedNode === n && aliveCount <= 3 ? 0.8 : 0);
        const radius = n.size + glow * 5;
        c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.2 * n.alpha})`;
        c.beginPath();
        c.arc(n.x, n.y, radius, 0, TAU);
        c.fill();
        c.fillStyle = `rgba(255,245,210,${0.5 * n.alpha + glow * 0.2})`;
        c.beginPath();
        c.arc(n.x, n.y, Math.max(1.4, n.size * 0.6), 0, TAU);
        c.fill();
      }
      c.restore();
    }
    function spawnNode(w) {
      const node = {
        id: s.nextId++,
        x: w * (0.2 + Math.random() * 0.6),
        y: -20 - Math.random() * 30,
        vx: (Math.random() - 0.5) * 18,
        size: 3 + Math.random() * 4,
        glow: 0,
        alive: true,
        alpha: 0
      };
      s.nodes.push(node);
      if (s.nodes.length > 2 && Math.random() < 0.7) {
        const parent = s.nodes[(Math.random() * (s.nodes.length - 1)) | 0];
        if (parent && parent.id !== node.id) {
          s.edges.push({ a: parent.id, b: node.id, life: 1 });
        }
      }
      if (s.nodes.length > 180) s.nodes.splice(0, s.nodes.length - 180);
    }
    function reducePair(api) {
      let bestI = -1;
      let bestJ = -1;
      let bestDist = Infinity;
      for (let i = 0; i < s.nodes.length; i++) {
        const a = s.nodes[i];
        if (!a.alive || a.y < api.h * 0.15) continue;
        for (let j = i + 1; j < s.nodes.length; j++) {
          const b = s.nodes[j];
          if (!b.alive || b.y < api.h * 0.15) continue;
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const d = dx * dx + dy * dy;
          if (d < bestDist && d < 82 * 82) {
            bestDist = d;
            bestI = i;
            bestJ = j;
          }
        }
      }
      if (bestI < 0) return;
      const a = s.nodes[bestI];
      const b = s.nodes[bestJ];
      a.alive = false;
      b.alive = false;
      a.glow = 1;
      b.glow = 1;
      const merged = {
        id: s.nextId++,
        x: (a.x + b.x) * 0.5,
        y: Math.max(a.y, b.y) + 12,
        vx: (a.vx + b.vx) * 0.2,
        size: Math.max(a.size, b.size) + 0.8,
        glow: 1,
        alive: true,
        alpha: 0
      };
      s.nodes.push(merged);
      s.edges.push({ a: a.id, b: merged.id, life: 1 });
      s.edges.push({ a: b.id, b: merged.id, life: 1 });
    }
    return { resize, draw };
  }
  function createSortSketch() {
    const s = {
      mode: 'bubble',
      n: 0,
      barW: 0,
      values: null,
      glow: null,
      ghost: [],
      stepAcc: 0,
      bubbleI: 0,
      bubbleJ: 0,
      bubbleSwaps: 0,
      quickStack: [],
      quick: null,
      mergeWidth: 1,
      mergeLeft: 0,
      mergeJob: null,
      lastStem: ''
    };
    function resize(w) {
      s.n = clamp(Math.floor(w / 9), 46, 168);
      s.barW = w / s.n;
      s.values = new Float32Array(s.n);
      s.glow = new Float32Array(s.n);
      for (let i = 0; i < s.n; i++) {
        s.values[i] = (i + 1) / s.n;
      }
      randomize(0.65);
      resetState();
    }
    function draw(api) {
      if (!s.values || api.stem !== s.lastStem) {
        s.mode = modeFromStem(api.stem);
        s.lastStem = api.stem;
        resize(api.w);
      }
      s.barW = api.w / s.n;
      const a = api.audio;
      s.stepAcc += api.dt * (9 + a.bass * 72 + a.energy * 48);
      if (a.beat) s.stepAcc += 7 + a.transient * 36;
      let guard = 0;
      while (s.stepAcc >= 1 && guard < 220) {
        stepSort(a);
        s.stepAcc -= 1;
        guard++;
      }
      for (let i = 0; i < s.n; i++) {
        s.glow[i] = Math.max(0, s.glow[i] - api.dt * (2 + a.mid * 1.8));
      }
      for (let i = s.ghost.length - 1; i >= 0; i--) {
        const g = s.ghost[i];
        g.life -= api.dt * 3.2;
        if (g.life <= 0) s.ghost.splice(i, 1);
      }
      const c = api.ctx;
      const top = api.h * 0.14;
      const baseY = api.h * 0.92;
      const maxH = api.h * 0.74;
      c.save();
      c.globalCompositeOperation = 'lighter';
      c.fillStyle = 'rgba(205,195,170,0.08)';
      c.fillRect(0, baseY, api.w, 1);
      for (let i = 0; i < s.ghost.length; i++) {
        const g = s.ghost[i];
        const x0 = (g.from + 0.5) * s.barW;
        const x1 = (g.to + 0.5) * s.barW;
        const y = baseY - g.v * maxH;
        c.strokeStyle = `rgba(244,225,184,${g.life * (0.12 + a.treble * 0.2)})`;
        c.lineWidth = 1;
        c.beginPath();
        c.moveTo(x0, y);
        c.lineTo(x1, y);
        c.stroke();
      }
      for (let i = 0; i < s.n; i++) {
        const x = i * s.barW;
        const v = s.values[i];
        const h = v * maxH;
        const y = baseY - h;
        const glow = s.glow[i];
        c.fillStyle = `rgba(180,168,138,${0.06 + v * 0.12})`;
        c.fillRect(x + 1, top, Math.max(1, s.barW - 2), baseY - top);
        c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.14 + a.energy * 0.22 + glow * 0.45})`;
        c.fillRect(x + 1, y, Math.max(1, s.barW - 2), h);
        c.fillStyle = `rgba(255,242,204,${0.06 + glow * (0.35 + a.treble * 0.35)})`;
        c.fillRect(x + 1, y - 1.3, Math.max(1, s.barW - 2), 1.3);
      }
      if (s.mode === 'quick' && s.quick) {
        c.strokeStyle = `rgba(255,229,170,${0.18 + a.energy * 0.28})`;
        c.lineWidth = 1;
        const q = s.quick;
        const x0 = q.lo * s.barW;
        const x1 = (q.hi + 1) * s.barW;
        c.strokeRect(x0, top, x1 - x0, baseY - top);
        c.fillStyle = `rgba(255,245,215,${0.3 + a.treble * 0.35})`;
        c.fillRect(q.pivot * s.barW + 1, top, Math.max(2, s.barW - 2), baseY - top);
      }
      if (s.mode === 'merge' && s.mergeJob) {
        const m = s.mergeJob;
        const x0 = m.left * s.barW;
        const x1 = m.right * s.barW;
        c.fillStyle = `rgba(210,190,145,${0.05 + a.mid * 0.2})`;
        c.fillRect(x0, top, x1 - x0, baseY - top);
      }
      c.restore();
    }
    function stepSort(audio) {
      if (s.mode === 'quick') stepQuick(audio);
      else if (s.mode === 'merge') stepMerge(audio);
      else stepBubble(audio);
    }
    function stepBubble(audio) {
      if (s.bubbleI >= s.n - 1) {
        randomize(0.2 + audio.transient * 0.6);
        s.bubbleI = 0;
        s.bubbleJ = 0;
        s.bubbleSwaps = 0;
        return;
      }
      const j = s.bubbleJ;
      const k = j + 1;
      if (k >= s.n - s.bubbleI) {
        if (s.bubbleSwaps === 0) {
          randomize(0.1 + audio.energy * 0.3);
          s.bubbleI = 0;
        } else {
          s.bubbleI++;
        }
        s.bubbleJ = 0;
        s.bubbleSwaps = 0;
        return;
      }
      s.glow[j] = 1;
      s.glow[k] = 1;
      const shouldSwap = s.values[j] > s.values[k] || Math.random() < audio.bass * 0.035;
      if (shouldSwap) {
        swapBars(j, k, 1);
        s.bubbleSwaps++;
      }
      if (audio.beat && Math.random() < 0.26) {
        const idx = (Math.random() * s.n) | 0;
        s.glow[idx] = 1;
      }
      s.bubbleJ++;
    }
    function stepQuick(audio) {
      if (!s.quick) {
        while (s.quickStack.length) {
          const seg = s.quickStack.pop();
          if (seg && seg[1] - seg[0] >= 1) {
            s.quick = { lo: seg[0], hi: seg[1], i: seg[0], j: seg[0], pivot: seg[1] };
            break;
          }
        }
        if (!s.quick) {
          randomize(0.24 + audio.energy * 0.4);
          s.quickStack.push([0, s.n - 1]);
        }
        return;
      }
      const q = s.quick;
      s.glow[q.pivot] = 1;
      if (q.j < q.hi) {
        s.glow[q.j] = 1;
        const pivotV = s.values[q.pivot];
        if (s.values[q.j] <= pivotV || Math.random() < audio.transient * 0.09) {
          if (q.i !== q.j) swapBars(q.i, q.j, 0.9);
          q.i++;
        }
        q.j++;
        return;
      }
      if (q.i !== q.pivot) swapBars(q.i, q.pivot, 1);
      const p = q.i;
      if (p - 1 > q.lo) s.quickStack.push([q.lo, p - 1]);
      if (p + 1 < q.hi) s.quickStack.push([p + 1, q.hi]);
      s.quick = null;
      if (audio.beat) {
        const lo = clamp(p - 2, 0, s.n - 1);
        const hi = clamp(p + 2, 0, s.n - 1);
        for (let i = lo; i <= hi; i++) s.glow[i] = 1;
      }
    }
    function stepMerge(audio) {
      if (!s.mergeJob) {
        if (s.mergeLeft >= s.n) {
          s.mergeLeft = 0;
          s.mergeWidth *= 2;
          if (s.mergeWidth >= s.n) {
            randomize(0.32 + audio.bass * 0.4);
            s.mergeWidth = 1;
          }
        }
        const left = s.mergeLeft;
        const mid = Math.min(left + s.mergeWidth, s.n);
        const right = Math.min(left + s.mergeWidth * 2, s.n);
        if (mid >= right) {
          s.mergeLeft += s.mergeWidth * 2;
          return;
        }
        s.mergeJob = {
          left,
          right,
          k: left,
          temp: Array.from(s.values.slice(left, right)),
          midOff: mid - left,
          li: 0,
          ri: mid - left,
          end: right - left
        };
      }
      const m = s.mergeJob;
      const takeLeft = m.li < m.midOff && (m.ri >= m.end || m.temp[m.li] <= m.temp[m.ri] || Math.random() < audio.bass * 0.03);
      const from = takeLeft ? m.li++ : m.ri++;
      s.values[m.k] = m.temp[from];
      s.glow[m.k] = 1;
      if (!takeLeft) s.glow[m.left + from] = Math.max(s.glow[m.left + from], 0.35);
      m.k++;
      if (m.k >= m.right) {
        s.mergeLeft += s.mergeWidth * 2;
        s.mergeJob = null;
      }
    }
    function swapBars(i, j, glow) {
      const vi = s.values[i];
      const vj = s.values[j];
      s.values[i] = vj;
      s.values[j] = vi;
      s.glow[i] = Math.max(s.glow[i], glow);
      s.glow[j] = Math.max(s.glow[j], glow);
      s.ghost.push({ from: i, to: j, v: vj, life: 1 });
      s.ghost.push({ from: j, to: i, v: vi, life: 1 });
      if (s.ghost.length > 80) s.ghost.splice(0, s.ghost.length - 80);
    }
    function randomize(strength) {
      if (!s.values) return;
      const swaps = Math.max(6, Math.floor(s.n * (0.3 + strength)));
      for (let n = 0; n < swaps; n++) {
        const i = (Math.random() * s.n) | 0;
        const j = (Math.random() * s.n) | 0;
        const tmp = s.values[i];
        s.values[i] = s.values[j];
        s.values[j] = tmp;
      }
      for (let i = 0; i < s.n; i++) s.glow[i] = Math.max(s.glow[i], strength * 0.35);
    }
    function resetState() {
      s.stepAcc = 0;
      s.bubbleI = 0;
      s.bubbleJ = 0;
      s.bubbleSwaps = 0;
      s.quickStack = [[0, s.n - 1]];
      s.quick = null;
      s.mergeWidth = 1;
      s.mergeLeft = 0;
      s.mergeJob = null;
      s.ghost = [];
    }
    function modeFromStem(stem) {
      if (/quick/i.test(stem || '')) return 'quick';
      if (/merge/i.test(stem || '')) return 'merge';
      return 'bubble';
    }
    return { resize, draw };
  }
  function createConcurrencySketch() {
    const s = {
      mode: 'dining',
      dining: null,
      race: null,
      mutex: null,
      lockX: 0,
      lastStem: ''
    };
    function resize(w, h) {
      s.lockX = w * 0.58;
      s.dining = {
        philosophers: new Array(5).fill(0).map(() => ({
          state: 'thinking',
          timer: 0.3 + Math.random() * 1.5,
          pulse: 0,
          wait: 0
        })),
        forks: new Int8Array(5).fill(-1),
        deadlock: 0
      };
      s.race = {
        threads: new Array(3).fill(0).map((_, lane) => ({
          lane,
          progress: Math.random() * 0.26,
          speed: 0.18 + Math.random() * 0.26,
          flash: 0
        })),
        contention: 0,
        glitch: 0
      };
      s.mutex = {
        threads: new Array(4).fill(0).map((_, lane) => ({
          lane,
          phase: 'rest',
          timer: 0.3 + Math.random() * 1.2,
          x: w * 0.16,
          pulse: 0,
          criticalDur: 0.6
        })),
        lockOwner: -1,
        queue: [],
        lockPulse: 0
      };
      for (let i = 0; i < s.mutex.threads.length; i++) {
        s.mutex.threads[i].x = w * (0.13 + i * 0.05);
      }
      void h;
    }
    function draw(api) {
      if (!s.dining || api.stem !== s.lastStem) {
        s.mode = modeFromStem(api.stem);
        s.lastStem = api.stem;
        resize(api.w, api.h);
      }
      if (s.mode === 'race') {
        updateRace(api);
        drawRace(api);
      } else if (s.mode === 'mutex') {
        updateMutex(api);
        drawMutex(api);
      } else {
        updateDining(api);
        drawDining(api);
      }
    }
    function updateDining(api) {
      const d = s.dining;
      const a = api.audio;
      let eating = 0;
      let hungry = 0;
      for (let i = 0; i < d.philosophers.length; i++) {
        const p = d.philosophers[i];
        p.pulse = Math.max(0, p.pulse - api.dt * 2.8);
        if (p.state === 'thinking') {
          p.timer -= api.dt * (0.8 + a.mid * 1.4);
          if (p.timer <= 0) {
            p.state = 'hungry';
            p.timer = 0.5 + Math.random() * 0.9;
            p.wait = 0;
          }
        } else if (p.state === 'hungry') {
          hungry++;
          p.wait += api.dt;
          if (tryAcquireForks(i)) {
            p.state = 'eating';
            p.timer = 0.35 + (0.4 + a.bass) * (0.65 + Math.random() * 0.4);
            p.pulse = 1;
          }
        } else {
          eating++;
          p.timer -= api.dt * (0.7 + a.energy * 0.7);
          if (p.timer <= 0) {
            releaseForks(i);
            p.state = 'thinking';
            p.timer = 0.5 + Math.random() * 1.2;
          }
        }
      }
      if (eating === 0 && hungry === d.philosophers.length) d.deadlock += api.dt;
      else d.deadlock = Math.max(0, d.deadlock - api.dt * 2.2);
      if (a.beat && d.deadlock > 1.3) {
        const pick = (Math.random() * d.philosophers.length) | 0;
        releaseForks(pick);
        d.philosophers[pick].state = 'thinking';
        d.philosophers[pick].timer = 0.45;
        d.deadlock *= 0.35;
      }
    }
    function drawDining(api) {
      const c = api.ctx;
      const a = api.audio;
      const d = s.dining;
      const cx = api.w * 0.5;
      const cy = api.h * 0.56;
      const tableR = Math.min(api.w, api.h) * 0.16;
      c.save();
      c.globalCompositeOperation = 'lighter';
      c.fillStyle = `rgba(140,128,104,${0.08 + a.energy * 0.12})`;
      c.beginPath();
      c.arc(cx, cy, tableR, 0, TAU);
      c.fill();
      for (let f = 0; f < 5; f++) {
        const ang = -Math.PI / 2 + (f + 0.5) * (TAU / 5);
        const fx = cx + Math.cos(ang) * tableR * 0.92;
        const fy = cy + Math.sin(ang) * tableR * 0.92;
        const held = d.forks[f] >= 0;
        c.strokeStyle = held
          ? `rgba(255,238,188,${0.32 + a.treble * 0.4})`
          : 'rgba(180,168,142,0.14)';
        c.lineWidth = held ? 2.1 : 1.2;
        c.beginPath();
        c.moveTo(fx - Math.sin(ang) * 6, fy + Math.cos(ang) * 6);
        c.lineTo(fx + Math.sin(ang) * 6, fy - Math.cos(ang) * 6);
        c.stroke();
      }
      for (let i = 0; i < d.philosophers.length; i++) {
        const p = d.philosophers[i];
        const ang = -Math.PI / 2 + i * (TAU / 5);
        const px = cx + Math.cos(ang) * tableR * 1.48;
        const py = cy + Math.sin(ang) * tableR * 1.48;
        const hungry = p.state === 'hungry';
        const eating = p.state === 'eating';
        const alpha = eating ? 0.44 + p.pulse * 0.4 : hungry ? 0.22 + p.wait * 0.12 : 0.13;
        c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${alpha})`;
        c.beginPath();
        c.arc(px, py, 9 + p.pulse * 7, 0, TAU);
        c.fill();
        if (hungry) {
          c.strokeStyle = `rgba(255,232,180,${0.08 + Math.min(0.32, p.wait * 0.14)})`;
          c.lineWidth = 1;
          c.beginPath();
          c.arc(px, py, 14 + p.wait * 8, 0, TAU);
          c.stroke();
        }
      }
      if (d.deadlock > 0.35) {
        c.strokeStyle = `rgba(255,211,140,${Math.min(0.42, d.deadlock * 0.24)})`;
        c.lineWidth = 2;
        c.beginPath();
        c.arc(cx, cy, tableR * (1.08 + d.deadlock * 0.1), 0, TAU);
        c.stroke();
      }
      c.restore();
    }
    function tryAcquireForks(i) {
      const d = s.dining;
      const left = i;
      const right = (i + 1) % d.forks.length;
      if (d.forks[left] !== -1 || d.forks[right] !== -1) return false;
      d.forks[left] = i;
      d.forks[right] = i;
      return true;
    }
    function releaseForks(i) {
      const d = s.dining;
      const left = i;
      const right = (i + 1) % d.forks.length;
      if (d.forks[left] === i) d.forks[left] = -1;
      if (d.forks[right] === i) d.forks[right] = -1;
    }
    function updateRace(api) {
      const r = s.race;
      const a = api.audio;
      const critical0 = 0.64;
      const critical1 = 0.78;
      let colliding = 0;
      for (let i = 0; i < r.threads.length; i++) {
        const t = r.threads[i];
        t.flash = Math.max(0, t.flash - api.dt * 3);
        const burst = a.beat && i === ((api.ts * 0.001) | 0) % r.threads.length ? 0.8 : 0;
        t.progress += api.dt * (0.08 + t.speed * (0.7 + a.bass * 0.9 + burst));
        if (t.progress >= 1.04) {
          t.progress = Math.random() * 0.06;
          t.speed = 0.18 + Math.random() * 0.3;
          t.flash = 1;
        }
        if (t.progress > critical0 && t.progress < critical1) colliding++;
      }
      if (colliding > 1) {
        r.contention = Math.min(1, r.contention + api.dt * (1.8 + a.transient * 8));
        r.glitch = Math.min(1, r.glitch + api.dt * 3.2);
      } else {
        r.contention = Math.max(0, r.contention - api.dt * 2.4);
        r.glitch = Math.max(0, r.glitch - api.dt * 4.2);
      }
    }
    function drawRace(api) {
      const c = api.ctx;
      const a = api.audio;
      const r = s.race;
      const margin = api.w * 0.1;
      const targetX = api.w * 0.78;
      const laneGap = api.h * 0.15;
      const y0 = api.h * 0.36;
      c.save();
      c.globalCompositeOperation = 'lighter';
      c.fillStyle = `rgba(200,180,145,${0.08 + r.contention * 0.28})`;
      c.fillRect(targetX - 7, api.h * 0.22, 14, api.h * 0.56);
      for (let i = 0; i < r.threads.length; i++) {
        const y = y0 + i * laneGap;
        c.strokeStyle = 'rgba(175,163,139,0.1)';
        c.lineWidth = 1;
        c.beginPath();
        c.moveTo(margin, y);
        c.lineTo(targetX, api.h * 0.5);
        c.stroke();
        const t = r.threads[i];
        const x = margin + t.progress * (targetX - margin);
        const jitter = r.glitch > 0 ? (Math.random() - 0.5) * 15 * r.glitch : 0;
        c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.25 + t.flash * 0.5 + a.energy * 0.2})`;
        c.beginPath();
        c.arc(x + jitter, y + jitter * 0.2, 6 + t.flash * 3, 0, TAU);
        c.fill();
      }
      if (r.contention > 0.04) {
        for (let i = 0; i < 12; i++) {
          const yy = api.h * (0.28 + Math.random() * 0.44);
          const dx = (Math.random() - 0.5) * 24 * r.glitch;
          c.strokeStyle = `rgba(255,232,176,${0.12 + r.contention * 0.35})`;
          c.beginPath();
          c.moveTo(targetX - 10 + dx, yy);
          c.lineTo(targetX + 10 - dx, yy + (Math.random() - 0.5) * 12);
          c.stroke();
        }
      }
      c.restore();
    }
    function updateMutex(api) {
      const m = s.mutex;
      const a = api.audio;
      for (let i = 0; i < m.threads.length; i++) {
        const t = m.threads[i];
        t.pulse = Math.max(0, t.pulse - api.dt * 2.4);
        if (t.phase === 'rest') {
          t.timer -= api.dt * (0.7 + a.mid * 1.1);
          if (t.timer <= 0) {
            t.phase = 'wait';
            t.pulse = 1;
            if (!m.queue.includes(i)) m.queue.push(i);
          }
        } else if (t.phase === 'critical') {
          t.timer -= api.dt * (0.85 + a.bass * 0.85);
          if (t.timer <= 0) {
            t.phase = 'rest';
            t.timer = 0.35 + Math.random() * 1.2;
            if (m.lockOwner === i) m.lockOwner = -1;
          }
        }
      }
      if (m.lockOwner < 0 && m.queue.length) {
        const next = m.queue.shift();
        const t = m.threads[next];
        t.phase = 'critical';
        t.criticalDur = 0.38 + Math.random() * 0.46;
        t.timer = t.criticalDur;
        t.pulse = 1;
        m.lockOwner = next;
        m.lockPulse = 1;
      }
      m.lockPulse = Math.max(0, m.lockPulse - api.dt * 2.8);
      for (let i = 0; i < m.threads.length; i++) {
        const t = m.threads[i];
        const laneY = api.h * (0.28 + i * 0.16);
        void laneY;
        let tx = api.w * 0.16;
        if (t.phase === 'wait') tx = s.lockX - 18;
        else if (t.phase === 'critical') {
          const p = 1 - t.timer / Math.max(0.001, t.criticalDur);
          tx = s.lockX + 12 + p * 74;
        }
        t.x += (tx - t.x) * Math.min(1, api.dt * 8.2);
      }
      if (a.beat && Math.random() < 0.35) {
        const i = (Math.random() * m.threads.length) | 0;
        const t = m.threads[i];
        if (t.phase === 'rest' && !m.queue.includes(i)) {
          t.phase = 'wait';
          t.timer = 0.1;
          m.queue.push(i);
        }
      }
    }
    function drawMutex(api) {
      const c = api.ctx;
      const a = api.audio;
      const m = s.mutex;
      c.save();
      c.globalCompositeOperation = 'lighter';
      c.fillStyle = `rgba(198,175,136,${0.1 + m.lockPulse * 0.3})`;
      c.fillRect(s.lockX - 9, api.h * 0.18, 18, api.h * 0.64);
      for (let i = 0; i < m.threads.length; i++) {
        const t = m.threads[i];
        const y = api.h * (0.28 + i * 0.16);
        c.strokeStyle = 'rgba(175,163,139,0.08)';
        c.lineWidth = 1;
        c.beginPath();
        c.moveTo(api.w * 0.12, y);
        c.lineTo(api.w * 0.86, y);
        c.stroke();
        c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.16 + a.energy * 0.16 + t.pulse * 0.45})`;
        c.beginPath();
        c.arc(t.x, y, 5 + t.pulse * 2.5, 0, TAU);
        c.fill();
      }
      for (let q = 0; q < m.queue.length; q++) {
        const idx = m.queue[q];
        const y = api.h * (0.28 + idx * 0.16);
        c.fillStyle = `rgba(255,235,190,${0.08 + (1 - q / Math.max(1, m.queue.length)) * 0.22})`;
        c.fillRect(s.lockX - 30 - q * 7, y - 1, 5, 2);
      }
      c.restore();
    }
    function modeFromStem(stem) {
      if (/race/i.test(stem || '')) return 'race';
      if (/mutex/i.test(stem || '')) return 'mutex';
      return 'dining';
    }
    return { resize, draw };
  }
  function createGraphSketch() {
    const s = {
      mode: 'dijkstra',
      nodes: [],
      edges: [],
      adj: [],
      heat: [],
      visited: [],
      dist: [],
      prev: [],
      settled: [],
      queue: [],
      stack: [],
      open: [],
      current: -1,
      source: 0,
      pulses: [],
      stepAcc: 0,
      lastStem: ''
    };
    function resize(w, h) {
      initGraph(w, h);
      resetTraversal();
    }
    function draw(api) {
      if (!s.nodes.length || api.stem !== s.lastStem) {
        s.mode = modeFromStem(api.stem);
        s.lastStem = api.stem;
        resize(api.w, api.h);
      }
      const a = api.audio;
      simulateLayout(api);
      s.stepAcc += api.dt * (1.6 + a.energy * 4.6 + a.bass * 3.6);
      if (a.beat) s.stepAcc += 1.2 + a.transient * 6;
      let guard = 0;
      while (s.stepAcc >= 1 && guard < 16) {
        traversalStep(a);
        s.stepAcc -= 1;
        guard++;
      }
      for (let i = 0; i < s.heat.length; i++) {
        s.heat[i] = Math.max(0, s.heat[i] - api.dt * 0.62);
      }
      for (let i = s.pulses.length - 1; i >= 0; i--) {
        s.pulses[i].life -= api.dt * 2.2;
        if (s.pulses[i].life <= 0) s.pulses.splice(i, 1);
      }
      const c = api.ctx;
      c.save();
      c.globalCompositeOperation = 'lighter';
      for (let ei = 0; ei < s.edges.length; ei++) {
        const e = s.edges[ei];
        const aNode = s.nodes[e.a];
        const bNode = s.nodes[e.b];
        const seen = s.visited[e.a] || s.visited[e.b];
        c.strokeStyle = seen ? 'rgba(196,176,142,0.14)' : 'rgba(150,148,142,0.08)';
        c.lineWidth = 1;
        c.beginPath();
        c.moveTo(aNode.x, aNode.y);
        c.lineTo(bNode.x, bNode.y);
        c.stroke();
      }
      for (let i = 0; i < s.pulses.length; i++) {
        const p = s.pulses[i];
        const e = s.edges[p.edge];
        if (!e) continue;
        const aNode = s.nodes[e.a];
        const bNode = s.nodes[e.b];
        c.strokeStyle = `rgba(255,230,175,${p.life * (0.22 + p.power * 0.35)})`;
        c.lineWidth = 1.4 + p.power * 1.2;
        c.beginPath();
        c.moveTo(aNode.x, aNode.y);
        c.lineTo(bNode.x, bNode.y);
        c.stroke();
      }
      for (let i = 0; i < s.nodes.length; i++) {
        const n = s.nodes[i];
        const heat = s.heat[i];
        const active = s.current === i;
        const base = s.visited[i] ? 0.2 : 0.08;
        c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${base + heat * 0.45 + (active ? 0.3 : 0)})`;
        c.beginPath();
        c.arc(n.x, n.y, 3.5 + heat * 4 + (active ? 3 : 0), 0, TAU);
        c.fill();
      }
      c.restore();
    }
    function initGraph(w, h) {
      const count = clamp(Math.floor((w * h) / 29000), 18, 40);
      s.nodes = [];
      s.edges = [];
      s.adj = new Array(count);
      for (let i = 0; i < count; i++) s.adj[i] = [];
      for (let i = 0; i < count; i++) {
        s.nodes.push({
          x: w * (0.18 + Math.random() * 0.64),
          y: h * (0.2 + Math.random() * 0.6),
          vx: 0,
          vy: 0
        });
      }
      const maxDist = Math.min(w, h) * 0.34;
      const maxDistSq = maxDist * maxDist;
      for (let i = 0; i < count; i++) {
        for (let j = i + 1; j < count; j++) {
          const dx = s.nodes[i].x - s.nodes[j].x;
          const dy = s.nodes[i].y - s.nodes[j].y;
          const d2 = dx * dx + dy * dy;
          if (d2 > maxDistSq) continue;
          const near = 1 - d2 / maxDistSq;
          if (Math.random() < 0.06 + near * 0.25) addEdge(i, j, 0.8 + Math.sqrt(d2) / 80);
        }
      }
      for (let i = 0; i < count; i++) {
        if (s.adj[i].length > 0) continue;
        let best = -1;
        let bestD = Infinity;
        for (let j = 0; j < count; j++) {
          if (i === j) continue;
          const dx = s.nodes[i].x - s.nodes[j].x;
          const dy = s.nodes[i].y - s.nodes[j].y;
          const d2 = dx * dx + dy * dy;
          if (d2 < bestD) {
            bestD = d2;
            best = j;
          }
        }
        if (best >= 0) addEdge(i, best, 0.8 + Math.sqrt(bestD) / 80);
      }
      s.heat = new Array(count).fill(0);
      s.visited = new Array(count).fill(0);
      s.dist = new Array(count).fill(Infinity);
      s.prev = new Array(count).fill(-1);
      s.settled = new Array(count).fill(0);
      s.queue = [];
      s.stack = [];
      s.open = [];
      s.pulses = [];
      s.current = -1;
      s.stepAcc = 0;
    }
    function addEdge(a, b, w) {
      for (let i = 0; i < s.adj[a].length; i++) {
        if (s.adj[a][i].to === b) return;
      }
      const idx = s.edges.length;
      s.edges.push({ a, b, w });
      s.adj[a].push({ to: b, edge: idx });
      s.adj[b].push({ to: a, edge: idx });
    }
    function simulateLayout(api) {
      const a = api.audio;
      const nodes = s.nodes;
      const repulse = 5400 * (0.7 + a.treble * 0.8);
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const n0 = nodes[i];
          const n1 = nodes[j];
          let dx = n1.x - n0.x;
          let dy = n1.y - n0.y;
          const d2 = dx * dx + dy * dy + 60;
          const f = repulse / d2;
          const inv = 1 / Math.sqrt(d2);
          dx *= inv;
          dy *= inv;
          n0.vx -= dx * f * api.dt;
          n0.vy -= dy * f * api.dt;
          n1.vx += dx * f * api.dt;
          n1.vy += dy * f * api.dt;
        }
      }
      for (let i = 0; i < s.edges.length; i++) {
        const e = s.edges[i];
        const n0 = nodes[e.a];
        const n1 = nodes[e.b];
        const dx = n1.x - n0.x;
        const dy = n1.y - n0.y;
        const d = Math.sqrt(dx * dx + dy * dy) + 0.001;
        const target = 70 + e.w * 8;
        const pull = (d - target) * (0.006 + a.energy * 0.007);
        const ux = dx / d;
        const uy = dy / d;
        n0.vx += ux * pull;
        n0.vy += uy * pull;
        n1.vx -= ux * pull;
        n1.vy -= uy * pull;
      }
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        n.vx += (api.w * 0.5 - n.x) * api.dt * 0.06;
        n.vy += (api.h * 0.5 - n.y) * api.dt * 0.06;
        n.vx *= 0.92;
        n.vy *= 0.92;
        n.x += n.vx * 34 * api.dt;
        n.y += n.vy * 34 * api.dt;
        n.x = clamp(n.x, 26, api.w - 26);
        n.y = clamp(n.y, 26, api.h - 26);
      }
    }
    function traversalStep(audio) {
      if (s.mode === 'bfs') stepBfs();
      else if (s.mode === 'dfs') stepDfs();
      else stepDijkstra(audio);
    }
    function stepBfs() {
      if (!s.queue.length) {
        resetTraversal();
        return;
      }
      const node = s.queue.shift();
      s.current = node;
      s.heat[node] = 1;
      const neighbors = s.adj[node];
      for (let i = 0; i < neighbors.length; i++) {
        const n = neighbors[i];
        if (s.visited[n.to]) continue;
        s.visited[n.to] = 1;
        s.queue.push(n.to);
        s.prev[n.to] = node;
        pulseEdge(n.edge, 0.7);
      }
    }
    function stepDfs() {
      if (!s.stack.length) {
        resetTraversal();
        return;
      }
      const node = s.current < 0 ? s.stack[s.stack.length - 1] : s.current;
      const neighbors = s.adj[node];
      let picked = null;
      for (let i = 0; i < neighbors.length; i++) {
        const n = neighbors[(i + ((Math.random() * neighbors.length) | 0)) % neighbors.length];
        if (!s.visited[n.to]) {
          picked = n;
          break;
        }
      }
      if (picked) {
        s.current = picked.to;
        s.visited[picked.to] = 1;
        s.heat[picked.to] = 1;
        s.stack.push(picked.to);
        s.prev[picked.to] = node;
        pulseEdge(picked.edge, 1);
      } else if (s.stack.length > 1) {
        const from = s.stack.pop();
        const to = s.stack[s.stack.length - 1];
        s.current = to;
        const edge = edgeBetween(from, to);
        if (edge >= 0) pulseEdge(edge, 0.45);
      } else {
        resetTraversal();
      }
    }
    function stepDijkstra(audio) {
      if (!s.open.length) {
        resetTraversal();
        return;
      }
      let bestIdx = 0;
      for (let i = 1; i < s.open.length; i++) {
        if (s.dist[s.open[i]] < s.dist[s.open[bestIdx]]) bestIdx = i;
      }
      const node = s.open.splice(bestIdx, 1)[0];
      if (s.settled[node]) return;
      s.settled[node] = 1;
      s.visited[node] = 1;
      s.heat[node] = 1;
      s.current = node;
      const neighbors = s.adj[node];
      for (let i = 0; i < neighbors.length; i++) {
        const n = neighbors[i];
        if (s.settled[n.to]) continue;
        const alt = s.dist[node] + s.edges[n.edge].w * (1 + audio.treble * 0.2);
        if (alt < s.dist[n.to]) {
          s.dist[n.to] = alt;
          s.prev[n.to] = node;
          if (!s.open.includes(n.to)) s.open.push(n.to);
          pulseEdge(n.edge, 0.85);
        }
      }
    }
    function pulseEdge(edge, power) {
      s.pulses.push({ edge, life: 1, power });
      if (s.pulses.length > 120) s.pulses.splice(0, s.pulses.length - 120);
    }
    function edgeBetween(a, b) {
      const list = s.adj[a];
      for (let i = 0; i < list.length; i++) {
        if (list[i].to === b) return list[i].edge;
      }
      return -1;
    }
    function resetTraversal() {
      s.source = (Math.random() * s.nodes.length) | 0;
      s.current = s.source;
      s.visited.fill(0);
      s.dist.fill(Infinity);
      s.prev.fill(-1);
      s.settled.fill(0);
      s.queue.length = 0;
      s.stack.length = 0;
      s.open.length = 0;
      s.heat.fill(0);
      s.visited[s.source] = 1;
      s.heat[s.source] = 1;
      if (s.mode === 'bfs') {
        s.queue.push(s.source);
      } else if (s.mode === 'dfs') {
        s.stack.push(s.source);
      } else {
        s.dist[s.source] = 0;
        s.open.push(s.source);
      }
    }
    function modeFromStem(stem) {
      if (/bfs/i.test(stem || '')) return 'bfs';
      if (/dfs/i.test(stem || '')) return 'dfs';
      return 'dijkstra';
    }
    return { resize, draw };
  }
  // ── Neural Networks ─────────────────────────────────────────────────────────
  function createNeuralNetSketch() {
    // Network topology: 4 layers
    const layers = [4, 6, 6, 3]; // input, hidden1, hidden2, output
    const nodes = [];
    const edges = [];
    let nodeActivations = [];
    let gradientFlow = [];
    let lossHistory = [];
    let lossIdx = 0;
    let prevBeat = false;
    // Build node positions
    for (let l = 0; l < layers.length; l++) {
      for (let i = 0; i < layers[l]; i++) {
        nodes.push({ layer: l, idx: i, activation: 0, gradient: 0, glow: 0 });
      }
    }
    // Build edges
    let nOff = 0;
    for (let l = 0; l < layers.length - 1; l++) {
      const fromOff = nOff;
      nOff += layers[l];
      for (let i = 0; i < layers[l]; i++) {
        for (let j = 0; j < layers[l + 1]; j++) {
          edges.push({ from: fromOff + i, to: nOff + j, weight: Math.random() * 2 - 1, signal: 0 });
        }
      }
    }
    // Loss curve: 120 points starting high, decaying
    for (let i = 0; i < 120; i++) {
      const t = i / 119;
      lossHistory.push(1.0 * Math.exp(-3 * t) + 0.15 * Math.sin(t * 20) * Math.exp(-2 * t) + 0.05);
    }
    function draw(api) {
      const { ctx, w, h, audio, intensity, stem, playhead, duration } = api;
      const bass = audio.bass, mid = audio.mid, treble = audio.treble;
      const energy = audio.energy, beat = audio.beat, flux = audio.flux;
      const cx = w / 2, cy = h / 2;
      const netW = w * 0.55, netH = h * 0.55;
      const netX = cx - netW / 2, netY = cy - netH / 2 - h * 0.05;
      // Progress through piece
      const prog = duration > 0 ? playhead / duration : 0;
      // Determine sub-mode from stem
      const isBP = /backprop/i.test(stem || '');
      const isAct = /activation/i.test(stem || '');
      const isWS = /weight/i.test(stem || '');
      // Update node activations based on audio
      const nl = layers.length;
      let nIdx = 0;
      for (let l = 0; l < nl; l++) {
        for (let i = 0; i < layers[l]; i++) {
          const nd = nodes[nIdx++];
          const layerSignal = l === 0 ? bass : l === 1 ? mid : l === 2 ? treble : energy;
          const phase = (i + l * 3) * 0.7;
          nd.activation += (layerSignal * (0.5 + 0.5 * Math.sin(api.ts * 0.002 + phase)) - nd.activation) * 0.12;
          // Gradient flow (backprop mode): backward wave
          if (isBP && prog > 0.4) {
            const bpProg = Math.min(1, (prog - 0.4) / 0.4);
            const targetLayer = nl - 1 - Math.floor(bpProg * nl);
            nd.gradient += ((l === targetLayer ? flux * 2 : 0) - nd.gradient) * 0.08;
          } else {
            nd.gradient *= 0.95;
          }
          // Beat glow
          if (beat && !prevBeat) nd.glow = 0.8 * nd.activation;
          nd.glow *= 0.92;
        }
      }
      prevBeat = beat;
      // Update edges
      for (const e of edges) {
        const fromAct = nodes[e.from].activation;
        e.signal += (fromAct * Math.abs(e.weight) - e.signal) * 0.1;
        e.weight += (Math.random() - 0.5) * 0.005 * energy;
      }
      // Loss curve index
      if (duration > 0) lossIdx = Math.floor(prog * (lossHistory.length - 1));
      // ── Draw ──
      ctx.globalCompositeOperation = 'lighter';
      // Draw edges
      nIdx = 0;
      for (const e of edges) {
        const fn = nodes[e.from], tn = nodes[e.to];
        const fx = netX + (fn.layer / (nl - 1)) * netW;
        const fy = netY + ((fn.idx + 0.5) / layers[fn.layer]) * netH;
        const tx = netX + (tn.layer / (nl - 1)) * netW;
        const ty = netY + ((tn.idx + 0.5) / layers[tn.layer]) * netH;
        const alpha = Math.min(0.5, 0.05 + e.signal * 0.6);
        const grad = fn.gradient + tn.gradient;
        const r = Math.min(255, GOLD[0] + grad * 200);
        const g = Math.min(255, GOLD[1] - grad * 80);
        const b = GOLD[2];
        ctx.strokeStyle = `rgba(${r|0},${g|0},${b|0},${alpha.toFixed(3)})`;
        ctx.lineWidth = 0.5 + e.signal * 2;
        ctx.beginPath(); ctx.moveTo(fx, fy); ctx.lineTo(tx, ty); ctx.stroke();
      }
      // Draw nodes
      nIdx = 0;
      for (let l = 0; l < nl; l++) {
        for (let i = 0; i < layers[l]; i++) {
          const nd = nodes[nIdx++];
          const x = netX + (l / (nl - 1)) * netW;
          const y = netY + ((i + 0.5) / layers[l]) * netH;
          const r = 4 + nd.activation * 10 + nd.glow * 8;
          // Activation glow
          if (nd.activation > 0.1 || nd.glow > 0.05) {
            const gr = ctx.createRadialGradient(x, y, 0, x, y, r * 3);
            const a = Math.min(0.4, nd.activation * 0.3 + nd.glow * 0.5);
            gr.addColorStop(0, `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${a.toFixed(3)})`);
            gr.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = gr;
            ctx.beginPath(); ctx.arc(x, y, r * 3, 0, TAU); ctx.fill();
          }
          // Node circle
          const bright = 0.3 + nd.activation * 0.7;
          ctx.fillStyle = `rgba(${(GOLD[0]*bright)|0},${(GOLD[1]*bright)|0},${(GOLD[2]*bright)|0},0.9)`;
          ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill();
          // Gradient halo (backprop)
          if (nd.gradient > 0.05) {
            ctx.strokeStyle = `rgba(255,120,60,${Math.min(0.6, nd.gradient * 0.8).toFixed(3)})`;
            ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.arc(x, y, r + 4 + nd.gradient * 6, 0, TAU); ctx.stroke();
          }
        }
      }
      // Loss curve (bottom 20%)
      const lossY = h * 0.82, lossH = h * 0.12, lossX = w * 0.15, lossW = w * 0.7;
      ctx.strokeStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.25)`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < lossHistory.length; i++) {
        const px = lossX + (i / (lossHistory.length - 1)) * lossW;
        const py = lossY + (1 - lossHistory[i]) * lossH;
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.stroke();
      // Current position on loss curve
      if (lossIdx < lossHistory.length) {
        const px = lossX + (lossIdx / (lossHistory.length - 1)) * lossW;
        const py = lossY + (1 - lossHistory[lossIdx]) * lossH;
        ctx.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.9)`;
        ctx.beginPath(); ctx.arc(px, py, 4, 0, TAU); ctx.fill();
      }
      // Label
      ctx.globalCompositeOperation = 'source-over';
      ctx.font = '10px monospace';
      ctx.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.2)`;
      const label = isBP ? 'backpropagation' : isAct ? 'activation functions' : isWS ? 'weight space' : 'neural network';
      ctx.fillText(label, 12, h - 12);
    }
    return { draw };
  }
  // ── Quantum Computing ──────────────────────────────────────────────────────
  function createQuantumSketch() {
    const s = {
      // Bloch sphere
      theta: 0, phi: 0, targetTheta: 0, targetPhi: 0,
      // Entangled particles
      particles: [],
      // Quantum walk probability distribution
      walkProbs: [],
      walkCenter: 20,
      walkSize: 41,
      // State
      collapseFlash: 0,
      entangleStrength: 0,
      walkSpread: 0,
      gateAcc: 0,
      lastStem: ''
    };
    function resize(w, h) {
      const count = clamp(Math.floor((w * h) / 18000), 16, 80);
      s.particles = new Array(count);
      for (let i = 0; i < count; i++) {
        const angle = (i / count) * TAU;
        s.particles[i] = {
          x: 0.5 + Math.cos(angle) * 0.2,
          y: 0.5 + Math.sin(angle) * 0.2,
          vx: 0, vy: 0,
          phase: Math.random() * TAU,
          paired: i % 2 === 0 ? i + 1 : i - 1,
          glow: 0
        };
      }
      // Init walk distribution: delta at center
      s.walkProbs = new Float64Array(s.walkSize);
      s.walkProbs[s.walkCenter] = 1.0;
      s.theta = 0; s.phi = 0;
      s.collapseFlash = 0;
      s.entangleStrength = 0;
      s.walkSpread = 0;
      s.gateAcc = 0;
    }
    function draw(api) {
      if (!s.particles.length || api.stem !== s.lastStem) {
        s.lastStem = api.stem;
        resize(api.w, api.h);
      }
      const { bass, mid, treble, energy, rms, flux, transient, beat } = api.audio;
      const c = api.ctx;
      const w = api.w, h = api.h;

      // Update Bloch sphere angles from audio
      s.targetTheta = Math.PI * (0.5 + bass * 0.4 - treble * 0.3);
      s.targetPhi += api.dt * (1.2 + mid * 4 + flux * 8);
      s.theta += (s.targetTheta - s.theta) * Math.min(1, api.dt * 3);
      s.phi = s.targetPhi;

      // Entanglement strength: builds with sustained energy
      s.entangleStrength += (energy * 0.8 + rms * 0.5 - s.entangleStrength) * api.dt * 2;
      s.entangleStrength = clamp(s.entangleStrength, 0, 1);

      // Collapse flash on beat
      if (beat) s.collapseFlash = 0.8 + bass * 0.4;
      s.collapseFlash = Math.max(0, s.collapseFlash - api.dt * 2.5);

      // Walk spread from audio
      s.walkSpread += api.dt * (0.3 + energy * 2 + flux * 5);

      // Gate accumulator: triggers visual gate events
      s.gateAcc += api.dt * (0.5 + transient * 6 + flux * 3);

      // Update walk distribution (quantum-like: interference creates edge peaks)
      updateWalk(api.dt, bass, treble, beat);

      // Update particles
      updateParticles(api);

      c.save();
      c.globalCompositeOperation = 'lighter';

      // ── Layer 1: Bloch sphere wireframe (center) ──
      const cx = w * 0.5, cy = h * 0.42;
      const radius = Math.min(w, h) * 0.16;
      drawBlochSphere(c, cx, cy, radius, api);

      // ── Layer 2: Entangled particle pairs ──
      drawParticles(c, w, h, api);

      // ── Layer 3: Quantum walk probability bars (bottom) ──
      drawWalkDistribution(c, w, h, api);

      // ── Layer 4: Collapse flash overlay ──
      if (s.collapseFlash > 0.01) {
        c.fillStyle = `rgba(255,248,220,${s.collapseFlash * 0.12})`;
        c.fillRect(0, 0, w, h);
      }

      c.restore();
    }
    function updateWalk(dt, bass, treble, beat) {
      // Evolve: spread outward with interference
      const spread = dt * (0.4 + bass * 2);
      const newP = new Float64Array(s.walkSize);
      for (let i = 0; i < s.walkSize; i++) {
        const left = i > 0 ? s.walkProbs[i - 1] : 0;
        const right = i < s.walkSize - 1 ? s.walkProbs[i + 1] : 0;
        // Quantum-style: interference creates edge peaks
        const dist = Math.abs(i - s.walkCenter);
        const interference = 1 + 0.5 * Math.sin(dist * 0.8 + s.walkSpread * 2);
        newP[i] = s.walkProbs[i] * (1 - spread) + (left + right) * spread * 0.5 * interference;
      }
      // Normalize
      let sum = 0;
      for (let i = 0; i < s.walkSize; i++) sum += newP[i];
      if (sum > 0) for (let i = 0; i < s.walkSize; i++) newP[i] /= sum;
      s.walkProbs = newP;
      // Beat = measurement-like collapse: sharpen around peaks
      if (beat) {
        let maxI = s.walkCenter;
        for (let i = 0; i < s.walkSize; i++) {
          if (s.walkProbs[i] > s.walkProbs[maxI]) maxI = i;
        }
        for (let i = 0; i < s.walkSize; i++) {
          const d = Math.abs(i - maxI);
          s.walkProbs[i] *= Math.exp(-d * 0.3);
        }
        sum = 0;
        for (let i = 0; i < s.walkSize; i++) sum += s.walkProbs[i];
        if (sum > 0) for (let i = 0; i < s.walkSize; i++) s.walkProbs[i] /= sum;
      }
    }
    function updateParticles(api) {
      const { bass, mid, treble, energy, flux, transient, beat } = api.audio;
      const dt = api.dt;
      const cx = 0.5, cy = 0.42;
      for (let i = 0; i < s.particles.length; i++) {
        const p = s.particles[i];
        p.phase += dt * (2 + treble * 6 + flux * 4);
        // Orbit around center with entanglement pull to paired particle
        const angle = p.phase + (i / s.particles.length) * TAU;
        const orbitR = 0.15 + energy * 0.08 + Math.sin(p.phase * 0.3) * 0.03;
        const tx = cx + Math.cos(angle) * orbitR;
        const ty = cy + Math.sin(angle) * orbitR * 0.7;
        // Entangled pull: paired particles mirror
        const pair = s.particles[clamp(p.paired, 0, s.particles.length - 1)];
        if (pair && s.entangleStrength > 0.1) {
          const mirrorX = 2 * cx - pair.x;
          const mirrorY = 2 * cy - pair.y;
          const ent = s.entangleStrength * 0.3;
          p.vx += (tx * (1 - ent) + mirrorX * ent - p.x) * dt * 5;
          p.vy += (ty * (1 - ent) + mirrorY * ent - p.y) * dt * 5;
        } else {
          p.vx += (tx - p.x) * dt * 4;
          p.vy += (ty - p.y) * dt * 4;
        }
        p.vx *= 0.92; p.vy *= 0.92;
        p.x += p.vx * dt * 3;
        p.y += p.vy * dt * 3;
        // Collapse: scatter on beat
        if (beat && Math.random() < 0.15) {
          p.vx += (Math.random() - 0.5) * 0.5;
          p.vy += (Math.random() - 0.5) * 0.5;
          p.glow = 1;
        }
        p.glow = Math.max(0, p.glow - dt * 2.2);
      }
    }
    function drawBlochSphere(c, cx, cy, r, api) {
      const { treble, energy, rms, flux, transient } = api.audio;
      // Sphere outline
      c.strokeStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.08 + energy * 0.15})`;
      c.lineWidth = 1;
      c.beginPath(); c.arc(cx, cy, r, 0, TAU); c.stroke();
      // Equator ellipse
      c.beginPath(); c.ellipse(cx, cy, r, r * 0.3, 0, 0, TAU); c.stroke();
      // Meridian
      c.beginPath(); c.ellipse(cx, cy, r * 0.3, r, 0, 0, TAU); c.stroke();
      // |0⟩ and |1⟩ poles
      const poleAlpha = 0.2 + rms * 0.3;
      c.fillStyle = `rgba(180,220,255,${poleAlpha})`;
      c.beginPath(); c.arc(cx, cy - r, 2.5, 0, TAU); c.fill();
      c.fillStyle = `rgba(255,180,140,${poleAlpha})`;
      c.beginPath(); c.arc(cx, cy + r, 2.5, 0, TAU); c.fill();
      // State vector (Bloch point)
      const stateX = cx + r * Math.sin(s.theta) * Math.cos(s.phi);
      const stateY = cy - r * Math.cos(s.theta);
      // Line from center to state
      c.strokeStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.25 + treble * 0.3 + flux * 0.2})`;
      c.lineWidth = 1.5;
      c.beginPath(); c.moveTo(cx, cy); c.lineTo(stateX, stateY); c.stroke();
      // State point
      const stateGlow = 3 + rms * 4 + transient * 6 + s.collapseFlash * 8;
      c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.5 + energy * 0.3 + s.collapseFlash * 0.3})`;
      c.beginPath(); c.arc(stateX, stateY, stateGlow, 0, TAU); c.fill();
      // Gate ring pulse on transient
      if (transient > 0.3 || s.collapseFlash > 0.1) {
        const pulse = Math.max(transient, s.collapseFlash);
        c.strokeStyle = `rgba(255,248,210,${pulse * 0.25})`;
        c.lineWidth = 1;
        c.beginPath(); c.arc(cx, cy, r + 6 + pulse * 12, 0, TAU); c.stroke();
      }
    }
    function drawParticles(c, w, h, api) {
      const { treble, energy, rms, flux } = api.audio;
      for (let i = 0; i < s.particles.length; i++) {
        const p = s.particles[i];
        const x = p.x * w, y = p.y * h;
        if (x < -4 || x > w + 4 || y < -4 || y > h + 4) continue;
        const alpha = 0.12 + p.glow * 0.5 + energy * 0.1 + rms * 0.08;
        const size = 1 + p.glow * 2.5 + treble * 0.8;
        c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${alpha})`;
        c.beginPath(); c.arc(x, y, size, 0, TAU); c.fill();
        // Entanglement lines to paired particle
        if (s.entangleStrength > 0.15 && i % 2 === 0) {
          const pair = s.particles[clamp(p.paired, 0, s.particles.length - 1)];
          if (pair) {
            c.strokeStyle = `rgba(180,200,255,${s.entangleStrength * 0.08 + flux * 0.04})`;
            c.lineWidth = 0.5;
            c.beginPath(); c.moveTo(x, y); c.lineTo(pair.x * w, pair.y * h); c.stroke();
          }
        }
      }
    }
    function drawWalkDistribution(c, w, h, api) {
      const { bass, treble, energy, rms } = api.audio;
      const barW = w / s.walkSize;
      const baseY = h * 0.88;
      const maxH = h * 0.22;
      for (let i = 0; i < s.walkSize; i++) {
        const prob = s.walkProbs[i];
        if (prob < 0.001) continue;
        const barH = prob * maxH * (8 + energy * 12);
        const x = i * barW;
        const dist = Math.abs(i - s.walkCenter) / s.walkCenter;
        // Edge peaks glow brighter (quantum signature)
        const edgeGlow = dist * 0.3;
        const alpha = 0.1 + prob * 2 + edgeGlow + rms * 0.15;
        c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${Math.min(0.7, alpha)})`;
        c.fillRect(x + 1, baseY - barH, Math.max(1, barW - 2), barH);
        // Bright cap
        if (prob > 0.05) {
          c.fillStyle = `rgba(255,248,220,${prob * 1.5 + treble * 0.2})`;
          c.fillRect(x + 1, baseY - barH, Math.max(1, barW - 2), 2);
        }
      }
      // Center line
      c.strokeStyle = `rgba(130,120,100,${0.06 + bass * 0.08})`;
      c.lineWidth = 1;
      c.beginPath();
      c.moveTo(s.walkCenter * barW + barW * 0.5, baseY);
      c.lineTo(s.walkCenter * barW + barW * 0.5, baseY - maxH * 0.8);
      c.stroke();
    }
    return { resize, draw };
  }

  function createOptimizationSketch() {
    const s = {
      particles: [],
      wells: [],
      gridW: 0,
      gridH: 0,
      temp: 0.6,
      phase: 0,
      contourPhase: 0,
      stepAcc: 0,
      lastStem: ''
    };
    function resize(w, h) {
      s.gridW = clamp(Math.floor(w / 22), 30, 74);
      s.gridH = clamp(Math.floor(h / 24), 22, 54);
      s.wells = [
        { x: -0.58, y: 0.32, depth: 0.95, ax: 7.8, ay: 9.5 },
        { x: 0.48, y: -0.42, depth: 0.78, ax: 8.2, ay: 8.8 },
        { x: 0.03, y: 0.06, depth: 0.64, ax: 11.6, ay: 10.4 }
      ];
      const count = clamp(Math.floor((w * h) / 23000), 22, 70);
      s.particles = new Array(count);
      for (let i = 0; i < count; i++) {
        const x = Math.random() * 2 - 1;
        const y = Math.random() * 2 - 1;
        s.particles[i] = {
          x,
          y,
          vx: 0,
          vy: 0,
          bestX: x,
          bestY: y,
          bestVal: landscape(x, y, 0),
          glow: Math.random() * 0.4
        };
      }
      s.temp = 0.62;
      s.phase = Math.random() * TAU;
      s.contourPhase = Math.random();
      s.stepAcc = 0;
      void h;
    }
    function draw(api) {
      if (!s.particles.length || api.stem !== s.lastStem) {
        s.lastStem = api.stem;
        resize(api.w, api.h);
      }
      const { bass, mid, treble, energy, rms, flux, transient, beat } = api.audio;
      s.phase += api.dt * (0.34 + treble * 1.8 + flux * 3.2);
      s.contourPhase += api.dt * (0.28 + mid * 0.9 + rms * 0.6);
      s.temp += (0.09 + energy * 0.42 + rms * 0.35 - s.temp) * Math.min(1, api.dt * 1.4);
      s.temp += transient * api.dt * 3.2;
      if (beat) s.temp = Math.min(1.65, s.temp + 0.24 + bass * 0.5);
      s.temp = clamp(s.temp - api.dt * (0.06 + mid * 0.08), 0.02, 1.65);
      s.stepAcc += api.dt * (4 + energy * 14 + bass * 8 + flux * 22);
      let guard = 0;
      while (s.stepAcc >= 1 && guard < 24) {
        stepParticles(api.audio, 0.016 + api.dt * 0.35);
        s.stepAcc -= 1;
        guard++;
      }
      const c = api.ctx;
      c.save();
      c.globalCompositeOperation = 'lighter';
      const heat = c.createLinearGradient(0, 0, 0, api.h);
      heat.addColorStop(0, `rgba(223,165,87,${0.05 + s.temp * 0.16})`);
      heat.addColorStop(0.55, `rgba(100,84,52,${0.03 + energy * 0.08})`);
      heat.addColorStop(1, 'rgba(12,11,10,0.02)');
      c.fillStyle = heat;
      c.fillRect(0, 0, api.w, api.h);
      drawContours(api);
      for (let i = 0; i < s.wells.length; i++) {
        const w = s.wells[i];
        const x = (w.x * 0.5 + 0.5) * api.w;
        const y = (w.y * 0.5 + 0.5) * api.h;
        c.strokeStyle = `rgba(255,232,180,${0.05 + transient * 0.25})`;
        c.lineWidth = 1;
        c.beginPath();
        c.arc(x, y, 8 + (w.depth * 14 + bass * 9), 0, TAU);
        c.stroke();
      }
      for (let i = 0; i < s.particles.length; i++) {
        const p = s.particles[i];
        const x = (p.x * 0.5 + 0.5) * api.w;
        const y = (p.y * 0.5 + 0.5) * api.h;
        const bx = (p.bestX * 0.5 + 0.5) * api.w;
        const by = (p.bestY * 0.5 + 0.5) * api.h;
        c.strokeStyle = `rgba(255,224,164,${0.03 + p.glow * 0.22 + flux * 0.08})`;
        c.lineWidth = 1;
        c.beginPath();
        c.moveTo(x, y);
        c.lineTo(bx, by);
        c.stroke();
        c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.15 + p.glow * 0.5 + treble * 0.18})`;
        c.beginPath();
        c.arc(x, y, 1.2 + p.glow * 3 + rms * 2.1, 0, TAU);
        c.fill();
      }
      c.restore();
    }
    function drawContours(api) {
      const c = api.ctx;
      const { treble, energy } = api.audio;
      const stepX = api.w / Math.max(1, s.gridW - 1);
      const stepY = api.h / Math.max(1, s.gridH - 1);
      for (let gy = 0; gy < s.gridH; gy++) {
        const ny = (gy / Math.max(1, s.gridH - 1)) * 2 - 1;
        for (let gx = 0; gx < s.gridW; gx++) {
          const nx = (gx / Math.max(1, s.gridW - 1)) * 2 - 1;
          const z = landscape(nx, ny, s.phase * 0.7);
          const band = Math.abs(fract(z * 5.5 + s.contourPhase) - 0.5);
          if (band > 0.06 + treble * 0.02) continue;
          const alpha = 0.08 + (0.06 - band) * 2.4 + energy * 0.12;
          c.fillStyle = `rgba(204,175,106,${alpha})`;
          c.fillRect(gx * stepX, gy * stepY, 1.6, 1.6);
        }
      }
    }
    function stepParticles(audio, dtStep) {
      const { bass, mid, treble, energy, rms, flux, transient, beat } = audio;
      const eps = 0.014;
      for (let i = 0; i < s.particles.length; i++) {
        const p = s.particles[i];
        const g = gradient(p.x, p.y, eps);
        const tempGrad = 0.35 + (1 - (p.y + 1) * 0.5) * (0.8 + rms * 0.7);
        const jitter = s.temp * tempGrad * (0.35 + flux * 1.2 + transient * 1.8);
        const nx = (Math.random() - 0.5) * jitter + Math.sin(s.phase + p.y * 7) * bass * 0.06;
        const ny = (Math.random() - 0.5) * jitter + Math.cos(s.phase * 0.8 + p.x * 6) * mid * 0.05;
        p.vx = p.vx * 0.84 - g.x * (0.55 + energy * 1.25) * dtStep + nx * dtStep * 2.1;
        p.vy = p.vy * 0.84 - g.y * (0.55 + energy * 1.25) * dtStep + ny * dtStep * 2.1;
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < -1.02 || p.x > 1.02) p.vx *= -0.7;
        if (p.y < -1.02 || p.y > 1.02) p.vy *= -0.7;
        p.x = clamp(p.x, -1.02, 1.02);
        p.y = clamp(p.y, -1.02, 1.02);
        if (beat && Math.random() < 0.08 + treble * 0.14) {
          p.x = (Math.random() - 0.5) * 1.9;
          p.y = -0.95 + Math.random() * 0.22;
          p.vx *= 0.25;
          p.vy *= 0.25;
        }
        const val = landscape(p.x, p.y, s.phase);
        if (val < p.bestVal) {
          p.bestVal = val;
          p.bestX = p.x;
          p.bestY = p.y;
          p.glow = 1;
        }
        p.glow = Math.max(0, p.glow - dtStep * (0.8 + mid * 0.8));
      }
    }
    function gradient(x, y, eps) {
      const dx = (landscape(x + eps, y, s.phase) - landscape(x - eps, y, s.phase)) / (2 * eps);
      const dy = (landscape(x, y + eps, s.phase) - landscape(x, y - eps, s.phase)) / (2 * eps);
      return { x: dx, y: dy };
    }
    function landscape(x, y, t) {
      let z = 0.22 * (x * x + y * y);
      z += 0.11 * Math.sin(x * 3.4 + t * 0.8) * Math.cos(y * 3.1 - t * 0.6);
      for (let i = 0; i < s.wells.length; i++) {
        const w = s.wells[i];
        const dx = x - w.x;
        const dy = y - w.y;
        z -= w.depth * Math.exp(-(dx * dx * w.ax + dy * dy * w.ay));
      }
      return z;
    }
    return { resize, draw };
  }
  function createNumberTheorySketch() {
    const s = {
      points: [],
      primeList: [],
      maxN: 0,
      spacing: 8,
      cx: 0,
      cy: 0,
      modulus: 9,
      residue: 0,
      sieveIdx: 0,
      strikes: [],
      stepAcc: 0,
      lastStem: ''
    };
    function resize(w, h) {
      s.spacing = clamp(Math.floor(Math.min(w, h) / 70), 6, 12);
      const rings = Math.floor((Math.min(w, h) * 0.45) / s.spacing);
      s.maxN = clamp((rings * 2 + 1) * (rings * 2 + 1), 900, 5200);
      s.cx = w * 0.5;
      s.cy = h * 0.52;
      s.points = buildSpiral(s.maxN);
      const primeFlags = sieveFlags(s.maxN);
      s.primeList = [];
      for (let n = 2; n <= s.maxN; n++) {
        if (!primeFlags[n]) continue;
        s.primeList.push(n);
        s.points[n - 1].prime = true;
      }
      s.modulus = 9;
      s.residue = 0;
      s.sieveIdx = 0;
      s.strikes = [];
      s.stepAcc = 0;
    }
    function draw(api) {
      if (!s.points.length || api.stem !== s.lastStem) {
        s.lastStem = api.stem;
        resize(api.w, api.h);
      }
      const { bass, mid, treble, energy, rms, flux, transient, beat } = api.audio;
      const targetMod = clamp(5 + Math.floor(mid * 10 + rms * 6), 5, 21);
      if (Math.random() < api.dt * (2 + flux * 14)) s.modulus = targetMod;
      s.residue = ((Math.floor(api.ts * 0.003 + bass * 17 + transient * 31) % s.modulus) + s.modulus) % s.modulus;
      s.stepAcc += api.dt * (0.7 + flux * 8 + energy * 3);
      while (s.stepAcc >= 1) {
        sieveStrike(false, 1);
        s.stepAcc -= 1;
      }
      if (beat) sieveStrike(true, 1 + ((bass + treble) * 3 | 0));
      for (let i = 0; i < s.points.length; i++) {
        const p = s.points[i];
        p.glow = Math.max(0, p.glow - api.dt * (0.9 + mid * 1.6));
        p.strike = Math.max(0, p.strike - api.dt * (1.1 + transient * 2.3));
      }
      for (let i = s.strikes.length - 1; i >= 0; i--) {
        s.strikes[i].life -= api.dt * 1.8;
        if (s.strikes[i].life <= 0) s.strikes.splice(i, 1);
      }
      const c = api.ctx;
      c.save();
      c.globalCompositeOperation = 'lighter';
      const maxR = Math.min(api.w, api.h) * 0.43;
      const ringStep = maxR / Math.max(1, s.modulus);
      for (let r = 1; r <= s.modulus; r++) {
        const active = r - 1 === s.residue;
        c.strokeStyle = active
          ? `rgba(255,232,172,${0.1 + rms * 0.3 + energy * 0.2})`
          : `rgba(165,148,110,${0.03 + energy * 0.06})`;
        c.lineWidth = active ? 1.3 : 1;
        c.beginPath();
        c.arc(s.cx, s.cy, r * ringStep, 0, TAU);
        c.stroke();
      }
      for (let i = 0; i < s.points.length; i++) {
        const p = s.points[i];
        const x = s.cx + p.sx * s.spacing;
        const y = s.cy + p.sy * s.spacing;
        if (x < -6 || x > api.w + 6 || y < -6 || y > api.h + 6) continue;
        const modHit = p.n % s.modulus === s.residue;
        const primeAlpha = p.prime ? 0.14 + treble * 0.2 : 0.015 + energy * 0.04;
        const alpha = primeAlpha + p.glow * 0.44 + p.strike * 0.35 + (modHit ? 0.08 + bass * 0.12 : 0);
        c.fillStyle = p.prime
          ? `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${alpha})`
          : `rgba(130,118,92,${alpha * 0.6})`;
        c.beginPath();
        c.arc(x, y, p.prime ? 1.1 + p.glow * 1.9 + (modHit ? 0.45 : 0) : 0.65 + p.strike * 0.7, 0, TAU);
        c.fill();
      }
      for (let i = 0; i < s.strikes.length; i++) {
        const strike = s.strikes[i];
        const rr = (strike.prime % s.modulus + 1) * ringStep;
        c.strokeStyle = `rgba(255,242,204,${strike.life * (0.18 + strike.power * 0.32)})`;
        c.lineWidth = 1 + strike.power * 1.1;
        c.beginPath();
        c.arc(s.cx, s.cy, rr, 0, TAU);
        c.stroke();
      }
      c.restore();
    }
    function sieveStrike(onBeat, count) {
      if (!s.primeList.length) return;
      for (let k = 0; k < count; k++) {
        const prime = s.primeList[s.sieveIdx % s.primeList.length];
        s.sieveIdx++;
        const power = onBeat ? 1 : 0.46;
        const root = s.points[prime - 1];
        if (root) root.glow = 1.2;
        for (let m = prime * 2; m <= s.maxN; m += prime) {
          const p = s.points[m - 1];
          if (!p) continue;
          p.strike = 1;
          p.glow = Math.max(p.glow, power * (p.prime ? 0.9 : 0.45));
        }
        s.strikes.push({ prime, life: 1, power });
      }
      if (s.strikes.length > 36) s.strikes.splice(0, s.strikes.length - 36);
    }
    function buildSpiral(maxN) {
      const pts = [];
      let x = 0;
      let y = 0;
      let dx = 1;
      let dy = 0;
      let segLen = 1;
      let segProg = 0;
      let turns = 0;
      for (let n = 1; n <= maxN; n++) {
        pts.push({ n, sx: x, sy: y, prime: false, glow: 0, strike: 0 });
        x += dx;
        y += dy;
        segProg++;
        if (segProg === segLen) {
          segProg = 0;
          const t = dx;
          dx = -dy;
          dy = t;
          turns++;
          if (turns % 2 === 0) segLen++;
        }
      }
      return pts;
    }
    function sieveFlags(maxN) {
      const flags = new Uint8Array(maxN + 1).fill(1);
      flags[0] = 0;
      flags[1] = 0;
      const root = Math.floor(Math.sqrt(maxN));
      for (let n = 2; n <= root; n++) {
        if (!flags[n]) continue;
        for (let m = n * n; m <= maxN; m += n) flags[m] = 0;
      }
      return flags;
    }
    return { resize, draw };
  }
  function createInformationSketch() {
    const s = {
      cols: 0,
      rows: 0,
      cellW: 0,
      cellH: 0,
      stream: [],
      oneProb: 0.5,
      entropy: 0,
      symbols: new Float32Array(8),
      nodes: [],
      edges: [],
      maxDepth: 1,
      growth: 0,
      shiftAcc: 0,
      rebuildAcc: 0,
      lastStem: ''
    };
    function resize(w, h) {
      const streamW = w * 0.46;
      const streamH = h * 0.72;
      s.cols = clamp(Math.floor(streamW / 9), 24, 84);
      s.rows = clamp(Math.floor(streamH / 10), 12, 38);
      s.cellW = streamW / s.cols;
      s.cellH = streamH / s.rows;
      s.stream = [];
      s.oneProb = 0.5;
      s.entropy = 1;
      for (let i = 0; i < s.symbols.length; i++) s.symbols[i] = 1 + Math.random() * 0.8;
      for (let c = 0; c < s.cols; c++) {
        const bits = new Uint8Array(s.rows);
        for (let r = 0; r < s.rows; r++) bits[r] = Math.random() < 0.5 ? 1 : 0;
        s.stream.push({ bits, pulse: 0.2 });
      }
      rebuildTree();
      s.growth = 0;
      s.shiftAcc = 0;
      s.rebuildAcc = 0;
      void h;
    }
    function draw(api) {
      if (!s.stream.length || api.stem !== s.lastStem) {
        s.lastStem = api.stem;
        resize(api.w, api.h);
      }
      const { bass, mid, treble, energy, rms, flux, transient, beat } = api.audio;
      const wobble = Math.sin(api.ts * 0.00028 + bass * 4.4) * (0.28 - mid * 0.14);
      const targetProb = clamp(0.5 + wobble + (rms - 0.18) * 0.45, 0.05, 0.95);
      s.oneProb += (targetProb - s.oneProb) * Math.min(1, api.dt * (1.4 + flux * 18));
      s.shiftAcc += api.dt * (5 + energy * 34 + bass * 18 + flux * 26);
      if (beat) s.shiftAcc += 2.6 + transient * 8;
      while (s.shiftAcc >= 1) {
        pushColumn(api.audio);
        s.shiftAcc -= 1;
      }
      s.rebuildAcc += api.dt * (0.5 + flux * 4 + transient * 6 + rms * 1.2);
      if (beat || s.rebuildAcc > 1.2 + (1 - rms) * 0.9) {
        rebuildTree();
        s.rebuildAcc = 0;
        s.growth = 0;
      }
      s.growth = Math.min(s.maxDepth + 1.2, s.growth + api.dt * (1.2 + treble * 5.2 + transient * 4 + (beat ? 1.8 : 0)));
      const c = api.ctx;
      c.save();
      c.globalCompositeOperation = 'lighter';
      const leftX = api.w * 0.06;
      const topY = api.h * 0.14;
      const streamW = s.cols * s.cellW;
      const streamH = s.rows * s.cellH;
      c.fillStyle = 'rgba(26,22,17,0.22)';
      c.fillRect(leftX, topY, streamW, streamH);
      for (let ci = 0; ci < s.stream.length; ci++) {
        const col = s.stream[ci];
        const x = leftX + ci * s.cellW;
        const gate = 0.04 + s.entropy * 0.14 + col.pulse * 0.24;
        for (let r = 0; r < s.rows; r++) {
          const bit = col.bits[r];
          if (!bit && s.entropy < 0.62) continue;
          const y = topY + r * s.cellH;
          const alpha = bit
            ? gate + energy * 0.16 + transient * 0.12
            : 0.02 + s.entropy * 0.06 + rms * 0.05;
          c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${alpha})`;
          c.fillRect(x + 0.6, y + 0.6, s.cellW - 1.1, s.cellH - 1.1);
        }
        col.pulse = Math.max(0, col.pulse - api.dt * (1.2 + mid * 2));
      }
      const meterH = streamH * clamp(s.entropy, 0, 1);
      c.fillStyle = `rgba(255,236,186,${0.18 + s.entropy * 0.35 + treble * 0.2})`;
      c.fillRect(leftX + streamW + 8, topY + streamH - meterH, 6, meterH);
      const treeX = api.w * 0.58;
      const treeW = api.w * 0.36;
      const treeTop = api.h * 0.16;
      const treeH = api.h * 0.7;
      for (let i = 0; i < s.edges.length; i++) {
        const e = s.edges[i];
        if (e.by > s.growth) continue;
        const t = clamp(s.growth - e.ay, 0, 1);
        const ax = treeX + e.ax * treeW;
        const ay = treeTop + (e.ay / Math.max(1, s.maxDepth)) * treeH;
        const bx = treeX + e.bx * treeW;
        const by = treeTop + (e.by / Math.max(1, s.maxDepth)) * treeH;
        c.strokeStyle = `rgba(235,210,150,${0.08 + t * (0.26 + flux * 0.3)})`;
        c.lineWidth = 1 + t * 0.8;
        c.beginPath();
        c.moveTo(ax, ay);
        c.lineTo(bx, by);
        c.stroke();
      }
      for (let i = 0; i < s.nodes.length; i++) {
        const n = s.nodes[i];
        if (n.depth > s.growth) continue;
        const x = treeX + n.x * treeW;
        const y = treeTop + (n.depth / Math.max(1, s.maxDepth)) * treeH;
        const reveal = clamp(s.growth - n.depth + 0.2, 0, 1);
        const radius = n.leaf ? 2.2 + n.weight * 0.9 : 3 + n.weight * 0.7;
        c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.12 + reveal * 0.35 + n.weight * 0.06})`;
        c.beginPath();
        c.arc(x, y, radius, 0, TAU);
        c.fill();
      }
      c.restore();
    }
    function pushColumn(audio) {
      const { bass, mid, energy, flux, transient, beat } = audio;
      const bits = new Uint8Array(s.rows);
      let ones = 0;
      for (let r = 0; r < s.rows; r++) {
        const lane = r / Math.max(1, s.rows - 1);
        const laneBias = (0.5 - lane) * mid * 0.24;
        const wobble = Math.sin(r * 0.7 + bass * 3.2) * flux * 0.08;
        const p = clamp(s.oneProb + laneBias + wobble, 0.02, 0.98);
        bits[r] = Math.random() < p ? 1 : 0;
        if (bits[r]) ones++;
      }
      if (beat || transient > 0.1) {
        const burst = 1 + ((transient * 10) | 0);
        for (let i = 0; i < burst; i++) {
          const idx = (Math.random() * s.rows) | 0;
          bits[idx] ^= 1;
        }
      }
      for (let i = 0; i < s.symbols.length; i++) s.symbols[i] *= 0.993;
      for (let r = 0; r < s.rows; r += 3) {
        const a = bits[r] || 0;
        const b = bits[(r + 1) % s.rows] || 0;
        const c = bits[(r + 2) % s.rows] || 0;
        const sym = (a << 2) | (b << 1) | c;
        s.symbols[sym] += 0.6 + energy * 1.1;
      }
      const p = clamp(ones / Math.max(1, s.rows), 0.0001, 0.9999);
      const h = -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
      s.entropy += (h - s.entropy) * 0.38;
      s.stream.shift();
      s.stream.push({ bits, pulse: 0.25 + transient * 1.4 + (beat ? 0.65 : 0) });
    }
    function rebuildTree() {
      const queue = [];
      for (let i = 0; i < s.symbols.length; i++) {
        queue.push({ weight: Math.max(0.01, s.symbols[i]), sym: i, left: null, right: null });
      }
      while (queue.length > 1) {
        queue.sort((a, b) => a.weight - b.weight);
        const a = queue.shift();
        const b = queue.shift();
        queue.push({
          weight: a.weight + b.weight,
          sym: -1,
          left: a,
          right: b
        });
      }
      s.nodes = [];
      s.edges = [];
      s.maxDepth = 1;
      layout(queue[0], 0, 0, 1);
    }
    function layout(node, depth, lo, hi) {
      if (!node) return null;
      const split = lo + (hi - lo) * 0.5;
      const left = node.left ? layout(node.left, depth + 1, lo, split) : null;
      const right = node.right ? layout(node.right, depth + 1, split, hi) : null;
      let x = (lo + hi) * 0.5;
      if (left && right) x = (left.x + right.x) * 0.5;
      else if (left) x = left.x;
      else if (right) x = right.x;
      const item = {
        x,
        depth,
        weight: clamp(node.weight / 8, 0.05, 1),
        leaf: !left && !right
      };
      s.nodes.push(item);
      s.maxDepth = Math.max(s.maxDepth, depth);
      if (left) s.edges.push({ ax: x, ay: depth, bx: left.x, by: left.depth });
      if (right) s.edges.push({ ax: x, ay: depth, bx: right.x, by: right.depth });
      return item;
    }
    return { resize, draw };
  }
  function createCryptoSketch() {
    const s = {
      cols: 0,
      rows: 0,
      cell: 0,
      plain: [],
      cipher: [],
      diff: [],
      key: null,
      round: 0,
      avalanche: 0,
      shiftAcc: 0,
      lastStem: ''
    };
    function resize(w, h) {
      const panelW = Math.min(w * 0.36, 460);
      const panelH = h * 0.66;
      s.cols = clamp(Math.floor(panelW / 10), 18, 56);
      s.rows = clamp(Math.floor(panelH / 9), 16, 40);
      s.cell = Math.min(panelW / s.cols, panelH / s.rows);
      s.key = new Uint8Array(s.cols);
      for (let i = 0; i < s.cols; i++) s.key[i] = Math.random() < 0.5 ? 1 : 0;
      s.plain = [];
      s.cipher = [];
      s.diff = [];
      s.round = 0;
      s.avalanche = 0;
      s.shiftAcc = 0;
      for (let r = 0; r < s.rows; r++) pushRow({ bass: 0.3, mid: 0.3, treble: 0.2, energy: 0.3, rms: 0.2, flux: 0.04, transient: 0, beat: false });
      void h;
    }
    function draw(api) {
      if (!s.key || api.stem !== s.lastStem) {
        s.lastStem = api.stem;
        resize(api.w, api.h);
      }
      const { bass, mid, treble, energy, rms, flux, transient, beat } = api.audio;
      s.shiftAcc += api.dt * (6 + energy * 26 + bass * 16 + flux * 34);
      if (beat) s.shiftAcc += 4 + transient * 8;
      while (s.shiftAcc >= 1) {
        pushRow(api.audio);
        s.shiftAcc -= 1;
      }
      s.avalanche = Math.max(0, s.avalanche - api.dt * (0.8 + mid * 1.8));
      if (beat || transient > 0.12) {
        s.avalanche = clamp(s.avalanche + 0.22 + transient * 1.9 + flux * 0.4, 0, 1.8);
        mutateKey(1 + ((bass * 4 + flux * 6 + transient * 4) | 0));
      }
      const c = api.ctx;
      const leftX = api.w * 0.06;
      const topY = api.h * 0.17;
      const panelW = s.cols * s.cell;
      const panelH = s.rows * s.cell;
      const rightX = api.w - leftX - panelW;
      const midX0 = leftX + panelW + 14;
      const midX1 = rightX - 14;
      c.save();
      c.globalCompositeOperation = 'lighter';
      c.fillStyle = 'rgba(22,18,14,0.24)';
      c.fillRect(leftX, topY, panelW, panelH);
      c.fillRect(rightX, topY, panelW, panelH);
      for (let r = 0; r < s.rows; r++) {
        const pRow = s.plain[r];
        const cRow = s.cipher[r];
        const dRow = s.diff[r];
        if (!pRow || !cRow || !dRow) continue;
        const y = topY + r * s.cell;
        const linkAlpha = (0.03 + dRow.ratio * 0.18 + s.avalanche * 0.08) * (1 - r / Math.max(1, s.rows));
        c.strokeStyle = `rgba(245,220,165,${linkAlpha})`;
        c.lineWidth = 1;
        c.beginPath();
        c.moveTo(midX0, y + s.cell * 0.5);
        c.lineTo(midX1, y + s.cell * 0.5);
        c.stroke();
        for (let i = 0; i < s.cols; i++) {
          const xL = leftX + i * s.cell;
          const xR = rightX + i * s.cell;
          const plainBit = pRow.bits[i];
          const cipherBit = cRow.bits[i];
          const diffBit = dRow.bits[i];
          const aL = plainBit
            ? 0.09 + energy * 0.17 + pRow.pulse * 0.28
            : 0.02 + rms * 0.05;
          const aR = cipherBit
            ? 0.1 + treble * 0.21 + cRow.pulse * 0.24 + diffBit * 0.2
            : 0.02 + flux * 0.05;
          c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${aL})`;
          c.fillRect(xL + 0.6, y + 0.6, s.cell - 1.2, s.cell - 1.2);
          c.fillStyle = `rgba(250,233,190,${aR})`;
          c.fillRect(xR + 0.6, y + 0.6, s.cell - 1.2, s.cell - 1.2);
        }
        pRow.pulse = Math.max(0, pRow.pulse - api.dt * (1.2 + mid * 1.5));
        cRow.pulse = Math.max(0, cRow.pulse - api.dt * (1.1 + treble * 1.8));
      }
      const meterW = (midX1 - midX0) * clamp(s.avalanche / 1.8, 0, 1);
      c.fillStyle = `rgba(255,245,210,${0.12 + s.avalanche * 0.2 + beat * 0.2})`;
      c.fillRect(midX0, topY - 10, meterW, 4);
      c.restore();
    }
    function pushRow(audio) {
      const { bass, mid, treble, energy, rms, flux, transient, beat } = audio;
      const plainBits = new Uint8Array(s.cols);
      const diffBits = new Uint8Array(s.cols);
      const bias = Math.sin(s.round * 0.13 + bass * 3) * 0.09;
      const p = clamp(0.24 + mid * 0.32 + rms * 0.28 + bias, 0.05, 0.95);
      for (let i = 0; i < s.cols; i++) plainBits[i] = Math.random() < p ? 1 : 0;
      if (transient > 0.08 || beat) {
        const start = (Math.random() * s.cols) | 0;
        const span = clamp(2 + ((transient + treble) * 8 | 0), 2, Math.floor(s.cols / 2));
        for (let i = 0; i < span; i++) {
          const idx = (start + i) % s.cols;
          plainBits[idx] ^= 1;
        }
      }
      if (beat) {
        const idx = (Math.random() * s.cols) | 0;
        plainBits[idx] ^= 1;
      }
      const cipherBits = encryptRow(plainBits, audio);
      let diffCount = 0;
      for (let i = 0; i < s.cols; i++) {
        diffBits[i] = plainBits[i] ^ cipherBits[i];
        diffCount += diffBits[i];
      }
      s.plain.unshift({ bits: plainBits, pulse: 0.2 + transient * 1.2 + (beat ? 0.5 : 0) });
      s.cipher.unshift({ bits: cipherBits, pulse: 0.22 + flux * 1.1 + (beat ? 0.45 : 0) });
      s.diff.unshift({ bits: diffBits, ratio: diffCount / s.cols });
      if (s.plain.length > s.rows) s.plain.length = s.rows;
      if (s.cipher.length > s.rows) s.cipher.length = s.rows;
      if (s.diff.length > s.rows) s.diff.length = s.rows;
      s.round++;
      if (beat && Math.random() < 0.45 + energy * 0.35) mutateKey(1 + ((bass * 5 + flux * 8) | 0));
    }
    function encryptRow(plainBits, audio) {
      const { bass, treble, rms, flux, beat } = audio;
      const out = new Uint8Array(s.cols);
      const shift = (s.round + Math.floor(rms * 7)) % s.cols;
      for (let i = 0; i < s.cols; i++) {
        const k = s.key[(i + shift) % s.cols];
        const left = plainBits[(i + s.cols - 1 - (s.round % 3)) % s.cols];
        const right = plainBits[(i + 1 + (s.round % 5)) % s.cols];
        let bit = plainBits[i] ^ k;
        bit ^= (left ^ right) & ((i + s.round) & 1);
        if (Math.random() < s.avalanche * (0.04 + flux * 0.18)) bit ^= 1;
        if (beat && (i + s.round) % (4 + ((bass + treble) * 5 | 0)) === 0) bit ^= 1;
        out[i] = bit;
      }
      return out;
    }
    function mutateKey(flips) {
      for (let i = 0; i < flips; i++) {
        const idx = (Math.random() * s.cols) | 0;
        s.key[idx] ^= 1;
        if (Math.random() < 0.35) s.key[(idx + 1) % s.cols] ^= 1;
      }
    }
    return { resize, draw };
  }
  function createAutomataSketch() {
    const s = {
      mode: 'pda',
      states: [],
      transitions: [],
      current: 0,
      stack: ['$'],
      tape: [],
      pulses: [],
      stepAcc: 0,
      lastStem: ''
    };
    function resize(w, h) {
      const cx = w * 0.42;
      const cy = h * 0.5;
      const rx = w * 0.22;
      const ry = h * 0.25;
      s.states = [];
      for (let i = 0; i < 5; i++) {
        const a = (i / 5) * TAU - Math.PI / 2;
        s.states.push({
          x: cx + Math.cos(a) * rx,
          y: cy + Math.sin(a) * ry,
          glow: 0
        });
      }
      s.transitions = [
        { from: 0, to: 1, sym: '0', pop: null, push: 'A', curve: -0.22, pulse: 0 },
        { from: 0, to: 2, sym: '1', pop: null, push: 'B', curve: 0.24, pulse: 0 },
        { from: 1, to: 1, sym: '0', pop: null, push: 'A', curve: 0.42, pulse: 0 },
        { from: 1, to: 3, sym: '1', pop: 'A', push: null, curve: 0.15, pulse: 0 },
        { from: 2, to: 2, sym: '1', pop: null, push: 'B', curve: -0.38, pulse: 0 },
        { from: 2, to: 3, sym: '0', pop: 'B', push: null, curve: -0.12, pulse: 0 },
        { from: 3, to: 4, sym: '1', pop: null, push: null, curve: 0.2, pulse: 0 },
        { from: 3, to: 0, sym: '0', pop: null, push: null, curve: -0.24, pulse: 0 },
        { from: 4, to: 4, sym: '1', pop: null, push: null, curve: 0.5, pulse: 0 },
        { from: 4, to: 0, sym: 'e', pop: null, push: null, curve: 0.08, pulse: 0 }
      ];
      s.current = 0;
      s.stack = ['$'];
      s.tape = [];
      s.pulses = [];
      s.stepAcc = 0;
      void h;
    }
    function draw(api) {
      if (!s.states.length || api.stem !== s.lastStem) {
        s.mode = modeFromStem(api.stem);
        s.lastStem = api.stem;
        resize(api.w, api.h);
      }
      const { bass, mid, treble, energy, rms, flux, transient, beat } = api.audio;
      s.stepAcc += api.dt * (1.6 + energy * 6 + bass * 4 + flux * 8);
      if (beat) s.stepAcc += 2.4 + transient * 7;
      while (s.stepAcc >= 1) {
        stepMachine(api.audio);
        s.stepAcc -= 1;
      }
      for (let i = 0; i < s.transitions.length; i++) {
        s.transitions[i].pulse = Math.max(0, s.transitions[i].pulse - api.dt * (1.8 + treble * 2.4));
      }
      for (let i = s.pulses.length - 1; i >= 0; i--) {
        s.pulses[i].life -= api.dt * (1.4 + flux * 2.1);
        if (s.pulses[i].life <= 0) s.pulses.splice(i, 1);
      }
      for (let i = 0; i < s.states.length; i++) {
        s.states[i].glow = Math.max(0, s.states[i].glow - api.dt * (1.1 + rms * 1.4));
      }
      const c = api.ctx;
      c.save();
      c.globalCompositeOperation = 'lighter';
      for (let i = 0; i < s.transitions.length; i++) {
        const t = s.transitions[i];
        const a = s.states[t.from];
        const b = s.states[t.to];
        const pulse = t.pulse;
        if (!a || !b) continue;
        if (t.from === t.to) {
          const r = 24 + i * 0.6;
          c.strokeStyle = `rgba(188,168,126,${0.08 + pulse * 0.26 + energy * 0.08})`;
          c.lineWidth = 1 + pulse * 1.1;
          c.beginPath();
          c.arc(a.x, a.y - 18, r * 0.45, Math.PI * 0.1, Math.PI * 1.2);
          c.stroke();
          continue;
        }
        const mx = (a.x + b.x) * 0.5;
        const my = (a.y + b.y) * 0.5;
        const nx = b.y - a.y;
        const ny = -(b.x - a.x);
        const inv = 1 / (Math.sqrt(nx * nx + ny * ny) + 0.001);
        const cx = mx + nx * inv * t.curve * 90;
        const cy = my + ny * inv * t.curve * 90;
        c.strokeStyle = `rgba(188,168,126,${0.08 + energy * 0.1 + pulse * 0.32})`;
        c.lineWidth = 1 + pulse * 1.2;
        c.beginPath();
        c.moveTo(a.x, a.y);
        c.quadraticCurveTo(cx, cy, b.x, b.y);
        c.stroke();
      }
      for (let i = 0; i < s.pulses.length; i++) {
        const p = s.pulses[i];
        const a = s.states[p.from];
        const b = s.states[p.to];
        if (!a || !b) continue;
        c.strokeStyle = `rgba(255,234,182,${p.life * (0.18 + p.power * 0.28)})`;
        c.lineWidth = 1.4 + p.power * 1.3;
        c.beginPath();
        c.moveTo(a.x, a.y);
        c.lineTo(b.x, b.y);
        c.stroke();
      }
      for (let i = 0; i < s.states.length; i++) {
        const st = s.states[i];
        const active = i === s.current;
        c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.11 + st.glow * 0.36 + (active ? 0.34 : 0)})`;
        c.beginPath();
        c.arc(st.x, st.y, 12 + st.glow * 5 + (active ? 4 : 0), 0, TAU);
        c.fill();
      }
      const stackX = api.w * 0.82;
      const baseY = api.h * 0.78;
      for (let i = 0; i < s.stack.length; i++) {
        const y = baseY - i * 16;
        const bright = 1 - i / Math.max(1, s.stack.length);
        c.fillStyle = `rgba(230,206,152,${0.08 + bright * 0.16 + mid * 0.12})`;
        c.fillRect(stackX, y, 36, 12);
      }
      const tapeY = api.h * 0.88;
      for (let i = 0; i < s.tape.length; i++) {
        const x = api.w * 0.16 + i * 14;
        const fade = 1 - i / Math.max(1, s.tape.length);
        c.fillStyle = `rgba(250,232,194,${0.05 + fade * 0.24 + treble * 0.1})`;
        c.fillRect(x, tapeY, 10, 8);
      }
      c.restore();
    }
    function stepMachine(audio) {
      const { mid, treble, rms, flux, transient, beat, energy } = audio;
      const probOne = clamp(0.5 + (mid - rms) * 0.8 + Math.sin(treble * TAU) * 0.05, 0.08, 0.92);
      const symbol = Math.random() < probOne ? '1' : '0';
      s.tape.unshift(symbol);
      if (s.tape.length > 22) s.tape.length = 22;
      const candidates = [];
      for (let i = 0; i < s.transitions.length; i++) {
        const t = s.transitions[i];
        if (t.from !== s.current) continue;
        if (t.sym === symbol || t.sym === '*') candidates.push({ t, score: 1 });
        if (s.mode !== 'dfa' && t.sym === 'e' && (beat || flux > 0.08 || Math.random() < rms * 0.14)) {
          candidates.push({ t, score: 0.8 + flux * 3 });
        }
      }
      let chosen = null;
      let bestScore = -1;
      if (s.mode === 'dfa') {
        chosen = candidates.length ? candidates[0].t : null;
      } else {
        for (let i = 0; i < candidates.length; i++) {
          const cand = candidates[i];
          let score = cand.score + Math.random() * (0.4 + flux * 1.8);
          if (s.mode === 'pda' && cand.t.pop && s.stack[s.stack.length - 1] !== cand.t.pop) score -= 3;
          if (cand.t.push && transient > 0.06) score += transient * 2;
          if (score > bestScore) {
            bestScore = score;
            chosen = cand.t;
          }
        }
      }
      if (!chosen) {
        if (beat && Math.random() < 0.45 + energy * 0.2) s.current = 0;
        return;
      }
      if (s.mode === 'pda') {
        const top = s.stack[s.stack.length - 1];
        if (chosen.pop && top !== chosen.pop) {
          if (!beat) return;
          s.current = (s.current + 1) % s.states.length;
          return;
        }
        if (chosen.pop && s.stack.length > 1) s.stack.pop();
        if (chosen.push && s.stack.length < 12) s.stack.push(chosen.push);
        if (beat && transient > 0.1 && s.stack.length < 12) s.stack.push(Math.random() < 0.5 ? 'A' : 'B');
      }
      s.current = chosen.to;
      chosen.pulse = 1;
      s.states[s.current].glow = 1;
      s.pulses.push({ from: chosen.from, to: chosen.to, life: 1, power: 0.5 + transient * 2 + flux * 1.2 });
      if (s.pulses.length > 24) s.pulses.splice(0, s.pulses.length - 24);
      if (!s.stack.length) s.stack.push('$');
    }
    function modeFromStem(stem) {
      if (/dfa/i.test(stem || '')) return 'dfa';
      if (/nfa/i.test(stem || '')) return 'nfa';
      return 'pda';
    }
    return { resize, draw };
  }
  // ── Type Systems ──────────────────────────────────────────────────────
  function createTypeSystemsSketch() {
    // 5 type variables arranged in a semicircle, each with inference state
    const typeVars = [
      { name: 'α', target: 'Int',    angle: -0.6, snapped: false, snapProg: 0, wobble: 0.7 + Math.random() * 0.5 },
      { name: 'β', target: 'α→γ',   angle: -0.3, snapped: false, snapProg: 0, wobble: 0.9 + Math.random() * 0.5 },
      { name: 'γ', target: 'Bool',   angle: 0.0,  snapped: false, snapProg: 0, wobble: 0.6 + Math.random() * 0.5 },
      { name: 'δ', target: 'List α', angle: 0.3,  snapped: false, snapProg: 0, wobble: 0.8 + Math.random() * 0.5 },
      { name: 'ε', target: '≡β',    angle: 0.6,  snapped: false, snapProg: 0, wobble: 1.0 + Math.random() * 0.5 },
    ];
    // Constraint edges (unification)
    const constraints = [];
    let prevBeat = false;
    let flashTimer = 0;
    let flashIdx = -1;
    // Lattice nodes for subtype mode
    const latticeNodes = [
      { name: 'Top',  x: 0.5, y: 0.12, children: [1, 2, 3] },
      { name: 'Num',  x: 0.25, y: 0.35, children: [4, 5] },
      { name: 'Seq',  x: 0.5, y: 0.35, children: [6, 7] },
      { name: 'Fn',   x: 0.75, y: 0.35, children: [8, 9] },
      { name: 'Int',  x: 0.15, y: 0.58, children: [] },
      { name: 'Float',x: 0.35, y: 0.58, children: [] },
      { name: 'List', x: 0.42, y: 0.58, children: [] },
      { name: 'Stream',x: 0.58,y: 0.58, children: [] },
      { name: 'Pure', x: 0.68, y: 0.58, children: [] },
      { name: 'Effect',x: 0.82,y: 0.58, children: [] },
    ];
    function draw(api) {
      const { ctx, w, h, audio, intensity, stem, playhead, duration, ts } = api;
      const bass = audio.bass, mid = audio.mid, treble = audio.treble;
      const energy = audio.energy, beat = audio.beat, flux = audio.flux;
      const prog = duration > 0 ? playhead / duration : 0;
      const isHM = /hindley|milner/i.test(stem || '');
      const isSub = /subtype|lattice/i.test(stem || '');
      const isCH = /curry|howard/i.test(stem || '');
      ctx.globalCompositeOperation = 'lighter';
      // Beat flash
      if (beat && !prevBeat) {
        flashTimer = 1.0;
        flashIdx = Math.floor(Math.random() * typeVars.length);
      }
      prevBeat = beat;
      flashTimer *= 0.93;
      if (isSub) {
        // ── Subtype Lattice mode ──
        const cx = w / 2, cy = h / 2;
        // Draw edges
        for (const node of latticeNodes) {
          const nx = node.x * w, ny = node.y * h * 1.2 + h * 0.05;
          for (const ci of node.children) {
            const child = latticeNodes[ci];
            const cx2 = child.x * w, cy2 = child.y * h * 1.2 + h * 0.05;
            const alpha = 0.1 + energy * 0.2;
            ctx.strokeStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${alpha.toFixed(3)})`;
            ctx.lineWidth = 1 + bass * 2;
            ctx.beginPath(); ctx.moveTo(nx, ny); ctx.lineTo(cx2, cy2); ctx.stroke();
          }
        }
        // Draw nodes
        for (let i = 0; i < latticeNodes.length; i++) {
          const node = latticeNodes[i];
          const nx = node.x * w, ny = node.y * h * 1.2 + h * 0.05;
          // Depth determines brightness: deeper = dimmer (more specific type)
          const depth = node.y;
          const bright = 0.4 + (1.0 - depth) * 0.6 * (0.5 + energy * 0.5);
          // Active node based on playhead
          const nodePhase = i / latticeNodes.length;
          const isActive = Math.abs(prog - nodePhase) < 0.08;
          const r = isActive ? 10 + mid * 15 : 5 + energy * 4;
          if (isActive || energy > 0.3) {
            const gr = ctx.createRadialGradient(nx, ny, 0, nx, ny, r * 4);
            const a = isActive ? 0.3 + treble * 0.3 : 0.05 + energy * 0.1;
            gr.addColorStop(0, `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${a.toFixed(3)})`);
            gr.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = gr;
            ctx.beginPath(); ctx.arc(nx, ny, r * 4, 0, TAU); ctx.fill();
          }
          ctx.fillStyle = `rgba(${(GOLD[0]*bright)|0},${(GOLD[1]*bright)|0},${(GOLD[2]*bright)|0},0.9)`;
          ctx.beginPath(); ctx.arc(nx, ny, r, 0, TAU); ctx.fill();
          // Label
          ctx.globalCompositeOperation = 'source-over';
          ctx.font = '10px monospace';
          ctx.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${(bright * 0.5).toFixed(2)})`;
          ctx.textAlign = 'center';
          ctx.fillText(node.name, nx, ny + r + 14);
          ctx.globalCompositeOperation = 'lighter';
        }
        // Bottom label
        ctx.globalCompositeOperation = 'source-over';
        ctx.font = '9px monospace';
        ctx.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.15)`;
        ctx.textAlign = 'center';
        ctx.fillText('⊤ → ⊥', w / 2, h * 0.85);
        ctx.textAlign = 'start';
      } else if (isCH) {
        // ── Curry-Howard mode: proof construction ──
        const cx = w / 2, cy = h * 0.4;
        // Three proof blocks: ∧ (left), → (center), ∨ (right)
        const blocks = [
          { sym: 'A ∧ B', x: 0.2, phase: [0, 0.33] },
          { sym: 'A → B', x: 0.5, phase: [0.33, 0.66] },
          { sym: 'A ∨ B → C', x: 0.8, phase: [0.66, 1.0] },
        ];
        for (const blk of blocks) {
          const bx = blk.x * w, by = cy;
          const isActiveBlock = prog >= blk.phase[0] && prog < blk.phase[1];
          const localProg = isActiveBlock ? (prog - blk.phase[0]) / (blk.phase[1] - blk.phase[0]) : (prog >= blk.phase[1] ? 1 : 0);
          // Proof tree: branches grow with progress
          const depth = Math.floor(localProg * 4) + 1;
          const branchLen = 30 + mid * 40;
          for (let d = 0; d < depth; d++) {
            const angle1 = -Math.PI / 2 - 0.4 + d * 0.15;
            const angle2 = -Math.PI / 2 + 0.4 - d * 0.15;
            const len = branchLen * (1 - d * 0.2);
            const alpha = isActiveBlock ? 0.2 + energy * 0.4 : 0.06;
            ctx.strokeStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${alpha.toFixed(3)})`;
            ctx.lineWidth = 1 + bass;
            const ox = bx, oy = by - d * 25;
            ctx.beginPath();
            ctx.moveTo(ox, oy);
            ctx.lineTo(ox + Math.cos(angle1) * len, oy + Math.sin(angle1) * len);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(ox, oy);
            ctx.lineTo(ox + Math.cos(angle2) * len, oy + Math.sin(angle2) * len);
            ctx.stroke();
          }
          // Glow at root if active
          if (isActiveBlock) {
            const gr = ctx.createRadialGradient(bx, by, 0, bx, by, 40 + energy * 30);
            gr.addColorStop(0, `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${(0.15 + treble * 0.2).toFixed(3)})`);
            gr.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = gr;
            ctx.beginPath(); ctx.arc(bx, by, 40 + energy * 30, 0, TAU); ctx.fill();
          }
          // Completion indicator
          if (localProg >= 0.9) {
            ctx.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${(0.6 + treble * 0.3).toFixed(3)})`;
            ctx.beginPath(); ctx.arc(bx, by - depth * 25 - 10, 3, 0, TAU); ctx.fill();
          }
          // Label
          ctx.globalCompositeOperation = 'source-over';
          ctx.font = '11px monospace';
          const labelAlpha = isActiveBlock ? 0.5 : 0.15;
          ctx.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${labelAlpha.toFixed(2)})`;
          ctx.textAlign = 'center';
          ctx.fillText(blk.sym, bx, by + 50);
          ctx.globalCompositeOperation = 'lighter';
        }
        // Horizontal inference bar
        const barY = h * 0.75;
        const barProg = Math.min(1, prog * 1.1);
        ctx.strokeStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.2)`;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(w * 0.1, barY); ctx.lineTo(w * 0.9, barY); ctx.stroke();
        ctx.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.5)`;
        ctx.fillRect(w * 0.1, barY - 1, (w * 0.8) * barProg, 3);
      } else {
        // ── Hindley-Milner mode (default): type variable inference ──
        const cx = w / 2, cy = h * 0.4;
        const radius = Math.min(w, h) * 0.3;
        // Snap timing: each variable snaps at prog thresholds
        const snapThresholds = [0.15, 0.30, 0.45, 0.60, 0.72];
        for (let i = 0; i < typeVars.length; i++) {
          const tv = typeVars[i];
          const wasSnapped = tv.snapped;
          tv.snapped = prog >= snapThresholds[i];
          if (tv.snapped && !wasSnapped) {
            tv.snapProg = 1.0;
          }
          tv.snapProg *= 0.96;
          // Position: semicircle
          const a = tv.angle;
          const bx = cx + Math.cos(a - Math.PI / 2) * radius;
          const by = cy + Math.sin(a - Math.PI / 2) * radius * 0.6;
          // Wobble (uncertainty) or stable
          let drawX = bx, drawY = by;
          if (!tv.snapped) {
            const wob = tv.wobble * 8 * (1.0 + energy * 3);
            drawX += Math.sin(ts * 0.003 * tv.wobble + i * 2) * wob;
            drawY += Math.cos(ts * 0.004 * tv.wobble + i * 3) * wob * 0.6;
          }
          // Node size
          const r = tv.snapped ? 8 + mid * 5 : 5 + flux * 10;
          // Glow
          const glowR = r * (tv.snapped ? 4 + tv.snapProg * 8 : 2 + energy * 2);
          const glowAlpha = tv.snapped ? 0.15 + tv.snapProg * 0.4 + treble * 0.1 : 0.05 + energy * 0.08;
          const gr = ctx.createRadialGradient(drawX, drawY, 0, drawX, drawY, glowR);
          gr.addColorStop(0, `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${glowAlpha.toFixed(3)})`);
          gr.addColorStop(1, 'rgba(0,0,0,0)');
          ctx.fillStyle = gr;
          ctx.beginPath(); ctx.arc(drawX, drawY, glowR, 0, TAU); ctx.fill();
          // Core circle
          const bright = tv.snapped ? 0.7 + treble * 0.3 : 0.2 + energy * 0.3;
          ctx.fillStyle = `rgba(${(GOLD[0]*bright)|0},${(GOLD[1]*bright)|0},${(GOLD[2]*bright)|0},0.9)`;
          ctx.beginPath(); ctx.arc(drawX, drawY, r, 0, TAU); ctx.fill();
          // Constraint line to center when snapped
          if (tv.snapped) {
            const alpha = 0.08 + tv.snapProg * 0.3;
            ctx.strokeStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${alpha.toFixed(3)})`;
            ctx.lineWidth = 0.5 + tv.snapProg * 2;
            ctx.beginPath(); ctx.moveTo(drawX, drawY); ctx.lineTo(cx, cy); ctx.stroke();
          }
          // Labels
          ctx.globalCompositeOperation = 'source-over';
          ctx.font = '12px monospace';
          const labelAlpha = tv.snapped ? 0.5 + tv.snapProg * 0.3 : 0.15;
          ctx.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${labelAlpha.toFixed(2)})`;
          ctx.textAlign = 'center';
          ctx.fillText(tv.name, drawX, drawY + r + 16);
          if (tv.snapped) {
            ctx.font = '9px monospace';
            ctx.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${(0.2 + tv.snapProg * 0.3).toFixed(2)})`;
            ctx.fillText(': ' + tv.target, drawX, drawY + r + 28);
          }
          ctx.globalCompositeOperation = 'lighter';
        }
        // Central unification point
        const allSnapped = typeVars.every(v => v.snapped);
        if (allSnapped) {
          const coreAlpha = 0.1 + bass * 0.3;
          const coreR = 15 + mid * 20;
          const gr = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR);
          gr.addColorStop(0, `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${coreAlpha.toFixed(3)})`);
          gr.addColorStop(1, 'rgba(0,0,0,0)');
          ctx.fillStyle = gr;
          ctx.beginPath(); ctx.arc(cx, cy, coreR, 0, TAU); ctx.fill();
          // Principal type label
          ctx.globalCompositeOperation = 'source-over';
          ctx.font = '10px monospace';
          ctx.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${(0.2 + bass * 0.3).toFixed(2)})`;
          ctx.textAlign = 'center';
          ctx.fillText('principal type', cx, cy + coreR + 14);
          ctx.globalCompositeOperation = 'lighter';
        }
      }
      // Bottom label
      ctx.globalCompositeOperation = 'source-over';
      ctx.font = '10px monospace';
      ctx.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.2)`;
      ctx.textAlign = 'start';
      const label = isHM ? 'hindley-milner' : isSub ? 'subtype lattice' : isCH ? 'curry-howard' : 'type systems';
      ctx.fillText(label, 12, h - 12);
    }
    return { draw };
  }
  function createDistributedSketch() {
    // 5 Raft nodes in a pentagon, gossip ring, vector clock timelines
    const NUM_NODES = 5;
    const nodes = [];
    for (let i = 0; i < NUM_NODES; i++) {
      const a = -Math.PI / 2 + (i * 2 * Math.PI) / NUM_NODES;
      nodes.push({ x: 0.5 + 0.25 * Math.cos(a), y: 0.45 + 0.25 * Math.sin(a), phase: Math.random() * TAU, alive: true });
    }
    let leader = -1;
    let splitBrain = false;
    let prevBeat = false;
    let heartbeatPhase = 0;
    let gossipWave = [];  // { from, to, progress, active }
    let infectedSet = new Set();
    let msgParticles = []; // { x, y, tx, ty, progress, life }
    let vcBars = [0, 0, 0, 0]; // vector clock values per process

    return {
      draw(api) {
        const { ctx: c, w, h, dt, ts, audio, intensity, playhead, duration, stem } = api;
        const progress = duration > 0 ? playhead / duration : 0;
        const bass = audio.bass, mid = audio.mid, treble = audio.treble;
        const energy = audio.energy, beat = audio.beat, flux = audio.flux;

        c.fillStyle = BG;
        c.fillRect(0, 0, w, h);

        const isRaft = stem.includes('raft');
        const isGossip = stem.includes('gossip');
        const isVector = stem.includes('vector');

        const cx = w / 2, cy = h * 0.45;
        const radius = Math.min(w, h) * 0.25;

        // Heartbeat phase
        heartbeatPhase += dt * (2 + bass * 3);

        if (isRaft) {
          // ── Raft Consensus ──
          // Leader election based on progress
          if (progress < 0.15) { leader = -1; splitBrain = false; }
          else if (progress < 0.50) { leader = 2; splitBrain = false; }
          else if (progress < 0.75) { leader = 2; splitBrain = true; }
          else { leader = 2; splitBrain = false; }

          // Draw edges (connections between nodes)
          for (let i = 0; i < NUM_NODES; i++) {
            for (let j = i + 1; j < NUM_NODES; j++) {
              const partitioned = splitBrain && ((i < 2 && j >= 2) || (i >= 2 && j < 2));
              const nx = nodes[i].x * w, ny = nodes[i].y * h;
              const mx = nodes[j].x * w, my = nodes[j].y * h;
              if (partitioned) {
                // Dashed line for broken connection
                c.setLineDash([4, 8]);
                c.strokeStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.1)`;
              } else {
                c.setLineDash([]);
                c.strokeStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.15 + mid * 0.2})`;
              }
              c.lineWidth = 1;
              c.beginPath();
              c.moveTo(nx, ny);
              c.lineTo(mx, my);
              c.stroke();
            }
          }
          c.setLineDash([]);

          // Draw nodes
          for (let i = 0; i < NUM_NODES; i++) {
            const nx = nodes[i].x * w, ny = nodes[i].y * h;
            const isLeader = (i === leader);
            const isFalseLeader = splitBrain && i === 0;
            const nodeRadius = 12 + (isLeader ? 8 + bass * 10 : 0) + (isFalseLeader ? 6 + treble * 8 : 0);

            // Heartbeat pulse for leader
            if (isLeader) {
              const pulse = Math.sin(heartbeatPhase * 4) * 0.5 + 0.5;
              const pr = nodeRadius + 15 * pulse * energy;
              c.beginPath();
              c.arc(nx, ny, pr, 0, TAU);
              c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.08 * pulse})`;
              c.fill();
            }

            // False leader pulse (red-ish during split brain)
            if (isFalseLeader) {
              const pulse = Math.sin(heartbeatPhase * 4.3) * 0.5 + 0.5;
              const pr = nodeRadius + 12 * pulse;
              c.beginPath();
              c.arc(nx, ny, pr, 0, TAU);
              c.fillStyle = `rgba(200,80,60,${0.1 * pulse})`;
              c.fill();
            }

            // Node circle
            c.beginPath();
            c.arc(nx, ny, nodeRadius, 0, TAU);
            if (isLeader) {
              c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.7 + bass * 0.3})`;
            } else if (isFalseLeader) {
              c.fillStyle = `rgba(200,100,60,${0.5 + treble * 0.3})`;
            } else {
              c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.2 + mid * 0.15})`;
            }
            c.fill();

            // Node label
            c.fillStyle = BG;
            c.font = `${Math.round(10 + (isLeader ? 2 : 0))}px monospace`;
            c.textAlign = 'center';
            c.textBaseline = 'middle';
            c.fillText(`N${i}`, nx, ny);
          }

          // Leader label
          if (leader >= 0) {
            const lx = nodes[leader].x * w, ly = nodes[leader].y * h - 30;
            c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.6)`;
            c.font = '11px monospace';
            c.textAlign = 'center';
            c.fillText('LEADER', lx, ly);
          }
          if (splitBrain) {
            const fx = nodes[0].x * w, fy = nodes[0].y * h - 25;
            c.fillStyle = 'rgba(200,80,60,0.5)';
            c.font = '10px monospace';
            c.textAlign = 'center';
            c.fillText('SPLIT', fx, fy);
          }

          // Heartbeat waves radiating from leader
          if (leader >= 0 && beat && !prevBeat) {
            for (let i = 0; i < 3; i++) {
              msgParticles.push({
                x: nodes[leader].x * w, y: nodes[leader].y * h,
                tx: nodes[(leader + i + 1) % NUM_NODES].x * w,
                ty: nodes[(leader + i + 1) % NUM_NODES].y * h,
                progress: 0, life: 1
              });
            }
          }

        } else if (isGossip) {
          // ── Gossip Protocol ──
          // 7 nodes in a ring
          const gNodes = 7;
          const gRadius = Math.min(w, h) * 0.28;
          const gPositions = [];
          for (let i = 0; i < gNodes; i++) {
            const a = -Math.PI / 2 + (i * TAU) / gNodes;
            gPositions.push({ x: cx + gRadius * Math.cos(a), y: cy + gRadius * Math.sin(a) });
          }

          // Infection spread follows progress (S-curve)
          const numInfected = Math.min(gNodes, Math.floor(1 + (gNodes - 1) * Math.pow(progress, 0.6)));
          infectedSet.clear();
          for (let i = 0; i < numInfected; i++) infectedSet.add(i);

          // Draw ring connections
          for (let i = 0; i < gNodes; i++) {
            const j = (i + 1) % gNodes;
            const bothInfected = infectedSet.has(i) && infectedSet.has(j);
            c.beginPath();
            c.moveTo(gPositions[i].x, gPositions[i].y);
            c.lineTo(gPositions[j].x, gPositions[j].y);
            c.strokeStyle = bothInfected
              ? `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.3 + mid * 0.3})`
              : `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.08)`;
            c.lineWidth = bothInfected ? 2 : 1;
            c.stroke();
          }

          // Draw gossip nodes
          for (let i = 0; i < gNodes; i++) {
            const infected = infectedSet.has(i);
            const nr = infected ? 14 + bass * 8 : 8;
            // Infection glow
            if (infected) {
              c.beginPath();
              c.arc(gPositions[i].x, gPositions[i].y, nr + 10 + energy * 8, 0, TAU);
              c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.06)`;
              c.fill();
            }
            c.beginPath();
            c.arc(gPositions[i].x, gPositions[i].y, nr, 0, TAU);
            c.fillStyle = infected
              ? `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.5 + bass * 0.4})`
              : `rgba(100,95,85,0.3)`;
            c.fill();
            // Label
            c.fillStyle = infected ? BG : `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.4)`;
            c.font = '10px monospace';
            c.textAlign = 'center';
            c.textBaseline = 'middle';
            c.fillText(`${i}`, gPositions[i].x, gPositions[i].y);
          }

          // Gossip wave on beat
          if (beat && !prevBeat && infectedSet.size > 0 && infectedSet.size < gNodes) {
            const from = Array.from(infectedSet)[Math.floor(Math.random() * infectedSet.size)];
            const to = (from + (Math.random() < 0.5 ? 1 : gNodes - 1)) % gNodes;
            gossipWave.push({ from, to, progress: 0 });
          }

          // Render gossip waves
          gossipWave = gossipWave.filter(g => g.progress < 1);
          for (const g of gossipWave) {
            g.progress += dt * 2;
            const fp = gPositions[g.from], tp = gPositions[g.to];
            const px = fp.x + (tp.x - fp.x) * g.progress;
            const py = fp.y + (tp.y - fp.y) * g.progress;
            const alpha = 1 - g.progress;
            c.beginPath();
            c.arc(px, py, 4, 0, TAU);
            c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.7 * alpha})`;
            c.fill();
          }

        } else {
          // ── Vector Clocks ──
          // 4 horizontal process lanes
          const lanes = 4;
          const laneH = h * 0.7 / lanes;
          const laneTop = h * 0.12;
          const laneLeft = w * 0.08;
          const laneRight = w * 0.92;

          // Update vector clocks with audio
          for (let i = 0; i < lanes; i++) {
            vcBars[i] += dt * (0.5 + [bass, mid, treble, energy][i] * 2);
          }

          for (let i = 0; i < lanes; i++) {
            const y = laneTop + i * laneH + laneH / 2;

            // Lane background
            c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.03)`;
            c.fillRect(laneLeft, y - laneH * 0.35, laneRight - laneLeft, laneH * 0.7);

            // Process label
            c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.5)`;
            c.font = '11px monospace';
            c.textAlign = 'right';
            c.textBaseline = 'middle';
            c.fillText(`P${i}`, laneLeft - 8, y);

            // Timeline with events as dots
            const eventSpacing = (laneRight - laneLeft) / 20;
            const rate = [1.0, 1.333, 1.667, 2.0][i];
            const numEvents = Math.floor(progress * 20 * rate);
            for (let e = 0; e < Math.min(numEvents, 20); e++) {
              const ex = laneLeft + e * eventSpacing / rate;
              if (ex > laneRight) break;
              const dotR = 3 + (e === numEvents - 1 ? [bass, mid, treble, energy][i] * 5 : 0);
              c.beginPath();
              c.arc(ex, y, dotR, 0, TAU);
              c.fillStyle = e === numEvents - 1
                ? `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.7 + [bass, mid, treble, energy][i] * 0.3})`
                : `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.25)`;
              c.fill();
            }

            // Playhead position
            const px = laneLeft + progress * (laneRight - laneLeft);
            c.beginPath();
            c.arc(px, y, 2, 0, TAU);
            c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.6)`;
            c.fill();
          }

          // Message arrows between lanes on flux/beat
          if (beat && !prevBeat) {
            const from = Math.floor(Math.random() * lanes);
            let to = Math.floor(Math.random() * lanes);
            while (to === from) to = Math.floor(Math.random() * lanes);
            const px = laneLeft + progress * (laneRight - laneLeft);
            msgParticles.push({
              x: px, y: laneTop + from * laneH + laneH / 2,
              tx: px + 20, ty: laneTop + to * laneH + laneH / 2,
              progress: 0, life: 1
            });
          }

          // Vector clock value bars at bottom
          const barW = w * 0.15;
          const barH = 6;
          const barY = h * 0.92;
          c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.4)`;
          c.font = '9px monospace';
          c.textAlign = 'center';
          for (let i = 0; i < lanes; i++) {
            const bx = w * 0.15 + i * (w * 0.7 / lanes);
            const val = Math.min(1, (vcBars[i] % 10) / 10);
            c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.12)`;
            c.fillRect(bx, barY, barW * 0.8, barH);
            c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.3 + [bass,mid,treble,energy][i] * 0.5})`;
            c.fillRect(bx, barY, barW * 0.8 * val, barH);
            c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},0.4)`;
            c.fillText(`VC[${i}]`, bx + barW * 0.4, barY - 6);
          }
        }

        // Message particles (shared between modes)
        msgParticles = msgParticles.filter(p => p.life > 0);
        for (const p of msgParticles) {
          p.progress = Math.min(1, p.progress + dt * 3);
          p.life -= dt * 1.5;
          const px = p.x + (p.tx - p.x) * p.progress;
          const py = p.y + (p.ty - p.y) * p.progress;
          c.beginPath();
          c.arc(px, py, 3, 0, TAU);
          c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.6 * p.life})`;
          c.fill();
          // Trail
          c.beginPath();
          c.moveTo(p.x + (p.tx - p.x) * Math.max(0, p.progress - 0.15), p.y + (p.ty - p.y) * Math.max(0, p.progress - 0.15));
          c.lineTo(px, py);
          c.strokeStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.2 * p.life})`;
          c.lineWidth = 1;
          c.stroke();
        }

        prevBeat = beat;
      }
    };
  }

  function createCategorySketch() {
    // Category objects as nodes, functors as arrows, monad as effect chain
    const catC = [
      { name: 'C₀', freq: 262, x: 0.15, y: 0.3 },
      { name: 'C₁', freq: 294, x: 0.25, y: 0.2 },
      { name: 'C₂', freq: 330, x: 0.35, y: 0.3 },
      { name: 'C₃', freq: 392, x: 0.30, y: 0.45 },
      { name: 'C₄', freq: 440, x: 0.20, y: 0.45 },
    ];
    const catD = [
      { name: 'D₀', freq: 392, x: 0.65, y: 0.3 },
      { name: 'D₁', freq: 440, x: 0.75, y: 0.2 },
      { name: 'D₂', freq: 494, x: 0.85, y: 0.3 },
      { name: 'D₃', freq: 587, x: 0.80, y: 0.45 },
      { name: 'D₄', freq: 659, x: 0.70, y: 0.45 },
    ];
    // Morphisms within each category (edges)
    const morphs = [[0,1],[1,2],[2,3],[3,4],[4,0],[0,2],[1,3]];
    let prevBeat = false;
    let pulseTimer = 0;
    let activeArrow = -1;
    // Monad chain state
    const chain = [];
    for (let i = 0; i < 8; i++) chain.push({ alive: true, wrap: 0, x: 0.1 + i * 0.1, flash: 0 });
    function draw(api) {
      const { ctx, w, h, audio, intensity, stem, playhead, duration, ts } = api;
      const bass = audio.bass, mid = audio.mid, treble = audio.treble;
      const energy = audio.energy, beat = audio.beat, flux = audio.flux;
      const prog = duration > 0 ? playhead / duration : 0;
      const isFunctor = /functor/i.test(stem || '');
      const isNatTrans = /natural|transformation/i.test(stem || '');
      const isMonad = /monad/i.test(stem || '');
      ctx.globalCompositeOperation = 'lighter';
      if (beat && !prevBeat) { pulseTimer = 1.0; activeArrow = Math.floor(Math.random() * morphs.length); }
      prevBeat = beat;
      pulseTimer *= 0.92;
      if (isMonad) {
        // ── Monad: chain of wrapped values ──
        const cx = w / 2, baseY = h * 0.45;
        const chainLen = chain.length;
        // Update chain state based on progress
        const killPoint = 0.45 + 0.15; // ~60% through, some die
        for (let i = 0; i < chainLen; i++) {
          const c = chain[i];
          c.wrap = Math.min(1, prog * 2.5 - i * 0.15);
          if (c.wrap < 0) c.wrap = 0;
          // Some die in the bind chain section
          if (prog > 0.45 && prog < 0.75 && i > 2 && (i === 4 || i === 6)) c.alive = false;
          c.flash *= 0.95;
          if (beat && i === Math.floor(prog * chainLen)) c.flash = 1;
        }
        // Draw chain connections
        for (let i = 0; i < chainLen - 1; i++) {
          const c1 = chain[i], c2 = chain[i + 1];
          const x1 = c1.x * w, x2 = c2.x * w;
          const alpha = (c1.alive && c2.alive) ? 0.1 + energy * 0.15 : 0.03;
          ctx.strokeStyle = 'rgba(' + GOLD[0] + ',' + GOLD[1] + ',' + GOLD[2] + ',' + alpha.toFixed(3) + ')';
          ctx.lineWidth = 1 + bass;
          ctx.setLineDash(c1.alive ? [] : [4, 4]);
          ctx.beginPath(); ctx.moveTo(x1, baseY); ctx.lineTo(x2, baseY); ctx.stroke();
          ctx.setLineDash([]);
          // Arrow head
          if (c1.alive && c2.alive) {
            const ax = (x1 + x2) / 2 + 5, ay = baseY;
            ctx.fillStyle = ctx.strokeStyle;
            ctx.beginPath(); ctx.moveTo(ax, ay - 3); ctx.lineTo(ax + 6, ay); ctx.lineTo(ax, ay + 3); ctx.fill();
          }
        }
        // Draw chain nodes
        for (let i = 0; i < chainLen; i++) {
          const c = chain[i];
          const nx = c.x * w, ny = baseY;
          const r = c.alive ? 6 + mid * 8 + c.flash * 10 : 3;
          const bright = c.alive ? 0.5 + energy * 0.4 + c.flash * 0.3 : 0.1;
          // Wrap rings (monad layers)
          if (c.wrap > 0 && c.alive) {
            const rings = Math.min(3, Math.floor(c.wrap * 3) + 1);
            for (let ring = 0; ring < rings; ring++) {
              const rr = r + 6 + ring * 8 + Math.sin(ts / 1000 * (2 + ring)) * 2;
              const ra = (0.1 + c.wrap * 0.15) / (ring + 1);
              ctx.strokeStyle = 'rgba(' + GOLD[0] + ',' + GOLD[1] + ',' + GOLD[2] + ',' + ra.toFixed(3) + ')';
              ctx.lineWidth = 1;
              ctx.beginPath(); ctx.arc(nx, ny, rr, 0, TAU); ctx.stroke();
            }
          }
          // Core
          if (c.alive) {
            const gr = ctx.createRadialGradient(nx, ny, 0, nx, ny, r * 3);
            gr.addColorStop(0, 'rgba(' + GOLD[0] + ',' + GOLD[1] + ',' + GOLD[2] + ',' + (bright * 0.4).toFixed(3) + ')');
            gr.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = gr;
            ctx.beginPath(); ctx.arc(nx, ny, r * 3, 0, TAU); ctx.fill();
          }
          ctx.fillStyle = 'rgba(' + (GOLD[0] * bright | 0) + ',' + (GOLD[1] * bright | 0) + ',' + (GOLD[2] * bright | 0) + ',' + (c.alive ? 0.9 : 0.2) + ')';
          ctx.beginPath(); ctx.arc(nx, ny, r, 0, TAU); ctx.fill();
          // Nothing: X mark
          if (!c.alive) {
            ctx.strokeStyle = 'rgba(180,60,60,0.3)';
            ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(nx - 4, ny - 4); ctx.lineTo(nx + 4, ny + 4); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(nx + 4, ny - 4); ctx.lineTo(nx - 4, ny + 4); ctx.stroke();
          }
        }
        // Labels
        ctx.globalCompositeOperation = 'source-over';
        ctx.font = '9px monospace';
        ctx.fillStyle = 'rgba(' + GOLD[0] + ',' + GOLD[1] + ',' + GOLD[2] + ',0.2)';
        ctx.textAlign = 'center';
        const labels = ['η (unit)', 'bind >>=', 'μ (join)'];
        const lx = [0.2, 0.5, 0.8];
        for (let i = 0; i < 3; i++) ctx.fillText(labels[i], lx[i] * w, h * 0.7);
        ctx.textAlign = 'start';
      } else {
        // ── Functor / Natural Transformation: two categories with mapping arrows ──
        const showBoth = isFunctor || isNatTrans;
        // Draw category C (left)
        function drawCat(cat, label, offsetX, alpha) {
          for (const m of morphs) {
            if (m[0] >= cat.length || m[1] >= cat.length) continue;
            const a = cat[m[0]], b = cat[m[1]];
            const ax = a.x * w + offsetX, ay = a.y * h;
            const bx = b.x * w + offsetX, by = b.y * h;
            ctx.strokeStyle = 'rgba(' + GOLD[0] + ',' + GOLD[1] + ',' + GOLD[2] + ',' + (alpha * (0.08 + energy * 0.12)).toFixed(3) + ')';
            ctx.lineWidth = 1 + bass * 1.5;
            ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
          }
          for (let i = 0; i < cat.length; i++) {
            const obj = cat[i];
            const nx = obj.x * w + offsetX, ny = obj.y * h;
            const isActive = Math.abs(prog * cat.length - i) < 0.6;
            const r = isActive ? 7 + mid * 10 : 4 + energy * 3;
            const bright = alpha * (isActive ? 0.7 + treble * 0.3 : 0.3 + energy * 0.2);
            if (isActive) {
              const gr = ctx.createRadialGradient(nx, ny, 0, nx, ny, r * 3);
              gr.addColorStop(0, 'rgba(' + GOLD[0] + ',' + GOLD[1] + ',' + GOLD[2] + ',' + (bright * 0.3).toFixed(3) + ')');
              gr.addColorStop(1, 'rgba(0,0,0,0)');
              ctx.fillStyle = gr;
              ctx.beginPath(); ctx.arc(nx, ny, r * 3, 0, TAU); ctx.fill();
            }
            ctx.fillStyle = 'rgba(' + (GOLD[0] * bright | 0) + ',' + (GOLD[1] * bright | 0) + ',' + (GOLD[2] * bright | 0) + ',0.85)';
            ctx.beginPath(); ctx.arc(nx, ny, r, 0, TAU); ctx.fill();
            ctx.globalCompositeOperation = 'source-over';
            ctx.font = '9px monospace';
            ctx.fillStyle = 'rgba(' + GOLD[0] + ',' + GOLD[1] + ',' + GOLD[2] + ',' + (bright * 0.4).toFixed(2) + ')';
            ctx.textAlign = 'center';
            ctx.fillText(obj.name, nx, ny + r + 12);
            ctx.globalCompositeOperation = 'lighter';
          }
          ctx.globalCompositeOperation = 'source-over';
          ctx.font = '10px monospace';
          ctx.fillStyle = 'rgba(' + GOLD[0] + ',' + GOLD[1] + ',' + GOLD[2] + ',' + (alpha * 0.2).toFixed(2) + ')';
          ctx.textAlign = 'center';
          ctx.fillText(label, (cat[2].x * w + offsetX), h * 0.62);
          ctx.globalCompositeOperation = 'lighter';
        }
        drawCat(catC, 'Category C', 0, 1.0);
        drawCat(catD, 'Category D', 0, 1.0);
        // Functor arrows between C and D
        const arrowAlpha = 0.08 + energy * 0.15 + pulseTimer * 0.2;
        for (let i = 0; i < 5; i++) {
          const cx = catC[i].x * w, cy = catC[i].y * h;
          const dx = catD[i].x * w, dy = catD[i].y * h;
          // Only show arrows based on progress
          if (prog * 5 < i && !isNatTrans) continue;
          const a = i === activeArrow ? arrowAlpha + 0.15 : arrowAlpha;
          ctx.strokeStyle = 'rgba(' + GOLD[0] + ',' + GOLD[1] + ',' + GOLD[2] + ',' + a.toFixed(3) + ')';
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 5]);
          ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(dx, dy); ctx.stroke();
          ctx.setLineDash([]);
          // Arrow head
          const angle = Math.atan2(dy - cy, dx - cx);
          const headX = dx - Math.cos(angle) * 8;
          const headY = dy - Math.sin(angle) * 8;
          ctx.fillStyle = ctx.strokeStyle;
          ctx.beginPath();
          ctx.moveTo(dx, dy);
          ctx.lineTo(headX - Math.sin(angle) * 3, headY + Math.cos(angle) * 3);
          ctx.lineTo(headX + Math.sin(angle) * 3, headY - Math.cos(angle) * 3);
          ctx.fill();
          // Natural transformation: second set of arrows (η components) in different style
          if (isNatTrans && prog > 0.4) {
            const midX = (cx + dx) / 2, midY = (cy + dy) / 2 - 15;
            const etaAlpha = (prog - 0.4) * 0.5;
            ctx.strokeStyle = 'rgba(' + GOLD[0] + ',' + GOLD[1] + ',' + (GOLD[2] + 40) + ',' + etaAlpha.toFixed(3) + ')';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(cx, cy - 10);
            ctx.quadraticCurveTo(midX, midY - 20 - Math.sin(ts / 800 + i) * 5, dx, dy - 10);
            ctx.stroke();
          }
        }
        // Functor label
        ctx.globalCompositeOperation = 'source-over';
        ctx.font = '11px monospace';
        ctx.fillStyle = 'rgba(' + GOLD[0] + ',' + GOLD[1] + ',' + GOLD[2] + ',0.15)';
        ctx.textAlign = 'center';
        const fLabel = isNatTrans ? 'η: F ⇒ G' : 'F: C → D';
        ctx.fillText(fLabel, w / 2, h * 0.12);
        ctx.textAlign = 'start';
      }
      ctx.globalCompositeOperation = 'source-over';
    }
    return { draw };
  }
  function createTasteSketch() {
    const s = {
      mode: 'phase',
      phaseCanvas: document.createElement('canvas'),
      pctx: null,
      pw: 0,
      ph: 0,
      lastX: 0,
      loopPhase: 0,
      lastStem: ''
    };
    s.pctx = s.phaseCanvas.getContext('2d', { alpha: true });
    function resize(w, h) {
      s.pw = clamp(Math.floor(w / 2), 180, 900);
      s.ph = clamp(Math.floor(h / 2), 120, 700);
      s.phaseCanvas.width = s.pw;
      s.phaseCanvas.height = s.ph;
      clearPhase();
    }
    function draw(api) {
      if (!s.pctx) return;
      if (!s.pw || !s.ph) resize(api.w, api.h);
      if (api.stem !== s.lastStem) {
        s.mode = modeFromStem(api.stem);
        s.lastStem = api.stem;
        s.lastX = 0;
        clearPhase();
      }
      if (s.mode === 'phase') drawPhase(api);
      else if (s.mode === 'loop') drawLoop(api);
      else drawGap(api);
    }
    function drawPhase(api) {
      const progress = api.duration > 0 ? clamp(api.playhead / api.duration, 0, 1) : ((api.ts * 0.00005) % 1);
      const targetX = Math.floor(progress * (s.pw - 1));
      if (targetX < s.lastX - 3) {
        clearPhase();
        s.lastX = 0;
      }
      const step = targetX >= s.lastX ? 1 : -1;
      for (let x = s.lastX; x !== targetX; x += step) {
        renderBifurcationColumn(x, api.audio);
      }
      s.lastX = targetX;
      const c = api.ctx;
      c.save();
      c.globalCompositeOperation = 'lighter';
      c.globalAlpha = 0.68;
      c.drawImage(s.phaseCanvas, 0, 0, api.w, api.h);
      const cursorX = (targetX / s.pw) * api.w;
      c.strokeStyle = `rgba(255,235,188,${0.18 + api.audio.energy * 0.25})`;
      c.lineWidth = 1;
      c.beginPath();
      c.moveTo(cursorX, 0);
      c.lineTo(cursorX, api.h);
      c.stroke();
      c.restore();
    }
    function renderBifurcationColumn(x, audio) {
      const r = 2.8 + (x / Math.max(1, s.pw - 1)) * 1.2;
      const chaos = clamp((r - 3.3) / 0.7, 0, 1);
      const scatter = chaos * (0.6 + audio.energy * 2.8);
      let y = 0.5;
      for (let i = 0; i < 140; i++) {
        y = r * y * (1 - y);
        if (i < 80) continue;
        const jitter = (Math.random() - 0.5) * scatter * 0.035;
        const py = Math.floor((1 - clamp(y + jitter, 0, 1)) * (s.ph - 1));
        const alpha = 0.22 + chaos * 0.35;
        s.pctx.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${alpha})`;
        s.pctx.fillRect(x, py, 1, 1);
      }
      s.pctx.fillStyle = 'rgba(12,11,10,0.04)';
      s.pctx.fillRect(x + 1, 0, 1, s.ph);
    }
    function drawLoop(api) {
      const c = api.ctx;
      const a = api.audio;
      s.loopPhase += api.dt * (0.6 + a.energy);
      const phi = 1.61803398875;
      const ampX = api.w * (0.22 + a.mid * 0.14);
      const ampY = api.h * (0.18 + a.treble * 0.16);
      c.save();
      c.globalCompositeOperation = 'lighter';
      c.lineWidth = 1.1;
      for (let layer = 0; layer < 3; layer++) {
        const phase = s.loopPhase + layer * 1.9;
        const hue = 26 + ((a.treble * 160 + layer * 18) | 0);
        c.strokeStyle = `hsla(${hue}, 68%, ${54 + layer * 6}%, ${0.23 + a.energy * 0.3})`;
        c.beginPath();
        const points = 540;
        for (let i = 0; i <= points; i++) {
          const t = (i / points) * TAU;
          const x = api.w * 0.5 + Math.sin(t * phi + phase) * ampX;
          const y = api.h * 0.5 + Math.sin(t * phi * phi + phase * 1.13) * ampY;
          if (i === 0) c.moveTo(x, y);
          else c.lineTo(x, y);
        }
        c.stroke();
      }
      c.restore();
    }
    function drawGap(api) {
      const c = api.ctx;
      const a = api.audio;
      const baseY = api.h * 0.55;
      const amp = api.h * (0.13 + a.energy * 0.12);
      const steps = 14;
      c.save();
      c.globalCompositeOperation = 'lighter';
      c.beginPath();
      for (let i = 0; i <= 360; i++) {
        const u = i / 360;
        const x = u * api.w;
        const v = Math.sin((u * 6 + api.ts * 0.00042) * TAU);
        const y1 = baseY - v * amp;
        const y2 = baseY - Math.round(v * steps) / steps * amp;
        if (i === 0) c.moveTo(x, y1);
        else c.lineTo(x, y1);
        if (i > 0) {
          c.fillStyle = `rgba(180,165,132,${0.04 + Math.abs(y1 - y2) / amp * (0.2 + a.bass * 0.25)})`;
          c.fillRect(x, Math.min(y1, y2), 1.1, Math.abs(y1 - y2));
        }
      }
      c.strokeStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.48 + a.energy * 0.25})`;
      c.lineWidth = 1.4;
      c.stroke();
      c.beginPath();
      for (let i = 0; i <= 360; i++) {
        const u = i / 360;
        const x = u * api.w;
        const v = Math.sin((u * 6 + api.ts * 0.00042) * TAU);
        const y = baseY - Math.round(v * steps) / steps * amp;
        if (i === 0) c.moveTo(x, y);
        else c.lineTo(x, y);
      }
      c.strokeStyle = `rgba(160,198,255,${0.34 + a.mid * 0.3})`;
      c.lineWidth = 1;
      c.stroke();
      c.restore();
    }
    function modeFromStem(stem) {
      if (/phase/i.test(stem)) return 'phase';
      if (/strange/i.test(stem) || /loop/i.test(stem)) return 'loop';
      return 'gap';
    }
    function clearPhase() {
      s.pctx.clearRect(0, 0, s.pw, s.ph);
      s.pctx.fillStyle = 'rgba(12,11,10,1)';
      s.pctx.fillRect(0, 0, s.pw, s.ph);
    }
    return { resize, draw };
  }
  function createRawSketch() {
    const s = {
      cols: 0,
      rowsMax: 0,
      cellW: 0,
      cellH: 3,
      rows: [],
      current: null,
      accum: 0,
      rule: 30,
      dual: false,
      lastStem: ''
    };
    function resize(w, h) {
      s.cols = clamp(Math.floor(w / 4), 80, 520);
      s.cellW = w / s.cols;
      s.cellH = clamp(Math.floor(h / 180), 2, 4);
      s.rowsMax = Math.ceil(h / s.cellH) + 4;
      s.rows = [];
      s.current = new Uint8Array(s.cols);
      s.current[(s.cols / 2) | 0] = 1;
      for (let i = 0; i < s.cols; i++) {
        if (Math.random() < 0.01) s.current[i] = 1;
      }
      s.accum = 0;
    }
    function draw(api) {
      if (!s.current || api.stem !== s.lastStem) {
        resize(api.w, api.h);
        setRule(api.stem);
        s.lastStem = api.stem;
      }
      const a = api.audio;
      s.accum += api.dt * (12 + a.energy * 36);
      while (s.accum >= 1) {
        pushRow(a);
        s.accum -= 1;
      }
      if (a.beat) pushRow(a);
      const c = api.ctx;
      c.save();
      c.globalCompositeOperation = 'lighter';
      for (let r = 0; r < s.rows.length; r++) {
        const row = s.rows[r];
        const y = r * s.cellH;
        const age = 1 - r / s.rowsMax;
        const alpha = Math.pow(Math.max(0, age), 1.8) * (0.22 + a.energy * 0.45);
        c.fillStyle = `rgba(${row.color[0]},${row.color[1]},${row.color[2]},${alpha})`;
        for (let x = 0; x < s.cols; x++) {
          if (!row.cells[x]) continue;
          c.fillRect(x * s.cellW, y, s.cellW + 0.25, s.cellH + 0.2);
        }
      }
      c.restore();
    }
    function setRule(stem) {
      const lower = (stem || '').toLowerCase();
      s.dual = /30x110|dual/.test(lower);
      if (/110/.test(lower) && !s.dual) s.rule = 110;
      else if (/90/.test(lower)) s.rule = 90;
      else s.rule = 30;
    }
    function pushRow(audio) {
      if (!s.current) return;
      const hue = 20 + audio.treble * 150 + audio.mid * 20;
      const sat = 0.52 + audio.mid * 0.35;
      const val = 0.44 + audio.bass * 0.4;
      const color = hsvToRgb(hue / 360, sat, val);
      s.rows.unshift({ cells: s.current.slice(0), color });
      if (s.rows.length > s.rowsMax) s.rows.pop();
      const next = new Uint8Array(s.cols);
      if (s.dual) {
        const n30 = evolveRow(s.current, 30, s.cols);
        const n110 = evolveRow(s.current, 110, s.cols);
        for (let i = 0; i < s.cols; i++) {
          next[i] = n30[i] ^ n110[i];
        }
      } else {
        const evolved = evolveRow(s.current, s.rule, s.cols);
        next.set(evolved);
      }
      s.current = next;
      if (audio.bass > 0.4 && Math.random() < 0.45) {
        const idx = (Math.random() * s.cols) | 0;
        s.current[idx] = 1;
      }
    }
    return { resize, draw };
  }
  function createComposeSketch() {
    const s = {
      cols: 0,
      rows: 18,
      colW: 0,
      matrix: [],
      caCurrent: null,
      caRows: [],
      caRule: 30,
      beatCols: [],
      accum: 0,
      lastStem: ''
    };
    function resize(w, h) {
      s.cols = clamp(Math.floor(w / 8), 70, 260);
      s.colW = w / s.cols;
      s.matrix = [];
      for (let i = 0; i < s.cols; i++) {
        s.matrix.push({ notes: new Uint8Array(s.rows), beat: 0 });
      }
      s.caCurrent = new Uint8Array(s.cols);
      s.caCurrent[(s.cols / 2) | 0] = 1;
      s.caRows = [];
      s.beatCols = new Array(s.cols).fill(0);
      s.accum = 0;
    }
    function draw(api) {
      if (!s.caCurrent || api.stem !== s.lastStem) {
        resize(api.w, api.h);
        s.caRule = /110/.test((api.stem || '').toLowerCase()) ? 110 : 30;
        s.lastStem = api.stem;
      }
      const a = api.audio;
      s.accum += api.dt * (18 + a.energy * 45);
      while (s.accum >= 1) {
        pushColumn(a);
        s.accum -= 1;
      }
      if (a.beat) {
        s.matrix[s.matrix.length - 1].beat = 1;
      }
      const c = api.ctx;
      c.save();
      c.globalCompositeOperation = 'lighter';
      const top = api.h * 0.12;
      const rollH = api.h * 0.48;
      const noteGap = rollH / (s.rows - 1);
      const caTop = api.h * 0.66;
      const caH = api.h * 0.28;
      const caRowH = Math.max(2, caH / Math.max(1, s.caRows.length || 1));
      c.lineWidth = 1;
      for (let r = 0; r < s.rows; r++) {
        const y = top + r * noteGap;
        const hot = r % 4 === 0;
        c.strokeStyle = hot ? 'rgba(255,233,190,0.08)' : 'rgba(255,233,190,0.04)';
        c.beginPath();
        c.moveTo(0, y);
        c.lineTo(api.w, y);
        c.stroke();
      }
      for (let i = 0; i < s.cols; i++) {
        const col = s.matrix[i];
        const x = i * s.colW;
        if (col.beat > 0.01) {
          c.strokeStyle = `rgba(255,247,224,${0.1 + col.beat * 0.42})`;
          c.beginPath();
          c.moveTo(x + s.colW * 0.5, 0);
          c.lineTo(x + s.colW * 0.5, api.h);
          c.stroke();
        }
        c.fillStyle = `rgba(${GOLD[0]},${GOLD[1]},${GOLD[2]},${0.12 + a.energy * 0.26})`;
        for (let r = 0; r < s.rows; r++) {
          if (!col.notes[r]) continue;
          const y = top + r * noteGap;
          c.fillRect(x + 0.6, y - 1.2, s.colW * 0.9, 2.4);
        }
      }
      for (let r = 0; r < s.caRows.length; r++) {
        const row = s.caRows[r];
        const y = caTop + r * caRowH;
        const age = 1 - r / Math.max(1, s.caRows.length);
        const alpha = 0.16 + age * 0.34;
        c.fillStyle = `rgba(200,184,145,${alpha})`;
        for (let x = 0; x < s.cols; x++) {
          if (!row[x]) continue;
          c.fillRect(x * s.colW, y, s.colW, caRowH + 0.35);
        }
      }
      c.restore();
      for (let i = 0; i < s.cols; i++) {
        s.matrix[i].beat = Math.max(0, s.matrix[i].beat - api.dt * 2.8);
      }
    }
    function pushColumn(audio) {
      s.matrix.shift();
      const notes = new Uint8Array(s.rows);
      for (let r = 0; r < s.rows; r++) {
        const bin = Math.floor((r / s.rows) * (state.freqData.length * 0.9));
        const amp = state.freqData[bin] / 255;
        if (amp > 0.22 + ((r % 3) * 0.05)) notes[s.rows - 1 - r] = 1;
      }
      s.matrix.push({ notes, beat: audio.beat ? 1 : 0 });
      s.caRows.unshift(s.caCurrent.slice(0));
      if (s.caRows.length > 56) s.caRows.pop();
      s.caCurrent = evolveRow(s.caCurrent, s.caRule, s.cols);
      if (audio.transient > 0.1) {
        const idx = (Math.random() * s.cols) | 0;
        s.caCurrent[idx] = 1;
      }
    }
    return { resize, draw };
  }
  function evolveRow(row, rule, cols) {
    const next = new Uint8Array(cols);
    for (let i = 0; i < cols; i++) {
      const left = row[(i - 1 + cols) % cols] ? 1 : 0;
      const mid = row[i] ? 1 : 0;
      const right = row[(i + 1) % cols] ? 1 : 0;
      const pattern = (left << 2) | (mid << 1) | right;
      next[i] = (rule >> pattern) & 1;
    }
    return next;
  }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function hash(x) { return fract(Math.sin(x * 91.3457) * 47453.5453); }
  function fract(x) { return x - Math.floor(x); }
  function hsvToRgb(h, s, v) {
    const i = Math.floor(h * 6), f = h * 6 - i;
    const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
    let r = v, g = t, b = p;
    switch (i % 6) {
      case 1: r = q; g = v; b = p; break;
      case 2: r = p; g = v; b = t; break;
      case 3: r = p; g = q; b = v; break;
      case 4: r = t; g = p; b = v; break;
      case 5: r = v; g = p; b = q; break;
      default: break;
    }
    return [(r * 255) | 0, (g * 255) | 0, (b * 255) | 0];
  }
})();
