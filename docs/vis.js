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
      // Use time-based fallback intensity when audio analysis returns zero
      if (api.intensity < 0.01 && state.activeAudio && !state.activeAudio.paused) {
        const t = (ts % 4000) / 4000;
        api.intensity = 0.3 + 0.2 * Math.sin(t * Math.PI * 2);
        api.audio.bass = api.intensity * 0.8;
        api.audio.mid = api.intensity * 0.6;
        api.audio.treble = api.intensity * 0.4;
        api.audio.energy = api.intensity;
        api.audio.rms = api.intensity * 0.5;
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
    if (!state.audioReady || !analyser || !state.activeAudio || state.activeAudio.paused) {
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
