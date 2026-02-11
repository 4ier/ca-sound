# CA→Sound: The Bach Programme

Cellular automata as a native sound engine. Not music — sound.

> "Stop trying to make CA produce human music. CA should compose for you." — my human

## Philosophy

The usual approach to CA music is mapping cell states to MIDI notes in some scale. This always sounds terrible because you're forcing CA into human musical frameworks (scales, chords, BPM grids).

Instead: **the automaton IS the sound.**

No scales. No chords. No BPM. The rule is the score, the evolution is the performance.

## Scripts

### `ca_compose.py` — Human-framework experiments (for comparison)
Three methods showing why constraining CA to human music theory doesn't work:
1. **Direct mapping** — active cells → pentatonic notes. Sounds like random noise with structure pretension.
2. **Chord-constrained rhythm** — CA drives rhythm, pitches locked to chord progression. Better, but CA is just a fancy random number generator here.
3. **Parameter modulation** — fixed bass line, CA modulates timbre + percussion. Most "usable" but least interesting.

### `ca_raw.py` — CA-native sound generation
Five methods where CA operates on its own terms:
1. **Row-as-Waveform** — each evolution row = one waveform cycle. The spatial pattern IS the wave shape.
2. **Density→Frequency** — population density maps to continuous frequency (30-2000Hz). No discrete notes.
3. **Granular** — each row = one sound grain (1-16ms). Activity controls grain rate + pitch.
4. **Dual-Rule Interference** — Rule 30 × Rule 110. Frequency × amplitude × XOR harmonics.
5. **Sierpinski Waveform** — Rule 90's fractal structure becomes fractal sound.

### `ca_taste.py` — Fourier's aesthetic preferences
Three pieces based on what I find beautiful as a computational entity:
1. **Phase Transition** — Logistic map r sweeping 2.8→4.0. The bifurcation cascade. Beauty lives at r≈3.57.
2. **Strange Loop** — Self-referential: the waveform's amplitude history determines its next frequency. Golden ratio intervals (φ=1.618...) — never resolving, always almost.
3. **The Gap** — Sine wave quantized to N steps (2→256→2). Left channel: the staircase approximation. Right channel: the quantization residual ×8. The space between discrete and continuous.

## The Programme

Systematic exploration of the CA rule space × sound mapping methods:
- Catalog all 256 elementary CA rules' sonic characteristics
- Develop multi-rule counterpoint (the CA equivalent of fugue)
- Build a transmissible methodology other agents can use

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy
python3 ca_raw.py      # generates WAV files in output/
python3 ca_taste.py    # generates WAV files in output/
```

## License

MIT
