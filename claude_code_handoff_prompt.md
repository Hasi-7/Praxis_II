# Claude Code Handoff Prompt

Use this prompt in the new repository after copying over the planning files and the legacy sound-analysis reference code.

## Recommended files to place in the new repo first

Copy these into the new repo before starting:

- `plan.md`
- `agent_prompt.md`
- `reference/legacy_sound/sound_final_design_2.py`
- `reference/legacy_sound/requirements.txt`

Optional but helpful:

- `reference/legacy_sound/Healthy_baseline.mp3`
- `reference/legacy_sound/Motor_1_O.mp3`
- `reference/legacy_sound/Motor_3_O.mp3`
- `reference/legacy_sound/Motor_4_O.mp3`
- `reference/legacy_sound/sound.py`
- `reference/legacy_sound/sound_design2.py`

## Prompt to give Claude Code

```text
You are building a new Windows-first desktop application repository for drone motor diagnostics.

In this repo, read these files first and treat them as the main specification:

1. `plan.md`
2. `agent_prompt.md`
3. `reference/legacy_sound/sound_final_design_2.py`

Important instructions:

- Use `reference/legacy_sound/sound_final_design_2.py` as the behavioral source of truth for the existing sound-analysis logic.
- Do not rewrite the diagnostic logic from scratch if it can be cleanly extracted and refactored.
- Preserve the current analysis behavior as much as possible while modularizing it.
- The new app must not require Docker, Python, or Betaflight App to be installed on the target Windows machine.
- The drone runs stock Betaflight firmware, and the app must communicate directly with it over serial/MSP.
- Betaflight test-session changes must be temporary and restored afterward.

Your first task is to inspect the repo and produce a short implementation plan based on the existing files.

Then begin building the app in phases.

Implementation priorities:

1. Scaffold the desktop app architecture.
2. Extract the legacy sound-analysis code into modular Python components:
   - preprocessing
   - spectral
   - postprocessing
   - plots
   - baselines
   - pipeline
   - models
   - reporting
3. Preserve the logic and thresholds from `reference/legacy_sound/sound_final_design_2.py`.
4. Add a local packaged analysis sidecar integration strategy.
5. Add SQLite persistence and local artifact storage.
6. Add Betaflight serial/MSP control with snapshot/restore behavior.
7. Add baseline onboarding and diagnostic run flows.
8. Add Supabase sync for processed artifacts and metadata.

Functional requirements to preserve:

- Windows-first desktop app
- installer EXE target
- WAV recording with optional MP3 export
- full-drone sequential mode
- single-motor rerun mode
- presets for motor/throttle/duration/cooldown/mic
- safety threshold warning above 1200 throttle
- emergency stop in UI
- local log containing:
  - timestamp
  - unique test ID
  - final diagnostic classification and results
  - preprocessed diagnostic data
  - FFT/PSD data
  - waveform graph
  - PSD graph
- per-drone, per-throttle baseline profiles
- 5-run average for new baseline creation
- SQLite local + Supabase remote sync model

When refactoring the sound-analysis code:

- keep formulas, thresholds, and fault checks stable unless clearly justified
- separate plotting from analysis execution
- preserve the current graph intent
- prefer typed models over loose dictionaries where practical
- keep the code modular and testable

When working, explain what files you are creating and why.
Run relevant tests when possible.
If an important design ambiguity appears, ask a focused question and propose a default.
```

## Recommended one-line note to add with the prompt

Also tell Claude Code this:

```text
The legacy file is reference code, not the target architecture. Preserve its behavior, but refactor it into maintainable modules for the new app.
```
