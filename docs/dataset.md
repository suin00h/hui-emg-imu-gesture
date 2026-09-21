# Dataset

Eleven people, five sessions each, wearing a nine-electrode EMG ring on the forearm and an inertial
sensor on the upper arm. Sixteen gestures: five finger-counting postures and eleven arm commands.

## Contents

| | |
|---|---|
| Subjects | 11 (S01–S11), ages 23–39, 6 male / 5 female, forearm circumference 19–27 cm |
| Sessions | 5 per subject |
| Worn arm | right, for every subject including the one left-handed participant |
| Gestures | 16, plus rest |
| Windows | 62,776 total — 24,145 gesture, 38,631 rest (5,475–6,432 per subject) |
| Size | 2.2 GB, one file per subject |

## Files

`S01.h5` … `S11.h5`, each:

| dataset | shape | dtype | |
|---|---|---|---|
| `emg` | `[N, 1500, 9]` | float32 | raw, 1 kHz, unfiltered |
| `imu` | `[N, 150, 6]` | float32 | raw, 100 Hz — `acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z` |
| `label` | `[N]` | int16 | 0–15, or −1 for rest |
| `session` | `[N]` | int8 | 1–5 |

Attributes carry the subject id, sampling rates, window length and step, channel orders, class names
and the worn arm.

## Gestures

| index | name | group |
|---|---|---|
| 0–4 | `count_1` … `count_5` | counting — held finger postures |
| 5–8 | `go_forward`, `go_backward`, `go_left`, `go_right` | arm |
| 9–10 | `raise_height`, `lower_height` | arm |
| 11–12 | `speed_up`, `speed_down` | arm |
| 13–14 | `turn_counter_clockwise`, `turn_clockwise` | arm |
| 15 | `stop_movement` | arm |

The two groups are analysed separately throughout the paper because they are read from different
things: a counting posture from which electrode is loaded, an arm command from temporal shape.

Twenty-three gestures were recorded. Seven are not published: six pointing gestures, which duplicate
the arm group's directional commands, and a thigh tap, which is not a robot command. Calibration
blocks (a clap and a hard fist that open and close every recording) are also not published.

## Sensors

Nine EMG modules in a closed ring around the proximal third of the forearm, `ch1` through `ch9`
clockwise, `ch9` adjacent to `ch1`. Module 1 is aligned to the back-of-hand direction. The ring is
held by an elastic band, so the spacing between modules is not fixed and the alignment is
approximate. The inertial sensor sits on the upper arm near the shoulder.

## How it was recorded

Each session contains blocks of a single gesture. Participants were told which gesture to perform
but **not how many repetitions to make or how fast**, so block length and repetition rate vary
within and between people.

**The sensors were not re-donned between sessions.** Session-to-session variation in this dataset
therefore reflects signal drift, fatigue and posture, but not a change of electrode placement.
Numbers computed across sessions of the same person should be read with that in mind.

## Windows

Windows are 1.5 s with a 0.2 s step, pre-cut and stored as raw signal. Each window's label is the
majority label over its samples.

**A window is filtered independently of its neighbours.** The preprocessing in `src/preprocess.py`
runs `filtfilt` within each window, which pads at the window edges, so two overlapping windows do
not agree in their overlap. Windows cannot be concatenated back into a continuous recording, and the
data cannot be re-cut at a different window length.

## Known irregularities

- **S10 session 3 contains no `count_4` window; S11 session 3 contains no `lower_height` window.**
  The block was probably too short to survive the majority vote. Those sessions are still scored but
  cannot supply a complete one-shot support set, so those two subjects contribute four enrolment
  sessions rather than five.
- File sizes split into two groups (S01–S05 around 290 MB, S06–S11 around 125 MB) because the signal
  compresses differently; the window counts are comparable.

## Licence

The recordings are released under CC BY 4.0. The code in this repository is MIT; see `LICENSE` and
`LICENSE-DATA`.
