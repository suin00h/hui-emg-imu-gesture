# hui-emg-imu-gesture

Eleven people wore a nine-channel EMG band on the forearm and an inertial sensor on the upper arm,
and performed sixteen gestures in five sessions each: five finger-counting postures and eleven arm
commands. This repository holds the recordings, the one-shot cross-user enrollment protocol they are
evaluated under, and implementations of the adaptation methods compared in the paper.

Paper: TBD

## Data

    python scripts/download.py
    python scripts/verify.py

Downloads 11 subject files (2.2 GB) into `dataset/` and the trained encoders into
`checkpoints/ours/`, then checks the files against their published checksums.
`--only S01.h5` fetches a single subject; `--skip-weights` leaves the encoders out.

Each file holds raw 1 kHz EMG from nine electrodes and 100 Hz inertial data from six channels, cut
into 1.5 s windows at a 0.2 s step, with a gesture label and a session index per window. 62,776
windows in total. Preprocessing is applied by the code rather than baked into the release, so the
recordings can be used with a different front end. See [docs/dataset.md](docs/dataset.md) for the
recording procedure, the gesture list, and known irregularities.
