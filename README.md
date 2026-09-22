# hui-emg-imu-gesture

[ [`Paper`](TBD) ] [ [`Data`](https://github.com/suin00h/hui-emg-imu-gesture/releases/tag/v1.0-data) ] [ [`BibTeX`](#citation) ]

![](assets/readme-header.png)

A gesture dataset for cross-user recognition, pairing surface EMG from a nine-channel forearm band
with inertial data from the upper arm. Eleven people performed sixteen gestures in five sessions
each: eleven arm commands and five finger-counting postures.

The two groups are carried by different sensors and transfer very differently to a wearer the model
has never seen. Given one labelled example per gesture, arm commands reach 84.2% macro-F1 and
counting postures 47.0%. The release includes the one-shot enrollment protocol that gap is measured
under, together with the adaptation methods compared in the paper.

## Setup

```shell
git clone https://github.com/suin00h/hui-emg-imu-gesture.git
cd hui-emg-imu-gesture
pip install -r requirements.txt

python scripts/download.py     # dataset/ (2.2 GB) and checkpoints/ours/
python scripts/verify.py       # checksums, shapes, and one published number
```

`--only S01.h5` fetches a single subject; `--skip-weights` leaves the trained encoders out.

## Data

One file per subject, `S01.h5` through `S11.h5`. Each holds raw 1 kHz EMG from nine electrodes and
100 Hz inertial data from six channels, cut into 1.5 s windows at a 0.2 s step, with a gesture label
and a session index per window. 62,776 windows in total.

Preprocessing is applied by the code rather than baked into the release, so the recordings can be
used with a different front end. See [docs/dataset.md](docs/dataset.md) for the recording procedure,
the gesture list, and known irregularities.
