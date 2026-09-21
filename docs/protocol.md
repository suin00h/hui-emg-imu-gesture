# Evaluation protocol

A person the model has never seen puts the device on, records one enrolment session, and labels one
window per gesture. Everything below follows from that.

## Split

Leave-one-subject-out. The encoder is trained on ten subjects and frozen; the eleventh is the target
and appears nowhere in training.

## Episodes

One of the target's five sessions **enrols**. From it:

- the **support set** is one labelled window per class, drawn at random (16 windows);
- the **unlabelled pool** is every window of that session, support included. Withholding the support
  would model a deployment that throws its own enrolment recording away.

The other four sessions are **scored**. Support and pool never overlap the scored windows, so an
adaptation method sees no evaluation sample.

Each session enrols in turn and the support is redrawn 12 times: up to 60 episodes per subject, 636
in total. Two subjects contribute 48 rather than 60 — see `dataset.md`.

## What a method may use

| symbol | |
|---|---|
| — | nothing from the target |
| $\mathcal{U}$ | the unlabelled pool |
| $\mathcal{S}$ | the 16 labelled windows |
| $\mathcal{S}+\mathcal{U}$ | both |

The frozen source classifier is a source artefact and is free to use. Methods that do not supply
their own read-out keep it, so a difference between rows is a difference in adaptation.

Several benchmark methods were formulated to consume the unlabelled *query* set. Here they receive
the enrolment pool instead. All of them produce class prototypes, which apply to an unseen query
unchanged, so this is a choice of which unlabelled set to feed and not a change of algorithm.

## Metric

Macro-F1 over the 16 classes, averaged over the episodes of a subject and then over subjects. Rest
windows are present in the data and excluded here.

Spread is reported two ways: over the 636 episodes, which says how much the enrolment draw matters,
and over the 11 subjects, which says how much the person matters.

## Seeds

Episode sampling is seeded by subject, session, draw and shot count, so the episodes are identical
across methods and across runs. Encoder training is seeded by fold.
