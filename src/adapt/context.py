from dataclasses import dataclass

import numpy as np

from src.adapt.episodes import support_centroids


@dataclass
class Context:
    """Everything a method may look at for one episode."""

    fold: object
    episode: object
    Cs: np.ndarray
    Cshot: np.ndarray

    @classmethod
    def build(cls, fold, episode, Cs=None):
        Cs = fold.source_centroids() if Cs is None else Cs
        return cls(fold, episode, Cs, support_centroids(fold.Zt, episode, fold.n_classes))

    @property
    def n_classes(self):
        return self.fold.n_classes

    def support(self):
        idx = np.concatenate([self.episode.support_idx[c] for c in range(self.n_classes)])
        y = np.concatenate([np.full(len(self.episode.support_idx[c]), c)
                            for c in range(self.n_classes)])
        return self.fold.Zt[idx], y

    def pool(self):
        """The enrolment session's windows, unlabelled. Support windows are included: a deployment
        does not throw the enrolment recording away."""
        return self.fold.Zt[self.episode.train_mask]
