"""Importing this package registers every method. Registration happens as a side effect of
importing the modules below, so a method that is never imported is silently absent from HEADS.
"""
from src.adapt import baselines, published, rectification  # noqa: F401
from src.adapt.baselines import ESTIMATORS  # noqa: F401
from src.adapt.heads import HEADS  # noqa: F401
