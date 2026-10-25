"""Diffusion-Jev: a Jev-style typed decision model on a DiffusionGemma-style
discrete-diffusion conversion of an AR language model."""
from .decide import decide
from .io import load
from .sampler import SamplerConfig, generate
from .schema import Choice, Noul, Score
