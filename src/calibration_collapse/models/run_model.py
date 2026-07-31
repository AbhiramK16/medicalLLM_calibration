"""Shared model-call interface.

Provider SDK details, retries, timeouts, and token/cost capture belong here.
Experiment pipelines should call this layer instead of importing provider SDKs
directly.
"""
