"""
__init__.py

sedphot: Multi-Archive Photometry for SED Fitting
---------------------------------------------------------
Package root. sedphot retrieves catalog photometry and images from
public archives, measures every band with one uniform recipe, and
writes schema tables ready for SED fitting. Retrieval lives in
catalogs/ and images/, the image photometry engine in measure/;
pipeline.py orchestrates, remeasure.py re-reports a stored fit at a
new aperture, and cli.py is the command line.
"""
from __future__ import annotations

__version__ = "0.3.0"
