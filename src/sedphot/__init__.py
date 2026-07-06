"""
__init__.py

sedphot: Multi-Archive Photometry for SED Fitting
---------------------------------------------------------

Package root. sedphot retrieves catalog photometry and images from
public archives, measures every band with one uniform recipe, and
writes schema tables ready for SED fitting. The subpackages own the
behavior: catalogs/ and images/ for retrieval, measure/ for the
image-based photometry engine, pipeline.py for orchestration, and
cli.py for the command line.
"""
from __future__ import annotations

__version__ = "0.2.0"
