"""
__init__.py

Scene-Based Photometry Engine
---------------------------------------------------------

The measurement half of sedphot: build a scene from the survey catalog,
subtract measured stars, jointly solve component amplitudes (and shapes,
where the catalog declares misfit) against a bin-median-plane
background, then mask residual neighbor light, twin-fill the holes, and
integrate a curve of growth to the aperture flux -- one identical recipe
for every instrument, so residual band-to-band differences trace the
data rather than the method.

Stages, one module each: recipe (constants) -> stamp -> psf ->
components -> stars -> seats -> solve -> aperture -> engine (the
per-galaxy driver). render.py holds the shared image-model primitives,
background.py the one background estimator, sersic.py the single-Sersic
shape fit, calibrate.py the flux calibration.
"""
from __future__ import annotations
