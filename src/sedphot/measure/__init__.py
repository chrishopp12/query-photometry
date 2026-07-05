"""
measure

Image-Based Photometry Engine
---------------------------------------------------------
The measurement half of sedphot: load a calibrated image, build (or accept)
a neighbor mask, subtract an annulus sky, and integrate a curve of growth to
an aperture flux -- one identical recipe for every instrument, so residual
band-to-band differences trace the data rather than the method (the
uniform_phot principle). sersic.py adds the forced single-Sersic mode.
"""
from __future__ import annotations
