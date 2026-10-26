# Data of the examples

`wasp18_s2.npz` holds three arrays of the TESS 2-minute light curve of WASP-18 (TIC 100100827), sector 2: `time` (days, BTJD = BJD - 2457000), `flux` (the flattened PDCSAP flux) and `flux_err` (the error of the flux, relative). 9,225 points over the first 13.04 days of the sector. The file is 173,442 bytes and its SHA-256 is `48c0741fcd703ddaad1aaaac52b884e799b2788c53c5b77d09d5d6239af0cb05`.

TESS data are public: the Mikulski Archive for Space Telescopes (MAST, https://archive.stsci.edu/missions-and-data/tess) serves them without an account. The light curve file is the product of the Science Processing Operations Center (SPOC; Jenkins et al. 2016, SPIE 9913, 99133E) of the mission (Ricker et al. 2015, JATIS 1, 014003).

| | |
|---|---|
| Target | WASP-18, TIC 100100827, Tmag 8.83 |
| Sector, camera, CCD | 2, 2, 2 |
| Product | SPOC 2-minute light curve, `tess2018234235059-s0002-0000000100100827-0121-s_lc.fits` (2,004,480 bytes) |
| Source | `https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:TESS/product/tess2018234235059-s0002-0000000100100827-0121-s_lc.fits` |
| Downloaded | 2026-10-11 |
| SHA-256 of the file | `9e83d10406a7ee0274409cd19f39857abbadfbe907e0a1f033d18c00136b0c7b` |
| Columns | `TIME`, `PDCSAP_FLUX`, `PDCSAP_FLUX_ERR` of the first extension (the presearch data conditioning flux, SAP corrected for the instrument and for the crowding), `QUALITY` |

Acknowledgement and citation, as the archive asks for data of the mission: "This paper includes data collected by the TESS mission. Funding for the TESS mission is provided by the NASA Science Mission Directorate. The data were obtained from the Mikulski Archive for Space Telescopes (MAST) at the Space Telescope Science Institute (STScI), which is operated by the Association of Universities for Research in Astronomy, Inc., under NASA contract NAS 5-26555." The DOI of the TESS light curves at MAST is 10.17909/t9-nmc8-f686. TESS data are in the public domain and the file `wasp18_s2.npz` is a derived copy of one light curve, shipped under the same terms as the data.

How the arrays were made, from the file above (astropy and wotan):

```python
import numpy as np
from astropy.io import fits
from wotan import flatten   # Hippke et al. 2019, AJ 158, 143

with fits.open("tess2018234235059-s0002-0000000100100827-0121-s_lc.fits") as hdus:
    table = hdus[1].data
time, flux, error, quality = table["TIME"], table["PDCSAP_FLUX"], table["PDCSAP_FLUX_ERR"], table["QUALITY"]
good = (quality == 0) & np.isfinite(time) & np.isfinite(flux) & np.isfinite(error)
time, flux, error = time[good], flux[good], error[good]
median = np.median(flux)
flux, error = flux / median, error / median
flat = flatten(time, flux, method="biweight", window_length=0.75, return_trend=False)   # the transits are not masked
keep = np.isfinite(flat) & (time < time[0] + 14.0)
np.savez_compressed("wasp18_s2.npz", time=time[keep], flux=flat[keep], flux_err=error[keep])
```

The biweight window of 0.75 days is three times the 2.2 hour transit and more; it removes the slow variation of the star and the instrument and leaves the transits.

The published ephemeris that the tests compare the search with is the row of WASP-18 b in the NASA Exoplanet Archive table Planetary Systems Composite Parameters (`pscomppars`, https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html), read on 2026-10-20: orbital period 0.94145223 d, mid-transit time 2456740.8056 BJD, transit duration 2.21 h, transit depth 1.041 %.
