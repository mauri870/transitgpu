"""The 2-minute TESS light curves of stars.json from the public archive, checked against their SHA-256, and their normalized flux. usage: python fetch.py DIRECTORY (downloads what is missing)."""

import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def stars():
    return json.loads((HERE / "stars.json").read_text())


def download(entry, directory):
    """The bytes of the file of an entry, from the directory or the archive; a file whose SHA-256 differs from the entry's is an error."""
    path = Path(directory) / entry["file"]
    if not path.exists():
        with urllib.request.urlopen(entry["url"], timeout=120) as response:
            Path(directory).mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.read())
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != entry["sha256"]:
        raise ValueError(f"{entry['file']}: SHA-256 differs from stars.json")
    return content


def light_curve(content):
    """Return (time, flux, errors) of the PDCSAP flux of a SPOC light curve file with quality flag 0, divided by its median (Jenkins et al. 2016; columns TIME, PDCSAP_FLUX, PDCSAP_FLUX_ERR, QUALITY of the LIGHTCURVE extension)."""
    from astropy.io import fits

    with fits.open(io.BytesIO(content)) as hdus:
        table = hdus[1].data
        good = (table["QUALITY"] == 0) & np.isfinite(table["TIME"]) & np.isfinite(table["PDCSAP_FLUX"]) & np.isfinite(table["PDCSAP_FLUX_ERR"])
        time, flux, error = (np.asarray(table[name][good], dtype=float) for name in ("TIME", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR"))
    median = np.median(flux)
    return time - time[0], flux / median, error / median


if __name__ == "__main__":
    for entry in stars():
        download(entry, sys.argv[1])
        print(entry["tic"], entry["sector"], "ok", flush=True)
