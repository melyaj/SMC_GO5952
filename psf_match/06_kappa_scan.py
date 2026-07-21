"""
06_kappa_scan.py — retune kappa for kernels with excessive negative weight
==========================================================================
The kappa=1 kernels of the narrow-band and PSF-structured filters violate the
Aniano W- criterion (|W-| >~ 1: ringing, noise amplification) because their
near-monochromatic PSFs have deep nulls in Fourier space. Following Aniano+
2011 (and the kappa exploration in Tarantino psf_match.py), kappa is lowered —
a more aggressive low-pass cutoff — until |W-| < W_MAX, while checking that D
stays small.

The retained kernel (last one written by make_kernel) overwrites
kernels/{filt}_to_F2100W_kernel.fits; rerun 03 and 04 for these filters after.

Usage:
    python 06_kappa_scan.py F187N F212N F1130W F360M F300M
"""
import sys

import importlib
mk = importlib.import_module("02_make_kernels")

W_MAX = 1.0
D_MAX = 0.05
KAPPAS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

if __name__ == "__main__":
    filters = sys.argv[1:]
    results = {}
    for filt in filters:
        for kappa in KAPPAS:
            diag = mk.make_kernel(filt, kappa=kappa)
            if abs(diag["W-"]) < W_MAX and diag["D"] < D_MAX:
                results[filt] = diag
                break
        else:
            results[filt] = diag
            print(f"{filt}: WARNING — no kappa reached |W-| < {W_MAX}; "
                  f"kept kappa={diag['kappa']}")

    print("\n=== kappa scan summary ===")
    for filt, diag in results.items():
        flag = "" if abs(diag["W-"]) < W_MAX else "  <-- CHECK"
        print(f"{filt}: kappa={diag['kappa']:.2f}  D={diag['D']:.5f}  "
              f"W-={diag['W-']:.4f}{flag}")
