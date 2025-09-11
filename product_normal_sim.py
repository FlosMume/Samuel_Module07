#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo simulation of Z = X * Y where X,Y ~ N(0,1),
and comparison to the theoretical PDF f(z) = (1/pi) * K0(|z|).
Saves a figure 'product_normal_sim.png' in the same folder.
"""

import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

def pdf_product_normal(z):
    """Theoretical PDF for Z = X * Y with X,Y ~ N(0,1) independent.
    Uses modified Bessel function of the second kind, order 0: K0(|z|).
    """
    z = np.asarray(z, dtype=float)
    eps = 1e-12
    z_safe = np.where(np.abs(z) < eps, eps, z)
    k0_vec = np.vectorize(lambda t: float(mp.besselk(0, abs(t))))
    return (1/np.pi) * k0_vec(z_safe)

def main(seed=42, n=300_000, bins=200, xlim=(-5, 5), out_png="product_normal_sim.png"):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n)
    Y = rng.standard_normal(n)
    Z = X * Y

    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(Z, bins=bins, range=xlim, density=True, alpha=0.5, label="Monte Carlo (Z=XY)")

    xs = np.linspace(xlim[0], xlim[1], 1200)
    pdf_vals = pdf_product_normal(xs)
    ax.plot(xs, pdf_vals, linewidth=2, label=r"Theory: $f_Z(z)=\frac{1}{\pi}K_0(|z|)$")

    ax.set_title("Product of Two Independent Standard Normals: Monte Carlo vs Theory")
    ax.set_xlabel("z")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"Saved figure to: {out_png}")

if __name__ == "__main__":
    main()
