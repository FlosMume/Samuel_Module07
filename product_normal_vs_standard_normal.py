#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo simulation of Z = X * Y where X,Y ~ N(0,1),
and comparison to both:
  - Theoretical PDF f(z) = (1/pi) * K0(|z|)
  - Standard Normal distribution N(0,1)

Saves a figure 'product_normal_vs_standard_normal.png'.
"""

import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.stats import norm

def pdf_product_normal(z):
    """Theoretical PDF for Z = X * Y with X,Y ~ N(0,1) independent.
    f_Z(z) = (1/pi) K0(|z|), with K0 the modified Bessel function of the 2nd kind.
    """
    z = np.asarray(z, dtype=float)
    eps = 1e-12
    z_safe = np.where(np.abs(z) < eps, eps, z)
    k0_vec = np.vectorize(lambda t: float(mp.besselk(0, abs(t))))
    return (1/np.pi) * k0_vec(z_safe)

def main(seed=42, n=300_000, bins=200, xlim=(-5, 5),
         out_png="product_normal_vs_standard_normal.png"):
    # Monte Carlo sampling
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n)
    Y = rng.standard_normal(n)
    Z = X * Y

    # Plot histogram of simulation
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(Z, bins=bins, range=xlim, density=True, alpha=0.5,
            label="Monte Carlo (Z=XY)")

    # Theoretical product-normal PDF
    xs = np.linspace(xlim[0], xlim[1], 1200)
    pdf_vals = pdf_product_normal(xs)
    ax.plot(xs, pdf_vals, linewidth=2,
            label=r"Theory: $f_Z(z)=\frac{1}{\pi}K_0(|z|)$")

    # Standard Normal distribution
    ax.plot(xs, norm.pdf(xs, loc=0, scale=1), 'r--', linewidth=2,
            label="Standard Normal $N(0,1)$")

    # Labels and save
    ax.set_title("Distribution of Product of Two Standard Normals vs Normal(0,1)")
    ax.set_xlabel("z")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"Saved figure to: {out_png}")

if __name__ == "__main__":
    main()
