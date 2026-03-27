"""
pdcs_vdw.py — PDCS computational verification: van der Waals isotherm
======================================================================

Reproduces the thermodynamic verification results from:

    "A Formal Framework for Primitive-Dependent Combination Systems (PDCS)"
    Amartya Bhushan

What this script does
---------------------
Applies PDCS to the van der Waals equation of state

    P(V) = RT / (V - b) - a / V^2

at T = 0.85 (below the critical temperature T_c = 8a/(27Rb) ~ 2.96
for a=1, b=0.1, R=1).

The analytical spinodal is located at the inflection point of P(V), where
dP/dV = 0, i.e. where -RT/(V-b)^2 + 2a/V^3 = 0.

Expected results (from paper)
------------------------------
- Spinodal recovered at V = 2.138 without prior knowledge of its location
- Spinodal classified RUL (smooth structural incompatibility, C_int silent)
- U_int ~291x above threshold, L_int ~293x above threshold at spinodal
- C_int = 0 at spinodal (correct: van der Waals is analytic, no jump)
- Second failure region (RC) near V = b: excluded volume singularity
- Two physically distinct failure types, classified differently

Physical interpretation
-----------------------
The RUL classification at the spinodal is correct: the van der Waals
isotherm is analytic through the spinodal — no jump, no derivative
singularity. The structural incompatibility arises because a single-valued
P(V) cannot describe phase coexistence without a missing primitive: the
phase fraction x (continuous), recovered by the Maxwell construction.
This matches the prediction of Theorem 8.2 Part III.

The RC classification near V = b is also correct: the excluded volume
singularity requires a discrete domain boundary label.

Threshold calibration
---------------------
Auto-calibrated from stable region V in [0.5, 1.5] at alpha=0.99.
The gas is compressible in this region, so all factors have nonzero
variability — satisfying the precondition of Remark 5.1 (paper).

Dependencies
------------
    pip install numpy scipy matplotlib
    pdcs_core.py must be in the same directory (or on PYTHONPATH)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pdcs_core import PDCS


# ── van der Waals parameters ──────────────────────────────────────────────────

a, b, R = 1.0, 0.1, 1.0
T       = 0.85    # below critical temperature

def vdw(V, T=T, a=a, b=b, R=R):
    """van der Waals equation of state."""
    return R * T / (V - b) - a / V**2

def vdw_dP(V, T=T, a=a, b=b, R=R):
    """Analytical dP/dV — used to locate spinodal exactly."""
    return -R * T / (V - b)**2 + 2.0 * a / V**3


# ── Domain and function evaluation ───────────────────────────────────────────

V = np.linspace(0.15, 3.0, 2000)
P = vdw(V)

v_spinodal = brentq(vdw_dP, 1.5, 2.5)

print("=" * 60)
print("PDCS — van der Waals isotherm")
print(f"  T={T}, a={a}, b={b}, R={R}")
print(f"  Analytical spinodal: V = {v_spinodal:.4f}")
print("=" * 60)


# ── PDCS analysis ─────────────────────────────────────────────────────────────

model = PDCS(x=V, y=P, delta=0.05)

# Threshold calibration from stable region V in [0.5, 1.5].
# Precondition (Remark 5.1): the gas is compressible here,
# so all factors have nonzero variability — auto-calibration valid.
stable_mask   = (V > 0.5) & (V < 1.5)
Uth, Lth, Cth = model.auto_thresholds(stable_mask, alpha=0.99)

labels, U, L, C = model.classify(Uth, Lth, Cth)
boundary        = model.operational_boundary(Uth, Lth, Cth)
fp              = model.fingerprint(labels, Cth=Cth)


# ── Print results ─────────────────────────────────────────────────────────────

print(f"\nOperational boundaries at V = {V[boundary].round(4)}")

print("\nClassification breakdown:")
for lbl in ['valid', 'RC', 'RUL', 'Rmisc']:
    n = int(np.sum(labels == lbl))
    print(f"  {lbl:6s}: {n:4d} points ({100*n/len(V):.1f}%)")

sp_idx = int(np.argmin(np.abs(V - v_spinodal)))
print(f"\nFactor values at analytical spinodal (V = {v_spinodal:.4f}):")
print(f"  U_int = {U[sp_idx]:.4f}  ({U[sp_idx]/Uth:.0f}x above Uth = {Uth:.4f})")
print(f"  L_int = {L[sp_idx]:.4f}  ({L[sp_idx]/Lth:.0f}x above Lth = {Lth:.4f})")
silent_str = 'SILENT (correct)' if C[sp_idx] < Cth else 'ACTIVE'
print(f"  C_int = {C[sp_idx]:.6f}  (Cth = {Cth:.6f})  {silent_str}")
print(f"  Label = {labels[sp_idx]}")
print(f"  Inside failure region: {labels[sp_idx] != 'valid'}")

if len(boundary) > 0:
    nearest = V[boundary[np.argmin(np.abs(V[boundary] - v_spinodal))]]
    print(f"  Nearest PDCS boundary: V = {nearest:.4f}  "
          f"(error = {abs(nearest - v_spinodal):.4f})")

for lbl in ['RC', 'RUL']:
    idx = np.where(labels == lbl)[0]
    if len(idx) > 0:
        print(f"\n{lbl} region spans V = [{V[idx[0]]:.4f}, {V[idx[-1]]:.4f}]")
        if lbl == 'RUL':
            inside = V[idx[0]] <= v_spinodal <= V[idx[-1]]
            print(f"  Spinodal inside RUL: {inside}  (expected: True)")

print("\nStructural fingerprint:")
print(f"  Ordering      : {fp['ordering']}")
print(f"  RUL extent    : {fp['RUL_extent']:.1%} of failure region")
print(f"  RC extent     : {fp['RC_extent']:.1%} of failure region")
print(f"  Interpretation: {fp['interpretation']}")


# ── Plot ──────────────────────────────────────────────────────────────────────

colours = {'RC': 'red', 'RUL': 'orange', 'Rmisc': 'yellow'}

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle("PDCS — van der Waals isotherm (auto thresholds)", fontsize=13)

axes[0].plot(V, P, lw=1.2)
axes[0].axhline(0, color='gray', lw=0.5)
axes[0].set_ylabel('P = f(V)')

axes[1].plot(V, U, lw=1)
axes[1].axhline(Uth, color='orange', lw=1, ls='--', label=f'Uth={Uth:.3f}')
axes[1].set_ylabel('Uint')
axes[1].legend(fontsize=8)

axes[2].plot(V, np.clip(L, 1e-8, None), lw=1)
axes[2].axhline(Lth, color='orange', lw=1, ls='--', label=f'Lth={Lth:.3f}')
axes[2].set_yscale('log')
axes[2].set_ylabel('Lint (normalised, log)')
axes[2].legend(fontsize=8)

axes[3].plot(V, C, lw=1)
axes[3].axhline(Cth, color='orange', lw=1, ls='--', label=f'Cth={Cth:.6f}')
axes[3].set_ylabel('Cint')
axes[3].set_xlabel('Molar volume V')
axes[3].legend(fontsize=8)

for ax in axes:
    for i, lbl in enumerate(labels):
        if lbl != 'valid' and lbl in colours:
            ax.axvspan(V[i], V[min(i+1, len(V)-1)],
                       alpha=0.2, color=colours[lbl], lw=0)
    ax.axvline(v_spinodal, color='green', lw=1, ls=':', alpha=0.7)

axes[0].legend(
    handles=[plt.Line2D([0], [0], color='green', ls=':', lw=1)],
    labels=[f'Spinodal V={v_spinodal:.3f}'],
    fontsize=8, loc='upper right'
)

plt.tight_layout()
plt.savefig('pdcs_vdw_auto.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: pdcs_vdw_auto.png")
plt.show()
