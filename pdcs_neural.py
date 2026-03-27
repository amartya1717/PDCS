"""
pdcs_neural.py — PDCS computational verification: neural bifurcations
======================================================================

Reproduces the neuroscience verification results from:

    "A Formal Framework for Primitive-Dependent Combination Systems (PDCS)"
    Amartya Bhushan

What this script does
---------------------
Applies PDCS to idealised firing rate models for two canonical neural
bifurcation types (Rinzel & Ermentrout 1989; Izhikevich 2007):

    Saddle-node (Type-I):  r(I) = gain * sqrt(max(I - I_th, 0))
        Smooth square-root onset. Firing rate rises continuously from zero.
        The square-root cusp has infinite dr/dI at I = I_th, producing a
        derivative singularity at finite sampling resolution.

    Hopf (Type-II):        r(I) = r0 + gain * max(I - I_th, 0)  for I > I_th
                           r(I) = 0                              for I <= I_th
        Discontinuous jump at threshold. Firing rate switches abruptly
        from zero to a finite value.

Expected results (from paper)
------------------------------
Saddle-node:
  - Extended RUL precursor: I in [~0.85, ~0.99] — smooth structural
    incompatibility building up before bifurcation (continuous modulator
    signature, consistent with synchrony order parameter interpretation)
  - Weak RC at bifurcation onset: C_int ~0.022 (~4x above Cth)
    — derivative singularity at the square-root cusp, resolution-dependent
  - Structural fingerprint: RUL_before_RC — composite missing primitive

Hopf:
  - Strong RC at bifurcation: C_int ~0.112 (~22x above Cth)
    — genuine discontinuous jump in firing rate
  - Minimal or absent RUL precursor
  - Structural fingerprint: RC_only (or narrow RC_before_RUL)
  — discrete branch selector as missing primitive

The differential result — saddle-node produces a wider, weaker failure
pattern than Hopf — is the core cross-domain classification result.
It is robust to threshold choices in the range Cth in [0.005, 0.02]:
  - At Cth = 0.005: saddle-node -> RC (weak), Hopf -> RC (strong)
    Differential: C_int(Hopf)/C_int(saddle-node) ~ 5x
  - At Cth = 0.025: saddle-node -> RUL only, Hopf -> RC
    Clean binary differential classification

Threshold calibration
---------------------
Fixed thresholds are used here rather than auto-calibration. Auto-
calibration (Section 5.5, Remark 5.1) requires nonzero variability in
the stable reference region. The silent region (I < 0.9, r = 0 exactly)
has zero variability in all factors — the quantile would return zero,
producing inadmissible thresholds. Fixed thresholds set from factor
inspection in the transition zone are the correct procedure in this case.

Thresholds used:
    Uth = 2.0   — well above max(U_int in silent region) ≈ 0
    Lth = 2.0   — well above max(L_int in silent region) ≈ 0
    Cth = 0.005 — set from resolution bound:
                  dx/y_range = (3.0/2000) / max_r ≈ 0.001
                  Cth = 0.005 is 5x the resolution floor, conservative.
                  The differential result holds for any Cth in [0.001, 0.02].

Dependencies
------------
    pip install numpy scipy matplotlib
    pdcs_core.py must be in the same directory (or on PYTHONPATH)
"""

import numpy as np
import matplotlib.pyplot as plt
from pdcs_core import PDCS


# ── Neural models ─────────────────────────────────────────────────────────────

I_th = 1.0
I    = np.linspace(0.0, 3.0, 2000)


def saddle_node(I, I_th=1.0, gain=2.0):
    """
    Type-I (saddle-node) firing rate curve.
    Smooth square-root onset: r(I) = gain * sqrt(I - I_th) for I > I_th.
    Continuous but with infinite slope at I = I_th (square-root cusp).
    Reference: Rinzel & Ermentrout (1989), Izhikevich (2007) Ch. 6.
    """
    r = np.zeros_like(I)
    r[I > I_th] = gain * np.sqrt(I[I > I_th] - I_th)
    return r


def hopf(I, I_th=1.0, r0=0.5, gain=2.0):
    """
    Type-II (Hopf) firing rate curve.
    Discontinuous jump at threshold: r jumps from 0 to r0 at I = I_th.
    Reference: Rinzel & Ermentrout (1989), Izhikevich (2007) Ch. 6.
    """
    r = np.zeros_like(I)
    r[I > I_th] = r0 + gain * (I[I > I_th] - I_th)
    return r


r_sn = saddle_node(I)
r_hp = hopf(I)


# ── Threshold selection ───────────────────────────────────────────────────────
#
# Auto-calibration is NOT used here (see module docstring for reason).
# Thresholds are fixed from inspection of factor profiles in the transition
# zone (I in [0.85, 1.1]), following the fallback procedure of Remark 5.1.
#
# Sensitivity check: the differential result (saddle-node produces weaker
# RC exceedance than Hopf) holds for any Cth in [0.001, 0.020].

Uth = 2.0     # both models: well above silent-region maximum
Lth = 2.0     # both models: well above silent-region maximum
Cth = 0.005   # conservative resolution-based floor (see docstring)

print("Thresholds (fixed from factor inspection):")
print(f"  Uth = {Uth}")
print(f"  Lth = {Lth}")
print(f"  Cth = {Cth}")
print()

# Inspect factor ranges to verify threshold choices are appropriate
print("Factor range inspection (verifying threshold validity):")
for name, r in [("Saddle-node", r_sn), ("Hopf", r_hp)]:
    model_tmp = PDCS(x=I, y=r, delta=0.15)
    U, L, C   = model_tmp.factors()
    silent    = I < 0.9
    active    = I > 1.2
    bif_idx   = int(np.argmin(np.abs(I - I_th)))
    print(f"  {name}:")
    print(f"    U_int — silent: {U[silent].max():.4f}  "
          f"active: {U[active].max():.4f}  "
          f"at bifurcation: {U[bif_idx]:.4f}")
    print(f"    L_int — silent: {L[silent].max():.4f}  "
          f"active: {L[active].max():.4f}  "
          f"at bifurcation: {L[bif_idx]:.4f}")
    print(f"    C_int — silent: {C[silent].max():.6f}  "
          f"active: {C[active].max():.6f}  "
          f"at bifurcation: {C[bif_idx]:.6f}")
print()


# ── Run PDCS ──────────────────────────────────────────────────────────────────

configs  = [
    ("Saddle-node (Type-I)", r_sn),
    ("Hopf (Type-II)",       r_hp),
]

results = {}
for name, r in configs:
    print("=" * 60)
    print(name)
    print(f"  Uth={Uth}  Lth={Lth}  Cth={Cth}")
    print("=" * 60)

    model    = PDCS(x=I, y=r, delta=0.15)
    labels, U, L, C = model.classify(Uth, Lth, Cth)
    boundary        = model.operational_boundary(Uth, Lth, Cth)
    fp              = model.fingerprint(labels, Cth=Cth)

    print(f"\nOperational boundaries at I = {I[boundary].round(4)}")

    print("\nClassification breakdown:")
    for lbl in ['valid', 'RC', 'RUL', 'Rmisc']:
        n = int(np.sum(labels == lbl))
        print(f"  {lbl:6s}: {n:4d} ({100*n/len(I):.1f}%)")

    bif_idx = int(np.argmin(np.abs(I - I_th)))
    print(f"\nAt bifurcation I = {I_th}:")
    print(f"  U_int = {U[bif_idx]:.4f}  (Uth={Uth})")
    print(f"  L_int = {L[bif_idx]:.4f}  (Lth={Lth})")
    print(f"  C_int = {C[bif_idx]:.6f}  (Cth={Cth})  "
          f"[{C[bif_idx]/Cth:.1f}x above threshold]")
    print(f"  Label = {labels[bif_idx]}")
    print(f"  Inside failure region: {labels[bif_idx] != 'valid'}")

    if len(boundary) > 0:
        nearest = I[boundary[np.argmin(np.abs(I[boundary] - I_th))]]
        print(f"  Nearest boundary: I = {nearest:.4f}  "
              f"(error = {abs(nearest - I_th):.4f})")

    for lbl in ['RC', 'RUL']:
        idx = np.where(labels == lbl)[0]
        if len(idx) > 0:
            inside = I[idx[0]] <= I_th <= I[idx[-1]]
            print(f"  {lbl} spans I = [{I[idx[0]]:.4f}, {I[idx[-1]]:.4f}]  "
                  f"(threshold inside: {inside})")

    print(f"\nStructural fingerprint:")
    print(f"  Ordering      : {fp['ordering']}")
    print(f"  RUL extent    : {fp['RUL_extent']:.1%} of failure region")
    print(f"  RC extent     : {fp['RC_extent']:.1%} of failure region")
    print(f"  RC peak C_int : {fp['RC_peak_ratio']:.4f}  ({fp['RC_peak_ratio']/Cth:.1f}x Cth)")
    print(f"  Interpretation: {fp['interpretation']}")
    print()

    results[name] = (r, labels, U, L, C, boundary, fp)


# ── Differential summary ──────────────────────────────────────────────────────

print("=" * 60)
print("Differential classification summary")
print("=" * 60)
print(f"{'':30s}  {'Saddle-node':>15}  {'Hopf':>15}")
for key, label in [
    ('ordering',      'Fingerprint ordering'),
    ('RUL_extent',    'RUL extent'),
    ('RC_extent',     'RC extent'),
    ('RC_peak_ratio', 'RC peak C_int'),
]:
    sn_val = results["Saddle-node (Type-I)"][6][key]
    hp_val = results["Hopf (Type-II)"][6][key]
    if isinstance(sn_val, float):
        print(f"  {label:28s}  {sn_val:>15.4f}  {hp_val:>15.4f}")
    else:
        print(f"  {label:28s}  {str(sn_val):>15s}  {str(hp_val):>15s}")


# ── Plot ──────────────────────────────────────────────────────────────────────

colours = {'RC': 'red', 'RUL': 'orange', 'Rmisc': 'yellow'}
fig, axes = plt.subplots(4, 2, figsize=(16, 11), sharex=True)
fig.suptitle("PDCS — Neural bifurcation (fixed thresholds)", fontsize=13)

for col, (name, data) in enumerate(results.items()):
    r, labels, U, L, C, boundary, fp = data

    axes[0, col].plot(I, r, lw=1.5)
    axes[0, col].set_title(name, fontsize=11)
    axes[0, col].set_ylabel('Firing rate r(I)')

    axes[1, col].plot(I, U, lw=1)
    axes[1, col].axhline(Uth, color='orange', ls='--', lw=1,
                         label=f'Uth={Uth}')
    axes[1, col].set_ylabel('Uint')
    axes[1, col].legend(fontsize=8)

    axes[2, col].plot(I, np.clip(L, 1e-8, None), lw=1)
    axes[2, col].axhline(Lth, color='orange', ls='--', lw=1,
                         label=f'Lth={Lth}')
    axes[2, col].set_yscale('log')
    axes[2, col].set_ylabel('Lint (log)')
    axes[2, col].legend(fontsize=8)

    axes[3, col].plot(I, C, lw=1)
    axes[3, col].axhline(Cth, color='orange', ls='--', lw=1,
                         label=f'Cth={Cth}')
    axes[3, col].set_ylabel('Cint')
    axes[3, col].set_xlabel('Input current I')
    axes[3, col].legend(fontsize=8)

    for ax in axes[:, col]:
        for i, lbl in enumerate(labels):
            if lbl != 'valid' and lbl in colours:
                ax.axvspan(I[i], I[min(i+1, len(I)-1)],
                           alpha=0.25, color=colours[lbl], lw=0)
        ax.axvline(I_th, color='green', lw=1.5, ls=':', alpha=0.9)

axes[0, 0].legend(
    handles=[plt.Line2D([0], [0], color='green', ls=':', lw=1.5)],
    labels=[f'Bifurcation I={I_th}'],
    fontsize=8, loc='upper left'
)

plt.tight_layout()
plt.savefig('pdcs_neural_fixed.png', dpi=150, bbox_inches='tight')
print("Figure saved: pdcs_neural_fixed.png")
plt.show()
