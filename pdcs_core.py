"""
pdcs_core.py — Canonical implementation of Primitive-Dependent Combination Systems (PDCS)
==========================================================================================

This module provides the single shared PDCS class used by all computational
verifications accompanying the paper:

    "A Formal Framework for Primitive-Dependent Combination Systems (PDCS)"
    Amartya Bhushan

Both experiment scripts (pdcs_vdw.py and pdcs_neural.py) import from this
module to guarantee that all verifications use an identical implementation.

Implementation notes vs. paper definitions
-------------------------------------------

(1) Derivative estimation (Sections 4.1–4.3)
    The paper states structural factors in terms of f'(x) and f''(x).
    In this implementation, derivatives are estimated by Savitzky–Golay
    local polynomial fitting (scipy.signal.savgol_filter, polyorder=3)
    rather than raw finite differences. SG filtering suppresses noise
    amplification that would otherwise dominate f'' at practical sampling
    resolutions. The neighbourhood scale delta controls the filter window.
    This corresponds to the "local polynomial fits" option mentioned in
    Step 1 of Corollary 8.1 (Algorithmic Corollary).

(2) Linearity factor normalisation (Definition 4.2)
    The paper defines:
        L_int(x) = |f''(x) * dx^2 / (f'(x) + eps)|

    The implementation computes L_raw identically, then divides by
    median(L_raw) over the full domain:
        L_int(x) = L_raw(x) / (median(L_raw) + eps)

    This median normalisation is a numerical stability measure. In a
    stable region where f' is nonzero and smooth, L_raw has a well-defined
    positive median, so the normalised value is O(1) in the stable regime.
    The threshold Lth is expressed in normalised units and is therefore
    dimensionless and scale-independent. The differential classification
    results are invariant to this normalisation because all comparisons
    are between L_int and Lth, both in the same normalised units.

(3) Uniformity neighbourhood (Definition 4.1)
    N_delta(x) is approximated by a sliding window of width
    ceil(delta / dx) grid points, using scipy.ndimage maximum_filter1d
    and minimum_filter1d. This is the discrete analogue of the
    sup/inf over a neighbourhood ball.

(4) Continuity factor (Definition 4.3)
    C_int(x) = |f(x + dx) - f(x)| / (f_max - f_min + eps)
    implemented via np.roll with boundary correction at the last point.

Structural classification (Section 8.5, Step 4)
------------------------------------------------
    RC    : C_int >= Cth                           (discontinuity / branch switch)
    RUL   : C_int < Cth  AND  U_int >= Uth  AND  L_int >= Lth
                                                   (smooth but structurally incompatible)
    Rmisc : C_int < Cth  AND  (U_int >= Uth OR L_int >= Lth) but not both
                                                   (partial single-factor failure)
    valid : all factors below threshold
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import maximum_filter1d, minimum_filter1d


class PDCS:
    """
    Primitive-Dependent Combination Systems — computational implementation.

    Takes an observed dependence mapping f: x -> y and computes three
    structural factors at each point:

        U_int  — local sensitivity variation (uniformity), Definition 4.1
        L_int  — curvature-to-slope ratio (linearity), median-normalised, Definition 4.2
        C_int  — step-normalised local jump (continuity), Definition 4.3

    Classifies each point as:
        valid  — all factors below threshold
        RC     — continuity failure (C_int >= Cth)
        RUL    — joint U+L failure, C silent (smooth but structurally incompatible)
        Rmisc  — single factor failure (partial)

    Operational boundaries mark transitions between valid and failure regions.

    The nature of the missing primitive is inferred from the failure class
    and the structural fingerprint of the failure region (Step 8, Corollary 8.1):
        RC alone, strong C_int     -> discrete branch selector
        RUL precursor + weak RC    -> composite (continuous modulator + discrete onset)
        RUL alone, no RC           -> purely continuous modulator

    Parameters
    ----------
    x     : array_like   Domain values (must be uniformly sampled or near-uniform)
    y     : array_like   Observed dependence mapping values f(x)
    delta : float        Neighbourhood radius for uniformity (default: 5 * dx)
    dx    : float        Sampling step (default: median(diff(x)))
    eps   : float        Regularisation constant (default: 1e-8)
    """

    def __init__(self, x, y, delta=None, dx=None, eps=1e-8):
        self.x       = np.asarray(x, dtype=float)
        self.y       = np.asarray(y, dtype=float)
        self.eps     = eps
        self.dx      = dx if dx is not None else float(np.median(np.diff(self.x)))
        self.delta   = delta if delta is not None else 5 * self.dx
        self.dy      = self._derivative(self.y)
        self.ddy     = self._derivative(self.dy)
        self.y_range = float(np.max(self.y) - np.min(self.y)) + eps

    # ------------------------------------------------------------------
    # Derivative estimation
    # ------------------------------------------------------------------

    def _derivative(self, y):
        """
        Estimate first derivative using Savitzky-Golay local polynomial
        fitting. Window width is set from delta to match the neighbourhood
        scale used in the uniformity factor. polyorder=3 provides accurate
        first and second derivative estimates without over-smoothing.
        """
        step   = self.dx
        n_pts  = max(3, int(self.delta / step))
        window = n_pts if n_pts % 2 == 1 else n_pts + 1
        window = max(window, 5)
        return savgol_filter(y, window_length=window,
                             polyorder=3, deriv=1, delta=step)

    # ------------------------------------------------------------------
    # Structural factors (Sections 4.1 – 4.3)
    # ------------------------------------------------------------------

    def uniformity(self):
        """
        U_int(x) = (max_{N_delta} f' - min_{N_delta} f') / (|f'(x)| + eps)

        Measures local spread of the slope relative to its typical magnitude.
        Large values indicate rapidly changing sensitivity — potential structural
        instability. (Definition 4.1)
        """
        n_pts   = max(3, int(self.delta / self.dx))
        loc_max = maximum_filter1d(self.dy, size=n_pts)
        loc_min = minimum_filter1d(self.dy, size=n_pts)
        return (loc_max - loc_min) / (np.abs(self.dy) + self.eps)

    def linearity(self):
        """
        L_int(x) = |f''(x) * dx^2 / (f'(x) + eps)| / median(L_raw)

        Compares the second-order term to the first-order term in the local
        Taylor expansion. Values >> 1 indicate the dependence is intrinsically
        nonlinear at scale dx. (Definition 4.2, with median normalisation —
        see module docstring note (2))
        """
        L_raw    = np.abs(self.ddy * self.dx**2 / (self.dy + self.eps))
        L_median = float(np.median(L_raw)) + self.eps
        return L_raw / L_median

    def continuity(self):
        """
        C_int(x) = |f(x + dx) - f(x)| / (f_max - f_min + eps)

        Measures the step-change in f relative to its total range. Values
        comparable to 1 indicate a macroscopic jump — branch switching or
        genuine discontinuity. (Definition 4.3)
        """
        y_shifted     = np.roll(self.y, -1)
        y_shifted[-1] = self.y[-1]          # no wrap at boundary
        return np.abs(y_shifted - self.y) / self.y_range

    def factors(self):
        """Return (U_int, L_int, C_int) arrays."""
        return self.uniformity(), self.linearity(), self.continuity()

    # ------------------------------------------------------------------
    # Threshold calibration (Section 5.5)
    # ------------------------------------------------------------------

    def auto_thresholds(self, stable_mask, alpha=0.99):
        """
        Calibrate thresholds from a known stable reference region.

        F_th = max(Q_{1-alpha}(F_int in stable region), F_resolution_th)

        The resolution floor for C_th is dx / y_range (minimum detectable
        jump). For U_th and L_th the resolution floors are negligible
        relative to the quantile values in practice.

        PRECONDITION (Remark 5.1 in paper): the stable region must have
        nonzero variability in each factor. If the stable region is exactly
        flat (f ≡ const), median(L_raw) ≈ 0 and the quantile will be
        near-zero or zero, producing an inadmissible threshold. In that
        case use fixed thresholds instead (see pdcs_neural.py).

        Parameters
        ----------
        stable_mask : boolean array   True where the domain is known to be stable
        alpha       : float           Quantile level (default 0.99)

        Returns
        -------
        Uth, Lth, Cth : floats
        """
        U, L, C = self.factors()

        # Check precondition
        if stable_mask.sum() == 0:
            raise ValueError("stable_mask selects no points — cannot calibrate.")
        if np.std(U[stable_mask]) == 0 or np.std(L[stable_mask]) == 0:
            import warnings
            warnings.warn(
                "Stable region has zero variability in U or L. "
                "Quantile threshold will be zero. Use fixed thresholds instead.",
                UserWarning
            )

        Uth = float(np.quantile(U[stable_mask], alpha))
        Lth = float(np.quantile(L[stable_mask], alpha))
        Cth_data = float(np.quantile(C[stable_mask], alpha))
        # Resolution floor: smallest detectable jump
        Cth_res  = self.dx / self.y_range
        Cth = max(Cth_data, Cth_res)

        print(f"Auto thresholds (alpha={alpha}):")
        print(f"  Uth = {Uth:.6f}")
        print(f"  Lth = {Lth:.6f}")
        print(f"  Cth = {Cth:.6f}  (data={Cth_data:.6f}, resolution floor={Cth_res:.6f})")
        return Uth, Lth, Cth

    # ------------------------------------------------------------------
    # Classification (Section 8.5, Step 4)
    # ------------------------------------------------------------------

    def classify(self, Uth, Lth, Cth):
        """
        Classify each domain point into one of four categories.

        Returns
        -------
        labels : object array   'valid', 'RC', 'RUL', or 'Rmisc'
        U, L, C : float arrays  Factor values at each point
        """
        U, L, C       = self.factors()
        labels        = np.full(len(self.x), 'valid', dtype=object)
        RC            = C >= Cth
        RUL           = (~RC) & (U >= Uth) & (L >= Lth)
        Rmisc         = (~RC) & (~RUL) & ((U >= Uth) | (L >= Lth))
        labels[RC]    = 'RC'
        labels[RUL]   = 'RUL'
        labels[Rmisc] = 'Rmisc'
        return labels, U, L, C

    def operational_boundary(self, Uth, Lth, Cth):
        """
        Return indices of grid points adjacent to valid/failure transitions.
        These are the estimated operational boundary points (Step 3, Corollary 8.1).
        """
        labels, _, _, _ = self.classify(Uth, Lth, Cth)
        is_valid        = (labels == 'valid').astype(int)
        return np.where(np.diff(is_valid) != 0)[0]

    # ------------------------------------------------------------------
    # Structural fingerprint (Section 8.5, Step 8)
    # ------------------------------------------------------------------

    def fingerprint(self, labels, Cth=None):
        """
        Compute the structural fingerprint of the failure region.

        Parameters
        ----------
        labels : object array   Output of classify()
        Cth    : float          Continuity threshold used in classify().
                                If provided, RC_peak_ratio is expressed as
                                C_int / Cth (dimensionless exceedance ratio).
                                If None, RC_peak_ratio is the raw C_int value.

        Returns a dict with:
            ordering      : 'RUL_before_RC', 'RC_before_RUL', 'RC_only',
                            'RUL_only', 'mixed', or 'none'
            RUL_extent    : fraction of failure region that is RUL
            RC_extent     : fraction of failure region that is RC
            RC_peak_ratio : max(C_int in RC) / Cth  (dimensionless)
            interpretation: plain-language summary of missing primitive type
        """
        U, L, C = self.factors()
        fail    = labels != 'valid'
        n_fail  = fail.sum()

        if n_fail == 0:
            return {'ordering': 'none', 'RUL_extent': 0.0,
                    'RC_extent': 0.0, 'RC_peak_ratio': 0.0,
                    'interpretation': 'No failure region detected.'}

        rc_idx  = np.where(labels == 'RC')[0]
        rul_idx = np.where(labels == 'RUL')[0]

        RUL_extent = len(rul_idx) / n_fail
        RC_extent  = len(rc_idx)  / n_fail

        # Spatial ordering
        if len(rc_idx) == 0:
            ordering = 'RUL_only'
        elif len(rul_idx) == 0:
            ordering = 'RC_only'
        elif rul_idx[0] < rc_idx[0]:
            ordering = 'RUL_before_RC'
        elif rc_idx[0] < rul_idx[0]:
            ordering = 'RC_before_RUL'
        else:
            ordering = 'mixed'

        C_peak_raw    = C[rc_idx].max() if len(rc_idx) > 0 else 0.0
        RC_peak_ratio = (C_peak_raw / Cth) if (Cth is not None and Cth > 0) else C_peak_raw

        # Interpret using both ordering and RC exceedance magnitude.
        # A strong RC signal (peak > 10x Cth) indicates a primarily discrete
        # missing primitive regardless of any RUL precursor extent.
        STRONG_RC = RC_peak_ratio > 10.0  # 10x above threshold

        if ordering == 'RUL_only':
            interp = "Missing primitive: purely continuous modulator."
        elif STRONG_RC and RUL_extent < 0.15:
            interp = "Missing primitive: primarily discrete branch selector (strong RC, C_int >> Cth)."
        elif STRONG_RC and RUL_extent >= 0.15:
            interp = ("Missing primitive: primarily discrete branch selector "
                      "(strong RC, C_int >> Cth). RUL precursor present — "
                      "approach dynamics visible before the discrete switch.")
        elif ordering == 'RUL_before_RC' and not STRONG_RC:
            interp = ("Missing primitive: composite — continuous modulator "
                      "during RUL precursor, weak discrete selector at RC onset. "
                      "RUL-to-RC transition marks the onset scale.")
        elif ordering in ('RC_only', 'RC_before_RUL'):
            interp = "Missing primitive: primarily discrete branch selector."
        else:
            interp = "Mixed failure pattern — inspect subregions individually."

        return {
            'ordering'      : ordering,
            'RUL_extent'    : RUL_extent,
            'RC_extent'     : RC_extent,
            'RC_peak_ratio' : RC_peak_ratio,
            'interpretation': interp,
        }
