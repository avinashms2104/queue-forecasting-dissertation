"""Markovian arrival process (MAP) utilities.

A MAP with m phases is given by two m x m matrices:
    D0  rates of phase changes WITHOUT an arrival (off-diagonal >= 0, diagonal < 0)
    D1  rates of phase changes WITH an arrival (all entries >= 0)
D = D0 + D1 is the generator of the phase process, so every row of D sums to 0.

Why a MAP: in a PH renewal process (PH/PH/1) the gaps between arrivals are
independent. In a MAP they can be CORRELATED: a short gap tends to be followed
by another short gap, so arrivals come in bursts. That is the one new feature
of the MAP/PH/1 stage, and the main thing the checks below test.

Special cases (used as checks in check_map_utils.py):
    Poisson(lam):      D0 = [[-lam]],  D1 = [[lam]]
    PH renewal (a,T):  D0 = T,         D1 = t a        (gaps independent)
    MMPP-2:            two-state Markov-modulated Poisson process (below)

Formulas (Neuts 1979; Lucantoni 1991):
    theta  stationary phase distribution: theta D = 0, theta 1 = 1
    lam    = theta D1 1                                   arrival rate
    phi    = theta D1 / lam                               phase just after an arrival
    P      = (-D0)^{-1} D1                                phase-to-phase at arrivals
    E[X^k] = k! phi (-D0)^{-k} 1                          gap moments
    E[X_0 X_j] = phi (-D0)^{-1} P^j (-D0)^{-1} 1          lag-j product moment
    corr_j = (E[X_0 X_j] - E[X]^2) / Var(X)

Everything here works for any MAP, so changing the arrival process later (for
example to a MAP fitted to the call-centre data) is a parameter change.
"""
import math

import numpy as np

from ph_utils import PH


class MAP:
    def __init__(self, D0, D1, name="MAP"):
        self.D0 = np.asarray(D0, dtype=float)
        self.D1 = np.asarray(D1, dtype=float)
        self.m = self.D0.shape[0]
        self.name = name
        self._validate()

        D = self.D0 + self.D1
        A = np.vstack([D.T, np.ones(self.m)])
        b = np.r_[np.zeros(self.m), 1.0]
        self.theta = np.linalg.lstsq(A, b, rcond=None)[0]           # stationary phase dist.
        self.lam = float(self.theta @ self.D1 @ np.ones(self.m))     # arrival rate
        self.phi = self.theta @ self.D1 / self.lam                   # phase right after an arrival
        self._M = np.linalg.inv(-self.D0)                            # (-D0)^{-1}
        self.P = self._M @ self.D1                                   # embedded chain at arrivals

    def _validate(self):
        D0, D1, m = self.D0, self.D1, self.m
        assert D0.shape == (m, m) and D1.shape == (m, m), "D0 and D1 must be square, same size"
        off = D0 - np.diag(np.diag(D0))
        assert (off >= -1e-12).all(), "D0 off-diagonal entries must be >= 0"
        assert (np.diag(D0) < 0).all(), "D0 diagonal entries must be < 0"
        assert (D1 >= -1e-12).all(), "D1 entries must be >= 0"
        assert np.allclose((D0 + D1).sum(axis=1), 0, atol=1e-10), "rows of D0 + D1 must sum to 0"
        assert D1.sum() > 0, "D1 must have at least one positive entry (otherwise no arrivals)"

    # ---------- exact properties of the gaps between arrivals ----------
    def moment(self, k):
        """E[X^k] of a stationary interarrival gap."""
        return math.factorial(k) * float(self.phi @ np.linalg.matrix_power(self._M, k) @ np.ones(self.m))

    @property
    def mean(self):
        return self.moment(1)                                       # equals 1 / lam

    @property
    def scv(self):
        m1, m2 = self.moment(1), self.moment(2)
        return (m2 - m1 ** 2) / m1 ** 2

    def lag_corr(self, j):
        """Correlation between gap X_0 and gap X_j (j >= 1). Zero for any renewal process."""
        m1, m2 = self.moment(1), self.moment(2)
        one = np.ones(self.m)
        e = float(self.phi @ self._M @ np.linalg.matrix_power(self.P, j) @ self._M @ one)
        return (e - m1 ** 2) / (m2 - m1 ** 2)

    def idc_limit(self):
        """Limiting index of dispersion for counts, Var(N_t)/E[N_t] as t -> infinity.

        Equals 1 for Poisson, and scv * (1 + 2 * sum_j corr_j) in general: one number
        that combines how variable the gaps are with how correlated they are.
        """
        one = np.ones(self.m)
        Pi = np.outer(one, self.theta)
        Z = np.linalg.inv(Pi - (self.D0 + self.D1))                 # fundamental matrix
        return 1.0 - 2.0 * self.lam + 2.0 * float(self.theta @ self.D1 @ Z @ self.D1 @ one) / self.lam

    # ---------- exact properties of the COUNTS of arrivals in a window ----------
    def count_variance(self, t):
        """Var(N_t): variance of the number of arrivals in a window of length t.

        Stationary MAP (Neuts 1979; Narayana & Neuts 1992), with D = D0 + D1,
        Z = (1 theta - D)^{-1} and d = theta D1:
            Var(N_t) = (lam - 2 lam^2 + 2 d Z D1 1) t - 2 d (I - e^{D t}) Z^2 D1 1
        Checked against simulated counts in check_map_levels.py.
        """
        from scipy.linalg import expm
        if t <= 0:
            return 0.0
        m, one = self.m, np.ones(self.m)
        D = self.D0 + self.D1
        Z = np.linalg.inv(np.outer(one, self.theta) - D)
        d = self.theta @ self.D1
        slope = self.lam - 2 * self.lam ** 2 + 2 * float(d @ Z @ self.D1 @ one)
        return slope * t - 2 * float(d @ (np.eye(m) - expm(D * t)) @ Z @ Z @ self.D1 @ one)

    def vtm(self, t):
        """Variance-to-mean ratio of the counts, Var(N_t) / E[N_t].

        1 for Poisson at every t. For a MAP it moves from 1 (tiny windows) to
        idc_limit() (very long windows); the shape of that curve is the dependence.
        """
        return self.count_variance(t) / (self.lam * t)

    def count_corr(self, tau, k):
        """Correlation between arrival counts in two windows of length tau, k windows apart."""
        V = self.count_variance
        cov = 0.5 * (V((k + 1) * tau) - 2 * V(k * tau) + V((k - 1) * tau))
        return cov / V(tau)

    def scaled(self, lam_new):
        """Same MAP with all rates multiplied so that the arrival rate becomes lam_new.

        This changes the time unit only: scv, correlations and burstiness are unchanged,
        just as erlang(k, 1/lam) keeps the same shape at every rho.
        """
        c = lam_new / self.lam
        return MAP(c * self.D0, c * self.D1, name=self.name)

    def as_qbd_input(self):
        """(D0, D1) for qbd_solver.solve_qbd."""
        return self.D0, self.D1

    # ---------- sampling ----------
    def sample(self, n, rng, phase0=None):
        """Simulate n CONSECUTIVE interarrival gaps (they are correlated, so order matters).

        Starting phase is drawn from phi (the stationary phase just after an arrival)
        unless phase0 is given. Returns (gaps, final_phase) so that a long path can be
        built in chunks without breaking the correlation between chunks.
        """
        m = self.m
        rates = -np.diag(self.D0)                                   # total leaving rate per phase
        # outcomes from phase i: columns 0..m-1 = hidden move to j, m..2m-1 = arrival + move to j
        Pj = np.zeros((m, 2 * m))
        for i in range(m):
            hid = self.D0[i].copy(); hid[i] = 0.0
            Pj[i, :m] = hid / rates[i]
            Pj[i, m:] = self.D1[i] / rates[i]
        cumP = np.cumsum(Pj, axis=1)
        cumP[:, -1] = 1.0

        phase = rng.choice(m, p=self.phi) if phase0 is None else int(phase0)
        gaps = np.empty(n)
        k, t = 0, 0.0
        # draw random numbers in blocks for speed
        block = max(4 * n, 1024)
        E = rng.exponential(size=block); U = rng.random(block); b = 0
        while k < n:
            if b == block:
                E = rng.exponential(size=block); U = rng.random(block); b = 0
            t += E[b] / rates[phase]
            nxt = int(np.searchsorted(cumP[phase], U[b], side="right"))
            b += 1
            if nxt >= m:                                            # an arrival
                gaps[k] = t; k += 1; t = 0.0
                phase = nxt - m
            else:
                phase = nxt
        return gaps, phase


# ---------- constructors ----------
def poisson_map(lam):
    return MAP([[-lam]], [[lam]], name=f"Poisson(rate={lam:g})")


def ph_renewal_map(ph: PH):
    """A PH renewal process written as a MAP: D0 = T, D1 = t alpha."""
    return MAP(ph.T, np.outer(ph.t, ph.alpha), name=f"renewal[{ph.name}]")


def mmpp2(lam1, lam2, r12, r21, name=None):
    """Two-state Markov-modulated Poisson process.

    The process sits in a 'busy' state (arrivals at rate lam1) or a 'quiet' state
    (rate lam2), switching busy->quiet at rate r12 and quiet->busy at rate r21.
    """
    D1 = np.diag([lam1, lam2])
    D0 = np.array([[-(lam1 + r12), r12],
                   [r21, -(lam2 + r21)]])
    return MAP(D0, D1, name=name or f"MMPP2(l1={lam1:g}, l2={lam2:g}, r12={r12:g}, r21={r21:g})")


def bursty_mmpp2(lam, ratio=4.0, busy_share=0.25, mean_busy_arrivals=10.0):
    """MMPP-2 with arrival rate lam, described by three interpretable numbers.

    ratio               arrival rate in the busy state / rate in the quiet state
    busy_share          long-run fraction of time spent in the busy state
    mean_busy_arrivals  average number of arrivals during one busy spell
                        (larger = longer bursts = stronger correlation between gaps)

    Every rate is proportional to lam, so the shape of the process (scv, correlations)
    is the same at every rho; only the time scale changes.

    Defaults (ratio 4, busy 25% of the time, 10 arrivals per burst) give gaps with
    scv 1.68 and lag-1 correlation 0.165, and a limiting index of dispersion of 4.6:
    clearly burstier than Poisson (1.0) and far burstier than Erlang-2 renewal (0.5),
    while keeping L_q at rho = 0.95 (about 47) within reach of 20,000-unit runs.
    """
    p = busy_share
    lam2 = lam / (p * ratio + (1 - p))           # quiet rate, so that p*lam1 + (1-p)*lam2 = lam
    lam1 = ratio * lam2
    r12 = lam1 / mean_busy_arrivals              # busy spell lasts mean_busy_arrivals / lam1
    r21 = r12 * p / (1 - p)                      # gives time share p in the busy state
    return mmpp2(lam1, lam2, r12, r21,
                 name=f"MMPP2(rate={lam:g}, ratio={ratio:g}, busy={p:g}, burst={mean_busy_arrivals:g})")

# ---------- the three controlled burst levels (Azam, 12 Oct) ----------
# Azam: "the three MAP models have approximately the same mean arrival rate and
# arrival-count variability (measured over a specified time interval), but
# different dependence structures."
#
# All three are MMPP-2 (bursty_mmpp2) and are matched EXACTLY on
#   - the arrival rate                          lam
#   - the variability of a single gap           gap scv = 1.684 (same as H2/PH/1)
#   - the arrival-count variability             VtM over MATCH_WINDOW mean gaps = 3.486
# They differ only in how long a burst lasts (arrivals per busy spell), i.e. in how
# long the dependence lasts: weak = short memory, strong = long memory.
# 'medium' is the original MAP/PH/1 process, unchanged, so its data are reused.
# For weak and strong, busy_share and ratio are solved so that both matches hold.
MATCH_WINDOW = 10.0          # in mean interarrival times (= 10 / lam time units)

MAP_LEVELS = {
    "weak":   dict(mean_busy_arrivals=5.0,  p_bracket=(0.03, 0.12)),
    "medium": dict(mean_busy_arrivals=10.0, ratio=4.0, busy_share=0.25),
    "strong": dict(mean_busy_arrivals=40.0, p_bracket=(0.36, 0.50)),
}
_LEVEL_CACHE = {}


def _solve_level(level):
    """(busy_share, ratio) for a level, found at lam = 1 (the shape does not depend on lam)."""
    from scipy.optimize import brentq
    if level in _LEVEL_CACHE:
        return _LEVEL_CACHE[level]
    spec = MAP_LEVELS[level]
    burst = spec["mean_busy_arrivals"]
    if "ratio" in spec:
        out = (spec["busy_share"], spec["ratio"])
    else:
        ref = map_level("medium", 1.0)
        target_vtm, target_scv = ref.vtm(MATCH_WINDOW), ref.scv

        def ratio_for(p):          # ratio giving the target VtM for this busy share
            f = lambda r: bursty_mmpp2(1.0, r, p, burst).vtm(MATCH_WINDOW) - target_vtm
            return brentq(f, 1.001, 200.0, xtol=1e-13)

        g = lambda p: bursty_mmpp2(1.0, ratio_for(p), p, burst).scv - target_scv
        p = brentq(g, *spec["p_bracket"], xtol=1e-13)
        out = (p, ratio_for(p))
    _LEVEL_CACHE[level] = out
    return out


def map_level(level, lam):
    """The MMPP-2 for burst level 'weak' | 'medium' | 'strong' at arrival rate lam."""
    p, ratio = _solve_level(level)
    burst = MAP_LEVELS[level]["mean_busy_arrivals"]
    M = bursty_mmpp2(lam, ratio=ratio, busy_share=p, mean_busy_arrivals=burst)
    M.name = f"MMPP2-{level}(rate={lam:g}, ratio={ratio:.3f}, busy={p:.3f}, burst={burst:g})"
    return M



# ---------- MAPs fitted to the real call-centre arrivals (callcenter_fit_map.py) ----------
# Two stable periods were found in the weekday data (Nov-Dec 1999): a busy afternoon and a
# quieter evening. Each fitted MMPP-2 is stored at arrival rate 1 in
# results/callcenter_map_fit.json and is rescaled here to the rate lam = rho of the queue.
CALLCENTRE_LEVELS = ("cc_afternoon", "cc_evening")


def callcentre_map(level, lam):
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "results" / "callcenter_map_fit.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run simulation/callcenter_fit_map.py first")
    fits = json.loads(path.read_text())
    keys = sorted(fits, key=lambda k: fits[k]["period"])          # earliest period first
    key = keys[0] if level == "cc_afternoon" else keys[-1]
    f = fits[key]
    M = MAP(np.array(f["D0"]), np.array(f["D1"])).scaled(lam)
    M.name = f"CallCentre-{level[3:]}({f['period']}, rate={lam:g})"
    return M


# ---------- one table of all MAP/PH/1 arrival settings, used by every MAP/PH/1 script ----------
#   level: (arrival label written into the data, file suffix, seed tag)
LEVEL_INFO = {
    "medium": ("MMPP2", "", 77),
    "weak": ("MMPP2-weak", "_weak", 78),
    "strong": ("MMPP2-strong", "_strong", 79),
    "cc_afternoon": ("CallCentre-afternoon", "_cc_afternoon", 80),
    "cc_evening": ("CallCentre-evening", "_cc_evening", 81),
}


def arrival_map(level, lam):
    """The arrival MAP for any MAP/PH/1 setting at arrival rate lam."""
    if level in CALLCENTRE_LEVELS:
        return callcentre_map(level, lam)
    return map_level(level, lam)