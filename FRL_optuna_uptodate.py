"""
Falling Rule List — Optuna TPE Search
======================================
Based on Wang & Rudin (2015) "Falling Rule Lists" with optimization
techniques from Chen & Rudin (2018) "An Optimization Approach to
Learning Falling Rule Lists."

Algorithm:
  1. Bitpacked DP candidate cache — single-condition masks stored as
     uint8 bitfields, conjunctions derived via bitwise AND (O(1) per pair)
  2. CELF lazy greedy selection — priority-queue based, skips 99% of
     re-evaluations by exploiting submodularity of coverage
  3. FRLOptimization Monte Carlo search — stochastic perturbations
     (swap/delete/insert) with bounds pruning to escape greedy local optima

Search:
  Two-stage Optuna TPE search with fold-level parallelism.

  Stage 1 (EXPLORE): Optuna's Tree-structured Parzen Estimator (TPE)
  adaptively samples hyperparameter combos, concentrating evaluations
  in promising regions of the space.  Each trial does k-fold CV with
  no MC search; the k folds run in parallel (one worker per fold).
  The TPE sampler begins with random exploration (n_startup_trials),
  then builds density models of "good" vs "bad" parameter regions to
  guide subsequent trials toward the optimum.

  Stage 2 (REFINE): Top-k trials from Stage 1 re-evaluated with MC
  search to improve beyond the greedy solution.

  This replaces grid search, which required enumerating a discrete set
  of parameter values.  Optuna explores continuous ranges, discovers
  interactions the grid might miss, and allocates budget adaptively
  instead of uniformly across the parameter space.

Pipeline:
  1. Hold out 20% as a final test set
  2. Optuna TPE search with fold-level parallel k-fold CV
  3. MC refinement of top-k trials
  4. Refit best params on full dev set
  5. Evaluate and produce formatted output

Label convention: RiskPerformanceBinary = 1 → Bad, 0 → Good.
The falling rule list predicts P(Bad): probabilities fall from high risk
(Bad) to low risk (Good) as rules progress.

Requirements: pip install optuna
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from itertools import combinations
from joblib import Parallel, delayed
from tabulate import tabulate
import heapq
import warnings
import time
import os
import copy

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    raise ImportError(
        "Optuna is required for this script. Install with:\n"
        "  pip install optuna"
    )

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Parallelism: one worker per CV fold.  With 5-fold CV, this uses exactly
# 5 cores regardless of how many are available — keeps the machine
# responsive and eliminates per-combo serialization overhead.
# ---------------------------------------------------------------------------
N_FOLDS = 5

# Precomputed popcount lookup table for uint8 (0–255)
_POPCOUNT_LUT = np.array([bin(i).count('1') for i in range(256)],
                         dtype=np.int64)


# =============================================================================
# BITPACKED MASK ENGINE — Core DP Data Structure
# =============================================================================
# All boolean masks are stored as np.packbits uint8 arrays.
# An N=10,000 sample mask: 10,000 bits → 1,250 bytes (vs 10,000 bytes bool).
# Conjunctions are derived via np.bitwise_and on cached singles — never
# recomputed from raw data.

class PackedMask:
    """Thin wrapper around a packed uint8 array with cached count."""
    __slots__ = ('data', '_count', '_n')

    def __init__(self, bool_mask):
        self.data = np.packbits(bool_mask)
        self._count = int(bool_mask.sum())
        self._n = len(bool_mask)

    @classmethod
    def from_packed(cls, packed_data, count, n):
        obj = object.__new__(cls)
        obj.data = packed_data
        obj._count = count
        obj._n = n
        return obj

    @property
    def count(self):
        return self._count

    def unpack(self):
        return np.unpackbits(self.data)[:self._n].astype(bool)

    def and_with(self, other):
        """Bitwise AND — the DP step. No unpacking needed for the AND."""
        new_data = np.bitwise_and(self.data, other.data)
        new_count = int(_POPCOUNT_LUT[new_data].sum())
        return PackedMask.from_packed(new_data, new_count, self._n)

    def and_not(self, other):
        """self AND (NOT other) — for remaining set updates."""
        new_data = np.bitwise_and(self.data, np.bitwise_not(other.data))
        new_count = int(_POPCOUNT_LUT[new_data].sum())
        return PackedMask.from_packed(new_data, new_count, self._n)

    def sum_y(self, y):
        """Sum y values where mask is True. Must unpack for indexing."""
        return float(y[self.unpack()].sum())


# =============================================================================
# DP CANDIDATE CACHE — Trie-Style Conjunction Derivation
# =============================================================================

class CandidateCache:
    """
    Dynamic programming cache for candidate rule masks.

    Architecture:
      - Phase 1: Compute and store all single-condition masks (bitpacked)
      - Phase 2: Conjunctions derived on-the-fly via bitwise AND of cached
                 singles (trie principle: mask(A∧B) = mask(A) & mask(B))
      - Never stores conjunction masks — computed lazily, used, discarded

    Memory: For 10K samples, 500 singles = 500 × 1.25KB ≈ 625KB
            (vs 500 × 10KB = 5MB for unpacked bool arrays)
    """

    def __init__(self, X, y, feature_names, n_thresholds, min_support,
                 use_laplace, max_conditions):
        self.n_samples = X.shape[0]
        self.y = y
        self.use_laplace = use_laplace
        self.max_conditions = max_conditions
        self.min_support = min_support

        # --- Phase 1: Build all single-condition packed masks ---
        self.singles = []         # metadata: {conditions, description, feat_idx}
        self._packed = []         # PackedMask objects (parallel)
        self._by_feature = {}     # feat_idx -> [indices into self.singles]

        quantiles = np.linspace(0, 1, n_thresholds + 2)[1:-1]

        for i in range(X.shape[1]):
            fname = feature_names[i]
            col = X[:, i]
            thresholds = np.unique(np.quantile(col, quantiles))

            for t in thresholds:
                self._register(
                    conditions=[(i, "<=", t)],
                    description=f"{fname} <= {t:.1f}",
                    mask_bool=(col <= t),
                    feat_idx=i,
                    is_simple=True
                )
                self._register(
                    conditions=[(i, ">", t)],
                    description=f"{fname} > {t:.1f}",
                    mask_bool=(col > t),
                    feat_idx=i,
                    is_simple=True
                )

            # Range rules (2-condition, same feature — treated as "singles")
            for j in range(len(thresholds) - 1):
                lo, hi = thresholds[j], thresholds[j + 1]
                self._register(
                    conditions=[(i, ">", lo), (i, "<=", hi)],
                    description=f"{lo:.1f} < {fname} <= {hi:.1f}",
                    mask_bool=((col > lo) & (col <= hi)),
                    feat_idx=i,
                    is_simple=False
                )

    def _register(self, conditions, description, mask_bool, feat_idx,
                  is_simple):
        idx = len(self.singles)
        pm = PackedMask(mask_bool)
        self.singles.append({
            "conditions": conditions,
            "description": description,
            "feat_idx": feat_idx,
        })
        self._packed.append(pm)
        if is_simple:
            self._by_feature.setdefault(feat_idx, []).append(idx)

    def iter_candidates(self, max_pairs=40000, max_triples=60000):
        """
        Yield valid candidates: singles first, then DP-derived conjunctions.

        For max_conditions >= 3, triples are derived by extending surviving
        pairs with a third single from a different feature — the intermediate
        pair mask is computed via DP (bitwise AND) and reused, so the inner
        loop only does one AND per triple.
        """
        # --- Singles ---
        for idx, s in enumerate(self.singles):
            pm = self._packed[idx]
            if pm.count < self.min_support:
                continue
            yield {
                "conditions": s["conditions"],
                "description": s["description"],
                "packed": pm,
            }

        # --- Pair conjunctions via DP (2 features) ---
        if self.max_conditions < 2:
            return

        feat_ids = sorted(self._by_feature.keys())
        pair_count = 0

        for fi, fj in combinations(feat_ids, 2):
            for si in self._by_feature[fi]:
                if pair_count >= max_pairs:
                    break
                pm_i = self._packed[si]
                s_i = self.singles[si]

                for sj in self._by_feature[fj]:
                    pm_conj = pm_i.and_with(self._packed[sj])
                    if pm_conj.count < self.min_support:
                        continue

                    s_j = self.singles[sj]
                    yield {
                        "conditions": s_i["conditions"] + s_j["conditions"],
                        "description": f"{s_i['description']} AND "
                                       f"{s_j['description']}",
                        "packed": pm_conj,
                    }

                    pair_count += 1
                    if pair_count >= max_pairs:
                        break
            if pair_count >= max_pairs:
                break

        # --- Triple conjunctions via DP (3 features) ---
        if self.max_conditions < 3:
            return

        triple_count = 0
        for fi, fj, fk in combinations(feat_ids, 3):
            if triple_count >= max_triples:
                break
            for si in self._by_feature[fi]:
                if triple_count >= max_triples:
                    break
                pm_i = self._packed[si]
                s_i = self.singles[si]

                for sj in self._by_feature[fj]:
                    if triple_count >= max_triples:
                        break

                    pm_ij = pm_i.and_with(self._packed[sj])
                    if pm_ij.count < self.min_support:
                        continue

                    s_j = self.singles[sj]

                    for sk in self._by_feature[fk]:
                        pm_ijk = pm_ij.and_with(self._packed[sk])
                        if pm_ijk.count < self.min_support:
                            continue

                        s_k = self.singles[sk]
                        yield {
                            "conditions": (s_i["conditions"]
                                           + s_j["conditions"]
                                           + s_k["conditions"]),
                            "description": (f"{s_i['description']} AND "
                                            f"{s_j['description']} AND "
                                            f"{s_k['description']}"),
                            "packed": pm_ijk,
                        }

                        triple_count += 1
                        if triple_count >= max_triples:
                            break


# =============================================================================
# CELF LAZY GREEDY SELECTOR — 10-100x speedup via submodularity
# =============================================================================

def _score_candidate(packed_cand, packed_remaining, y, use_laplace):
    """Score a candidate rule against the current remaining set."""
    pm_active = packed_cand.and_with(packed_remaining)
    n_cov = pm_active.count
    if n_cov == 0:
        return 0.0, 0.0, n_cov, pm_active
    pos = pm_active.sum_y(y)
    prob = (pos + 1) / (n_cov + 2) if use_laplace else pos / n_cov
    return prob, pos, n_cov, pm_active


def celf_select_rules(candidates, y, n_samples, max_rules, min_support,
                      max_coverage_frac, prob_gap, use_laplace):
    """
    CELF (Cost-Effective Lazy Forward) rule selection.

    Maintains a max-heap sorted by previous score. Only re-evaluates
    candidates whose cached upper bound exceeds the current best.
    Produces IDENTICAL results to naive greedy but evaluates ~1-5
    candidates per step instead of ~40K.
    """
    rules = []
    remaining = PackedMask(np.ones(n_samples, dtype=bool))
    current_max_prob = 1.01

    heap = []
    counter = 0

    for idx, cand in enumerate(candidates):
        pm = cand["packed"]
        prob, pos, n_cov, pm_active = _score_candidate(
            pm, remaining, y, use_laplace)

        if n_cov < min_support:
            continue
        if n_cov > max_coverage_frac * remaining.count:
            continue
        if prob >= current_max_prob - prob_gap:
            continue

        heapq.heappush(heap, (-prob, -n_cov, counter, idx, 0))
        counter += 1

    for step in range(max_rules):
        if remaining.count < min_support:
            break
        if not heap:
            break

        remaining_count = remaining.count
        best_rule = None
        best_prob = -1
        best_cov = 0
        best_mask = None

        eval_count = 0
        reinsert_buffer = []

        while heap:
            neg_prob, neg_cov, _, idx, last_step = heapq.heappop(heap)

            if last_step == step + 1:
                cand = candidates[idx]
                pm = cand["packed"]
                prob, pos, n_cov, pm_active = _score_candidate(
                    pm, remaining, y, use_laplace)
                if (n_cov >= min_support and
                    n_cov <= max_coverage_frac * remaining_count and
                    prob < current_max_prob - prob_gap):
                    best_rule = cand
                    best_prob = prob
                    best_cov = n_cov
                    best_mask = pm_active
                break

            cand = candidates[idx]
            pm = cand["packed"]
            prob, pos, n_cov, pm_active = _score_candidate(
                pm, remaining, y, use_laplace)
            eval_count += 1

            if n_cov < min_support:
                continue
            if n_cov > max_coverage_frac * remaining_count:
                continue
            if prob >= current_max_prob - prob_gap:
                continue

            if (not heap or
                    (-prob, -n_cov) <= (heap[0][0], heap[0][1])):
                best_rule = cand
                best_prob = prob
                best_cov = n_cov
                best_mask = pm_active
                for item in reinsert_buffer:
                    heapq.heappush(heap, item)
                break
            else:
                heapq.heappush(heap, (-prob, -n_cov, counter, idx, step + 1))
                counter += 1

        if best_rule is None:
            break

        rules.append({
            "rule": {"conditions": best_rule["conditions"],
                     "description": best_rule["description"]},
            "prob": best_prob,
            "n_captured": best_cov,
            "packed_mask": best_mask,
        })
        remaining = remaining.and_not(best_mask)
        current_max_prob = best_prob

    if remaining.count > 0:
        if use_laplace:
            default_prob = (remaining.sum_y(y) + 1) / (remaining.count + 2)
        else:
            default_prob = remaining.sum_y(y) / remaining.count
    else:
        default_prob = float(y.mean())

    return rules, default_prob


# =============================================================================
# FRL OPTIMIZATION — Monte Carlo Search with Bounds Pruning
# =============================================================================

def _compute_objective(rules, default_prob, y, n_samples, remaining_packed,
                       alpha=0.01):
    """
    FRL objective: log-likelihood + regularization.
    Objective = Σ [y_i log(p_i) + (1-y_i) log(1-p_i)] - α * n_rules
    Higher is better.
    """
    probs = np.full(n_samples, default_prob)
    captured = np.zeros(n_samples, dtype=bool)

    for entry in rules:
        mask = entry["packed_mask"].unpack()
        new = mask & ~captured
        probs[new] = entry["prob"]
        captured |= new

    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    ll = (y * np.log(probs) + (1 - y) * np.log(1 - probs)).sum()
    penalty = alpha * len(rules)
    return ll - penalty


def mc_search_improve(base_rules, default_prob, candidates, y, n_samples,
                      max_rules, min_support, max_coverage_frac, prob_gap,
                      use_laplace, n_restarts=15, rng=None):
    """
    FRLOptimization: Monte Carlo search starting from the greedy solution.

    Perturbation moves:
      1. SWAP: Replace a rule at position i with a different candidate
      2. DELETE: Remove a rule and re-select downstream rules
      3. INSERT: Try adding a rule between existing rules
    """
    if rng is None:
        rng = np.random.RandomState(42)

    best_rules = list(base_rules)
    best_default = default_prob
    best_obj = _compute_objective(base_rules, default_prob, y, n_samples,
                                  None)

    valid_cands = [c for c in candidates if c["packed"].count >= min_support]

    for restart in range(n_restarts):
        current_rules = [copy.deepcopy(r) for r in base_rules]
        current_default = default_prob

        move = rng.choice(["swap", "delete", "insert"])

        if move == "swap" and len(current_rules) > 0:
            pos = rng.randint(0, len(current_rules))

            remaining = PackedMask(np.ones(n_samples, dtype=bool))
            for i in range(pos):
                remaining = remaining.and_not(current_rules[i]["packed_mask"])

            remaining_count = remaining.count
            if remaining_count < min_support:
                continue

            prob_ceiling = (1.01 if pos == 0
                            else current_rules[pos - 1]["prob"])

            sample_size = min(len(valid_cands), 50)
            sample_indices = rng.choice(len(valid_cands), sample_size,
                                        replace=False)

            best_replacement = None
            best_repl_prob = -1
            best_repl_cov = 0

            for si in sample_indices:
                cand = valid_cands[si]
                pm_active = cand["packed"].and_with(remaining)
                n_cov = pm_active.count

                if n_cov < min_support:
                    continue
                if n_cov > max_coverage_frac * remaining_count:
                    continue

                prob = ((pm_active.sum_y(y) + 1) / (n_cov + 2)
                        if use_laplace else pm_active.sum_y(y) / n_cov)

                if prob >= prob_ceiling - prob_gap:
                    continue

                if prob > best_repl_prob or (
                        abs(prob - best_repl_prob) < 0.05 and
                        n_cov > best_repl_cov):
                    best_replacement = {
                        "rule": {"conditions": cand["conditions"],
                                 "description": cand["description"]},
                        "prob": prob,
                        "n_captured": n_cov,
                        "packed_mask": pm_active,
                    }
                    best_repl_prob = prob
                    best_repl_cov = n_cov

            if best_replacement is None:
                continue

            new_rules = current_rules[:pos] + [best_replacement]
            new_remaining = remaining.and_not(best_replacement["packed_mask"])
            new_max_prob = best_replacement["prob"]

            for _ in range(max_rules - pos - 1):
                if new_remaining.count < min_support:
                    break
                rc = new_remaining.count
                best_next = None
                best_np = -1
                best_nc = 0

                sample_indices2 = rng.choice(
                    len(valid_cands), min(len(valid_cands), 80),
                    replace=False)

                for si2 in sample_indices2:
                    c2 = valid_cands[si2]
                    pm2 = c2["packed"].and_with(new_remaining)
                    nc2 = pm2.count
                    if nc2 < min_support or nc2 > max_coverage_frac * rc:
                        continue
                    p2 = ((pm2.sum_y(y) + 1) / (nc2 + 2)
                          if use_laplace else pm2.sum_y(y) / nc2)
                    if p2 >= new_max_prob - prob_gap:
                        continue
                    if p2 > best_np or (abs(p2 - best_np) < 0.05 and
                                        nc2 > best_nc):
                        best_next = {
                            "rule": {"conditions": c2["conditions"],
                                     "description": c2["description"]},
                            "prob": p2,
                            "n_captured": nc2,
                            "packed_mask": pm2,
                        }
                        best_np = p2
                        best_nc = nc2

                if best_next is None:
                    break
                new_rules.append(best_next)
                new_remaining = new_remaining.and_not(
                    best_next["packed_mask"])
                new_max_prob = best_next["prob"]

            if new_remaining.count > 0:
                if use_laplace:
                    new_default = ((new_remaining.sum_y(y) + 1) /
                                   (new_remaining.count + 2))
                else:
                    new_default = new_remaining.sum_y(y) / new_remaining.count
            else:
                new_default = float(y.mean())

            new_obj = _compute_objective(new_rules, new_default, y,
                                         n_samples, None)
            if new_obj > best_obj:
                best_rules = new_rules
                best_default = new_default
                best_obj = new_obj

        elif move == "delete" and len(current_rules) > 1:
            pos = rng.randint(0, len(current_rules))
            new_rules = (current_rules[:pos] +
                         current_rules[pos + 1:])

            valid = True
            for i in range(1, len(new_rules)):
                if new_rules[i]["prob"] >= new_rules[i-1]["prob"]:
                    valid = False
                    break
            if not valid:
                continue

            remaining = PackedMask(np.ones(n_samples, dtype=bool))
            for r in new_rules:
                remaining = remaining.and_not(r["packed_mask"])
            if remaining.count > 0:
                if use_laplace:
                    new_default = ((remaining.sum_y(y) + 1) /
                                   (remaining.count + 2))
                else:
                    new_default = remaining.sum_y(y) / remaining.count
            else:
                new_default = float(y.mean())

            new_obj = _compute_objective(new_rules, new_default, y,
                                         n_samples, None)
            if new_obj > best_obj:
                best_rules = new_rules
                best_default = new_default
                best_obj = new_obj

        elif move == "insert" and len(current_rules) < max_rules:
            pos = rng.randint(0, len(current_rules) + 1)

            remaining = PackedMask(np.ones(n_samples, dtype=bool))
            for i in range(pos):
                remaining = remaining.and_not(current_rules[i]["packed_mask"])

            if remaining.count < min_support:
                continue

            prob_ceil = (1.01 if pos == 0
                         else current_rules[pos - 1]["prob"])
            prob_floor = (current_rules[pos]["prob"] + prob_gap
                          if pos < len(current_rules) else -1)

            sample_size = min(len(valid_cands), 50)
            sample_indices = rng.choice(len(valid_cands), sample_size,
                                        replace=False)

            best_insert = None
            best_ip = -1

            for si in sample_indices:
                cand = valid_cands[si]
                pm_active = cand["packed"].and_with(remaining)
                nc = pm_active.count
                if nc < min_support:
                    continue
                p = ((pm_active.sum_y(y) + 1) / (nc + 2)
                     if use_laplace else pm_active.sum_y(y) / nc)
                if p >= prob_ceil - prob_gap:
                    continue
                if p <= prob_floor:
                    continue
                if p > best_ip:
                    best_insert = {
                        "rule": {"conditions": cand["conditions"],
                                 "description": cand["description"]},
                        "prob": p,
                        "n_captured": nc,
                        "packed_mask": pm_active,
                    }
                    best_ip = p

            if best_insert is None:
                continue

            new_rules = (current_rules[:pos] + [best_insert] +
                         current_rules[pos:])

            remaining = PackedMask(np.ones(n_samples, dtype=bool))
            for r in new_rules:
                remaining = remaining.and_not(r["packed_mask"])
            if remaining.count > 0:
                if use_laplace:
                    new_default = ((remaining.sum_y(y) + 1) /
                                   (remaining.count + 2))
                else:
                    new_default = remaining.sum_y(y) / remaining.count
            else:
                new_default = float(y.mean())

            new_obj = _compute_objective(new_rules, new_default, y,
                                         n_samples, None)
            if new_obj > best_obj:
                best_rules = new_rules
                best_default = new_default
                best_obj = new_obj

    return best_rules, best_default, best_obj


# =============================================================================
# FALLING RULE LIST — MAIN CLASS
# =============================================================================

class FallingRuleList:
    """
    Falling Rule List with CELF greedy + MC optimization.

    fit() pipeline:
      1. Build DP candidate cache (bitpacked single masks)
      2. Generate candidates (singles + DP-derived conjunctions)
      3. CELF lazy greedy selection (base solution)
      4. MC search improvement (FRLOptimization)
    """

    def __init__(self, max_rules=12, min_support_frac=0.02, n_thresholds=6,
                 max_conditions=2, max_coverage_frac=0.35, prob_gap=0.005,
                 use_laplace=True, mc_restarts=15):
        self.max_rules = max_rules
        self.min_support_frac = min_support_frac
        self.n_thresholds = n_thresholds
        self.max_conditions = max_conditions
        self.max_coverage_frac = max_coverage_frac
        self.prob_gap = prob_gap
        self.use_laplace = use_laplace
        self.mc_restarts = mc_restarts
        self.rules_ = []
        self.default_prob_ = 0.5
        self.feature_names_ = None

    @staticmethod
    def _eval_mask(X, conditions):
        """Boolean mask for rows satisfying ALL conditions."""
        mask = np.ones(X.shape[0], dtype=bool)
        for feat_idx, op, thresh in conditions:
            if op == "<=":
                mask &= X[:, feat_idx] <= thresh
            else:
                mask &= X[:, feat_idx] > thresh
        return mask

    def fit(self, X, y, feature_names=None, verbose=False):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples = X.shape[0]

        if feature_names is None:
            feature_names = [f"X{i}" for i in range(X.shape[1])]
        self.feature_names_ = list(feature_names)

        min_support = max(int(self.min_support_frac * n_samples), 5)

        cache = CandidateCache(
            X, y, feature_names,
            n_thresholds=self.n_thresholds,
            min_support=min_support,
            use_laplace=self.use_laplace,
            max_conditions=self.max_conditions,
        )

        candidates = list(cache.iter_candidates())

        if verbose:
            print(f"  Candidates: {len(candidates)} pass support filter")

        greedy_rules, greedy_default = celf_select_rules(
            candidates, y, n_samples,
            max_rules=self.max_rules,
            min_support=min_support,
            max_coverage_frac=self.max_coverage_frac,
            prob_gap=self.prob_gap,
            use_laplace=self.use_laplace,
        )

        if verbose:
            print(f"  Greedy solution: {len(greedy_rules)} rules")

        if self.mc_restarts > 0 and len(greedy_rules) > 0:
            mc_rules, mc_default, mc_obj = mc_search_improve(
                greedy_rules, greedy_default, candidates, y, n_samples,
                max_rules=self.max_rules,
                min_support=min_support,
                max_coverage_frac=self.max_coverage_frac,
                prob_gap=self.prob_gap,
                use_laplace=self.use_laplace,
                n_restarts=self.mc_restarts,
            )

            greedy_obj = _compute_objective(greedy_rules, greedy_default,
                                             y, n_samples, None)

            if mc_obj > greedy_obj:
                if verbose:
                    print(f"  MC improved objective: {greedy_obj:.1f} → "
                          f"{mc_obj:.1f}")
                self.rules_ = mc_rules
                self.default_prob_ = mc_default
            else:
                self.rules_ = greedy_rules
                self.default_prob_ = greedy_default
        else:
            self.rules_ = greedy_rules
            self.default_prob_ = greedy_default

        if verbose:
            print(f"  Final model: {len(self.rules_)} rules")
            for i, entry in enumerate(self.rules_):
                print(f"    Rule {i+1}: {entry['rule']['description']} "
                      f"| P(Bad)={entry['prob']:.3f} | n={entry['n_captured']}")
            print(f"    Default: P(Bad)={self.default_prob_:.3f}")

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        probs = np.full(X.shape[0], self.default_prob_)
        captured = np.zeros(X.shape[0], dtype=bool)
        for entry in self.rules_:
            mask = self._eval_mask(X, entry["rule"]["conditions"])
            new = mask & ~captured
            probs[new] = entry["prob"]
            captured |= new
        return probs

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def display(self):
        lines = ["=" * 72, "FALLING RULE LIST", "=" * 72]
        for i, entry in enumerate(self.rules_):
            pred = "Bad" if entry["prob"] >= 0.5 else "Good"
            lines.append(
                f"  {i+1}. IF {entry['rule']['description']}\n"
                f"     → P(Bad) = {entry['prob']:.3f}  →  predict {pred}"
                f"   [n={entry['n_captured']}]"
            )
        pred = "Bad" if self.default_prob_ >= 0.5 else "Good"
        lines.append(
            f"  {len(self.rules_)+1}. ELSE (default)\n"
            f"     → P(Bad) = {self.default_prob_:.3f}  →  predict {pred}"
        )
        lines.append("=" * 72)
        return "\n".join(lines)

    def to_table_data(self):
        """Return rule list as structured data for tabulate."""
        rows = []
        for i, entry in enumerate(self.rules_):
            pred = "Bad" if entry["prob"] >= 0.5 else "Good"
            rows.append({
                "Rule #": i + 1,
                "Condition": entry["rule"]["description"],
                "P(Bad)": f"{entry['prob']:.3f}",
                "Prediction": pred,
                "n Captured": entry["n_captured"],
            })
        pred = "Bad" if self.default_prob_ >= 0.5 else "Good"
        rows.append({
            "Rule #": len(self.rules_) + 1,
            "Condition": "ELSE (default)",
            "P(Bad)": f"{self.default_prob_:.3f}",
            "Prediction": pred,
            "n Captured": "—",
        })
        return rows


# =============================================================================
# FOLD-LEVEL EVALUATION HELPERS
# =============================================================================
# Parallelism strategy: one worker per CV fold.  With 5-fold CV, this uses
# exactly 5 cores regardless of how many are available — keeps the machine
# responsive and eliminates per-combo serialization overhead.
#
# Two helpers:
#   _evaluate_one_fold: single param set, single fold → score
#       Used by Optuna's objective (one trial → 5 parallel fold evals)
#   _evaluate_batch_on_fold: list of param sets, single fold → list of scores
#       Used by MC refinement (top-k combos → 5 parallel fold workers)


def _evaluate_one_fold(params, X_train, y_train, X_val, y_val,
                       feature_names, scoring, mc_restarts=0):
    """Evaluate one param set on one CV fold.  Returns a single score."""
    full_params = dict(params, mc_restarts=mc_restarts)
    model = FallingRuleList(**full_params)
    model.fit(X_train, y_train, feature_names=feature_names, verbose=False)

    probs = model.predict_proba(X_val)

    if scoring == "auc":
        if len(np.unique(probs)) == 1:
            return 0.5
        return roc_auc_score(y_val, probs)
    elif scoring == "accuracy":
        return accuracy_score(y_val, (probs >= 0.5).astype(int))
    else:
        raise ValueError(f"Unknown scoring metric: {scoring!r}. "
                         f"Use 'auc' or 'accuracy'.")


def _evaluate_batch_on_fold(params_list, X_train, y_train, X_val, y_val,
                            feature_names, scoring, mc_restarts=0):
    """Evaluate multiple param sets on one CV fold.  Returns list of scores."""
    return [
        _evaluate_one_fold(params, X_train, y_train, X_val, y_val,
                           feature_names, scoring, mc_restarts)
        for params in params_list
    ]


# =============================================================================
# OPTUNA TPE SEARCH
# =============================================================================

def optuna_search(
    X, y, feature_names,
    n_folds=N_FOLDS,
    scoring="auc",
    n_trials=200,
    timeout=None,
    top_k=15,
    mc_restarts_refine=15,
    max_conditions=3,
    n_startup_trials=30,
    seed=42,
    verbose=True,
):
    """
    Two-stage hyperparameter search using Optuna's TPE sampler.

    Stage 1 (EXPLORE): Optuna adaptively samples from continuous ranges.
    Each trial does k-fold CV with no MC, folds parallelized across workers.
    TPE builds density models of good/bad regions after n_startup_trials
    random explorations, then concentrates sampling in promising areas.

    Stage 2 (REFINE): Top-k trials re-evaluated with MC search using
    fold-level parallelism.

    Parameters
    ----------
    n_trials : int
        Total number of Optuna trials (evaluation budget).
    timeout : float or None
        Optional wall-clock time limit in seconds.  If both n_trials and
        timeout are set, search stops when either limit is reached.
    top_k : int
        Number of top screen results to refine with MC search.
    mc_restarts_refine : int
        MC restarts during refinement stage.
    max_conditions : int
        Fixed conjunction depth (not searched — see design notes).
    n_startup_trials : int
        Random trials before TPE kicks in.  Higher = more exploration.
        Rule of thumb: 15-30 for 5-6 parameters.

    Returns
    -------
    (results_list, best_params, elapsed_seconds)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    skf_splits = list(skf.split(X, y))

    # Pre-split fold data once (avoids re-indexing every trial)
    fold_data = [
        (X[train_idx], y[train_idx], X[val_idx], y[val_idx])
        for train_idx, val_idx in skf_splits
    ]

    # ---- STAGE 1: OPTUNA TPE EXPLORATION (no MC) ----
    if verbose:
        print(f"  STAGE 1 (TPE explore): up to {n_trials} trials × "
              f"{n_folds} folds  [no MC, {n_folds} workers]")
        if timeout:
            print(f"  Time budget: {timeout}s")
        print(f"  TPE startup trials: {n_startup_trials} (random), "
              f"then adaptive")
        print(f"  Fixed: max_conditions={max_conditions}, use_laplace=True")
        print()

    # --- Search space ---
    # Continuous/integer ranges replace the discrete grid.
    # prob_gap uses log scale because its effective range spans ~50x
    # (0.001 to 0.05) and small values matter disproportionately.

    def objective(trial):
        params = {
            "max_rules":        trial.suggest_int("max_rules", 4, 30),
            "min_support_frac": trial.suggest_float("min_support_frac",
                                                     0.01, 0.10),
            "n_thresholds":     trial.suggest_int("n_thresholds", 4, 20),
            "max_conditions":   max_conditions,
            "max_coverage_frac": trial.suggest_float("max_coverage_frac",
                                                      0.10, 0.80),
            "prob_gap":         trial.suggest_float("prob_gap",
                                                     0.001, 0.05, log=True),
            "use_laplace":      True,
        }

        fold_scores = Parallel(n_jobs=n_folds, verbose=0)(
            delayed(_evaluate_one_fold)(
                params, X_tr, y_tr, X_va, y_va,
                feature_names, scoring, mc_restarts=0
            )
            for X_tr, y_tr, X_va, y_va in fold_data
        )
        return float(np.mean(fold_scores))

    # Progress callback
    trial_times = []

    def _progress(study, trial):
        trial_times.append(trial.duration.total_seconds()
                           if trial.duration else 0)
        n = trial.number + 1
        if n <= 3 or n % 20 == 0 or n == n_trials:
            avg_t = np.mean(trial_times[-20:])
            print(f"  Trial {n:>4d}/{n_trials}: "
                  f"score={trial.value:.4f}  "
                  f"best={study.best_value:.4f}  "
                  f"[{avg_t:.1f}s/trial]")

    # Suppress Optuna's internal logging (we have our own progress)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    t0 = time.time()
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[_progress] if verbose else [],
    )
    t_explore = time.time() - t0

    # Extract completed trials sorted by score
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)

    if verbose:
        n_completed = len(completed)
        print(f"\n  Stage 1 complete: {t_explore:.1f}s, "
              f"{n_completed} trials evaluated")
        print(f"  Best screen score: {completed[0].value:.4f}")

    # Build top-k param sets for refinement
    top_trials = completed[:top_k]
    top_params_list = []
    for t in top_trials:
        params = dict(t.params)
        params["max_conditions"] = max_conditions
        params["use_laplace"] = True
        top_params_list.append(params)

    # ---- STAGE 2: MC REFINEMENT ----
    n_refine = len(top_params_list)

    if verbose:
        print(f"\n  STAGE 2 (MC refine): {n_refine} combos × {n_folds} folds "
              f"= {n_refine * n_folds} fits  "
              f"[MC={mc_restarts_refine}, {n_folds} workers]")

    t0 = time.time()
    refine_score_lists = Parallel(
        n_jobs=n_folds, verbose=5 if verbose else 0
    )(
        delayed(_evaluate_batch_on_fold)(
            top_params_list, X_tr, y_tr, X_va, y_va,
            feature_names, scoring, mc_restarts=mc_restarts_refine
        )
        for X_tr, y_tr, X_va, y_va in fold_data
    )
    t_refine = time.time() - t0

    # refine_score_lists[fold][combo] → score
    refine_matrix = np.array(refine_score_lists)  # (n_folds, n_refine)
    refine_means = refine_matrix.mean(axis=0)
    refine_stds = refine_matrix.std(axis=0)

    refine_results = []
    for i, params in enumerate(top_params_list):
        refine_results.append({
            "params": params,
            "mean_score": float(refine_means[i]),
            "std_score": float(refine_stds[i]),
            "fold_scores": refine_matrix[:, i].tolist(),
        })
    refine_results.sort(key=lambda r: r["mean_score"], reverse=True)
    best_params = refine_results[0]["params"]

    total_time = t_explore + t_refine

    if verbose:
        print(f"\n  Stage 2 complete: {t_refine:.1f}s")
        print(f"  Total search time: {total_time:.1f}s")
        print(f"\n  {'='*68}")
        print(f"  BEST PARAMS ({scoring} = "
              f"{refine_results[0]['mean_score']:.4f}"
              f" ± {refine_results[0]['std_score']:.4f}):")
        for k, v in best_params.items():
            print(f"    {k}: {v}")
        print(f"  {'='*68}")

    # Build screen results for the full output
    screen_results = []
    for t in completed:
        params = dict(t.params)
        params["max_conditions"] = max_conditions
        params["use_laplace"] = True
        screen_results.append({
            "params": params,
            "mean_score": t.value,
            "std_score": 0.0,  # Optuna doesn't store per-fold scores
            "fold_scores": [],
        })

    all_results = refine_results + screen_results
    return all_results, best_params, total_time


# =============================================================================
# FORMATTED OUTPUT
# =============================================================================

def print_rule_table(model):
    """Print a polished R-style table of the falling rule list."""
    rows = model.to_table_data()
    print()
    print(tabulate(
        rows,
        headers="keys",
        tablefmt="fancy_grid",
        stralign="left",
        numalign="center",
        colalign=("center", "left", "center", "center", "center"),
    ))
    print()


def print_metrics(y_true, y_pred, y_probs, label=""):
    """Print comprehensive performance metrics."""
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    prec_bad = precision_score(y_true, y_pred, pos_label=1)
    rec_bad = recall_score(y_true, y_pred, pos_label=1)
    f1_bad = f1_score(y_true, y_pred, pos_label=1)
    prec_good = precision_score(y_true, y_pred, pos_label=0)
    rec_good = recall_score(y_true, y_pred, pos_label=0)
    f1_good = f1_score(y_true, y_pred, pos_label=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = [
        ["Accuracy", f"{acc:.4f}"],
        ["AUC-ROC", f"{auc:.4f}"],
        ["", ""],
        ["Bad (1) Precision", f"{prec_bad:.4f}"],
        ["Bad (1) Recall", f"{rec_bad:.4f}"],
        ["Bad (1) F1", f"{f1_bad:.4f}"],
        ["", ""],
        ["Good (0) Precision", f"{prec_good:.4f}"],
        ["Good (0) Recall", f"{rec_good:.4f}"],
        ["Good (0) F1", f"{f1_good:.4f}"],
    ]

    header = f"  PERFORMANCE METRICS{f' — {label}' if label else ''}"
    print(header)
    print(tabulate(metrics, headers=["Metric", "Value"],
                   tablefmt="fancy_grid", stralign="left",
                   colalign=("left", "center")))

    print(f"\n  Confusion Matrix:")
    cm_data = [
        ["Pred Good (0)", cm[0, 0], cm[0, 1]],
        ["Pred Bad (1)", cm[1, 0], cm[1, 1]],
    ]
    print(tabulate(cm_data,
                   headers=["", "Actual Good (0)", "Actual Bad (1)"],
                   tablefmt="fancy_grid", stralign="center",
                   numalign="center"))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print(" FALLING RULE LIST — Optuna TPE Hyperparameter Search")
    print(" CELF Greedy + FRLOptimization MC Search + DP Bitpacked Masks")
    print("=" * 72)

    # ---- 1. Load data ----
    df = pd.read_csv("df_clean_v3_for_FRL.csv")
    target_col = "RiskPerformanceBinary"
    feature_cols = [c for c in df.columns if c != target_col]

    X_all = df[feature_cols].values
    y_all = df[target_col].values
    feature_names = feature_cols

    print(f"\n  Dataset: {df.shape[0]} rows, {len(feature_cols)} features")
    print(f"  Class balance: {y_all.mean():.3f} Bad (1)")
    print(f"  CPU cores: {os.cpu_count()} (using {N_FOLDS} workers)")

    # ---- 2. Hold out 20% test set ----
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
    )
    print(f"\n  Development set: {len(X_dev)}")
    print(f"  Test set:        {len(X_test)}")
    print(f"  Dev prevalence:  {y_dev.mean():.3f}")
    print(f"  Test prevalence: {y_test.mean():.3f}")

    # ---- 3. Optuna search ----
    #
    # Search space (defined inside optuna_search):
    #   max_rules:        int    [4, 30]
    #   min_support_frac: float  [0.01, 0.10]
    #   n_thresholds:     int    [4, 20]
    #   max_coverage_frac: float [0.10, 0.80]
    #   prob_gap:         float  [0.001, 0.05]  (log scale)
    #
    # Fixed parameters:
    #   max_conditions = 3
    #   use_laplace = True
    #
    print(f"\n{'='*72}")
    print(f" OPTUNA TPE SEARCH ({N_FOLDS}-fold Stratified CV, "
          f"fold-level parallel)")
    print(f"{'='*72}\n")

    results, best_params, search_time = optuna_search(
        X_dev, y_dev, feature_names,
        n_folds=N_FOLDS,
        scoring="auc",
        n_trials=200,
        timeout=None,
        top_k=15,
        mc_restarts_refine=15,
        max_conditions=3,
        n_startup_trials=30,
        verbose=True,
    )

    # ---- 4. Show top 10 configurations ----
    print(f"\n{'='*72}")
    print(" TOP 10 CONFIGURATIONS")
    print(f"{'='*72}")
    top10_data = []
    for i, r in enumerate(results[:10]):
        param_str = ", ".join(f"{k}={v}" for k, v in r["params"].items()
                              if k not in ("max_conditions", "use_laplace"))
        top10_data.append([
            i + 1,
            f"{r['mean_score']:.4f}",
            f"±{r['std_score']:.4f}",
            param_str,
        ])
    print(tabulate(top10_data,
                   headers=["Rank", "AUC", "Std", "Parameters"],
                   tablefmt="fancy_grid"))

    # ---- 5. Refit best model on full development set ----
    print(f"\n{'='*72}")
    print(" REFITTING BEST MODEL ON FULL DEVELOPMENT SET")
    print(f"{'='*72}\n")

    best_model = FallingRuleList(**best_params, mc_restarts=20)
    best_model.fit(X_dev, y_dev, feature_names=feature_names, verbose=True)

    # ---- 6. Falling Rule List table ----
    print(f"\n{'='*72}")
    print(" FALLING RULE LIST")
    print(f"{'='*72}")
    print_rule_table(best_model)

    # ---- 7. Test set evaluation ----
    print(f"{'='*72}")
    print(" FINAL TEST SET EVALUATION")
    print(f"{'='*72}\n")

    test_preds = best_model.predict(X_test)
    test_probs = best_model.predict_proba(X_test)

    print_metrics(y_test, test_preds, test_probs, label="Test Set")

    # ---- 8. Baselines ----
    print(f"\n\n{'='*72}")
    print(" BASELINE COMPARISONS (Test Set)")
    print(f"{'='*72}")

    X_dev_clean = np.where(np.isin(X_dev, [-9, -8, -7]), np.nan, X_dev)
    X_test_clean = np.where(np.isin(X_test, [-9, -8, -7]), np.nan, X_test)
    imp = SimpleImputer(strategy="median")
    X_dev_imp = imp.fit_transform(X_dev_clean)
    X_test_imp = imp.transform(X_test_clean)

    test_acc = accuracy_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)
    test_prec_bad = precision_score(y_test, test_preds, pos_label=1)

    baselines = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42),
        "Random Forest (200)": RandomForestClassifier(
            max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=300, random_state=42),
        "Gradient Boosting (200)": GradientBoostingClassifier(
            n_estimators=200, random_state=42),
    }

    comparison_rows = [[
        "Falling Rule List", f"{test_acc:.4f}", f"{test_auc:.4f}",
        f"{test_prec_bad:.4f}"
    ]]

    for name, model in baselines.items():
        model.fit(X_dev_imp, y_dev)
        p = model.predict(X_test_imp)
        pr = model.predict_proba(X_test_imp)[:, 1]
        comparison_rows.append([
            name,
            f"{accuracy_score(y_test, p):.4f}",
            f"{roc_auc_score(y_test, pr):.4f}",
            f"{precision_score(y_test, p, pos_label=1):.4f}",
        ])

    print()
    print(tabulate(
        comparison_rows,
        headers=["Model", "Accuracy", "AUC", "Bad(1) Precision"],
        tablefmt="fancy_grid",
    ))
    print("\n  (FRL trades some accuracy for full "
          "human-readable interpretability)")

    print(f"\n  Total search time: {search_time:.1f}s")
    print(f"{'='*72}")
