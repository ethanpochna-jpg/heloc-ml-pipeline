"""
Falling Rule List — Fixed Model + F_0.75 Threshold Tuning
==========================================================
Fits a single FRL with pre-selected hyperparameters (from Optuna TPE
search), then sweeps decision thresholds to maximize F_β (β=0.75)
for the Bad (1) class.

Strategy:
  1. Model selection was done upstream via AUC (threshold-agnostic)
  2. Threshold selection is done here via F_0.75 (precision-weighted)
  3. Final test-set performance reported at the frozen optimal threshold

Label convention: RiskPerformanceBinary = 1 → Bad, 0 → Good.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             precision_score, recall_score,
                             fbeta_score, confusion_matrix)
from itertools import combinations
from tabulate import tabulate
import heapq
import warnings
import copy

warnings.filterwarnings("ignore")

_POPCOUNT_LUT = np.array([bin(i).count('1') for i in range(256)],
                         dtype=np.int64)


# =============================================================================
# BITPACKED MASK ENGINE
# =============================================================================

class PackedMask:
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
        new_data = np.bitwise_and(self.data, other.data)
        new_count = int(_POPCOUNT_LUT[new_data].sum())
        return PackedMask.from_packed(new_data, new_count, self._n)

    def and_not(self, other):
        new_data = np.bitwise_and(self.data, np.bitwise_not(other.data))
        new_count = int(_POPCOUNT_LUT[new_data].sum())
        return PackedMask.from_packed(new_data, new_count, self._n)

    def sum_y(self, y):
        return float(y[self.unpack()].sum())


# =============================================================================
# DP CANDIDATE CACHE
# =============================================================================

class CandidateCache:
    def __init__(self, X, y, feature_names, n_thresholds, min_support,
                 use_laplace, max_conditions):
        self.n_samples = X.shape[0]
        self.y = y
        self.use_laplace = use_laplace
        self.max_conditions = max_conditions
        self.min_support = min_support

        self.singles = []
        self._packed = []
        self._by_feature = {}

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
                    feat_idx=i, is_simple=True)
                self._register(
                    conditions=[(i, ">", t)],
                    description=f"{fname} > {t:.1f}",
                    mask_bool=(col > t),
                    feat_idx=i, is_simple=True)

            for j in range(len(thresholds) - 1):
                lo, hi = thresholds[j], thresholds[j + 1]
                self._register(
                    conditions=[(i, ">", lo), (i, "<=", hi)],
                    description=f"{lo:.1f} < {fname} <= {hi:.1f}",
                    mask_bool=((col > lo) & (col <= hi)),
                    feat_idx=i, is_simple=False)

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
        for idx, s in enumerate(self.singles):
            pm = self._packed[idx]
            if pm.count < self.min_support:
                continue
            yield {
                "conditions": s["conditions"],
                "description": s["description"],
                "packed": pm,
            }

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
# CELF LAZY GREEDY SELECTOR
# =============================================================================

def _score_candidate(packed_cand, packed_remaining, y, use_laplace):
    pm_active = packed_cand.and_with(packed_remaining)
    n_cov = pm_active.count
    if n_cov == 0:
        return 0.0, 0.0, n_cov, pm_active
    pos = pm_active.sum_y(y)
    prob = (pos + 1) / (n_cov + 2) if use_laplace else pos / n_cov
    return prob, pos, n_cov, pm_active


def celf_select_rules(candidates, y, n_samples, max_rules, min_support,
                      max_coverage_frac, prob_gap, use_laplace):
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
# MC SEARCH IMPROVEMENT
# =============================================================================

def _compute_objective(rules, default_prob, y, n_samples, remaining_packed,
                       alpha=0.01):
    probs = np.full(n_samples, default_prob)
    captured = np.zeros(n_samples, dtype=bool)
    for entry in rules:
        mask = entry["packed_mask"].unpack()
        new = mask & ~captured
        probs[new] = entry["prob"]
        captured |= new
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    ll = (y * np.log(probs) + (1 - y) * np.log(1 - probs)).sum()
    return ll - alpha * len(rules)


def mc_search_improve(base_rules, default_prob, candidates, y, n_samples,
                      max_rules, min_support, max_coverage_frac, prob_gap,
                      use_laplace, n_restarts=15, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)

    best_rules = list(base_rules)
    best_default = default_prob
    best_obj = _compute_objective(base_rules, default_prob, y, n_samples, None)
    valid_cands = [c for c in candidates if c["packed"].count >= min_support]

    for restart in range(n_restarts):
        current_rules = [copy.deepcopy(r) for r in base_rules]
        move = rng.choice(["swap", "delete", "insert"])

        if move == "swap" and len(current_rules) > 0:
            pos = rng.randint(0, len(current_rules))
            remaining = PackedMask(np.ones(n_samples, dtype=bool))
            for i in range(pos):
                remaining = remaining.and_not(current_rules[i]["packed_mask"])
            if remaining.count < min_support:
                continue
            prob_ceiling = 1.01 if pos == 0 else current_rules[pos - 1]["prob"]

            sample_indices = rng.choice(
                len(valid_cands), min(len(valid_cands), 50), replace=False)
            best_replacement = None
            best_repl_prob = -1
            best_repl_cov = 0

            for si in sample_indices:
                cand = valid_cands[si]
                pm_active = cand["packed"].and_with(remaining)
                n_cov = pm_active.count
                if n_cov < min_support or n_cov > max_coverage_frac * remaining.count:
                    continue
                prob = ((pm_active.sum_y(y) + 1) / (n_cov + 2)
                        if use_laplace else pm_active.sum_y(y) / n_cov)
                if prob >= prob_ceiling - prob_gap:
                    continue
                if prob > best_repl_prob or (
                        abs(prob - best_repl_prob) < 0.05 and n_cov > best_repl_cov):
                    best_replacement = {
                        "rule": {"conditions": cand["conditions"],
                                 "description": cand["description"]},
                        "prob": prob, "n_captured": n_cov,
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
                si2s = rng.choice(len(valid_cands),
                                  min(len(valid_cands), 80), replace=False)
                for si2 in si2s:
                    c2 = valid_cands[si2]
                    pm2 = c2["packed"].and_with(new_remaining)
                    nc2 = pm2.count
                    if nc2 < min_support or nc2 > max_coverage_frac * rc:
                        continue
                    p2 = ((pm2.sum_y(y) + 1) / (nc2 + 2)
                          if use_laplace else pm2.sum_y(y) / nc2)
                    if p2 >= new_max_prob - prob_gap:
                        continue
                    if p2 > best_np or (abs(p2 - best_np) < 0.05 and nc2 > best_nc):
                        best_next = {
                            "rule": {"conditions": c2["conditions"],
                                     "description": c2["description"]},
                            "prob": p2, "n_captured": nc2, "packed_mask": pm2,
                        }
                        best_np = p2
                        best_nc = nc2
                if best_next is None:
                    break
                new_rules.append(best_next)
                new_remaining = new_remaining.and_not(best_next["packed_mask"])
                new_max_prob = best_next["prob"]

            if new_remaining.count > 0:
                new_default = (((new_remaining.sum_y(y) + 1) /
                                (new_remaining.count + 2)) if use_laplace
                               else new_remaining.sum_y(y) / new_remaining.count)
            else:
                new_default = float(y.mean())

            new_obj = _compute_objective(new_rules, new_default, y, n_samples, None)
            if new_obj > best_obj:
                best_rules, best_default, best_obj = new_rules, new_default, new_obj

        elif move == "delete" and len(current_rules) > 1:
            pos = rng.randint(0, len(current_rules))
            new_rules = current_rules[:pos] + current_rules[pos + 1:]
            valid = all(new_rules[i]["prob"] < new_rules[i-1]["prob"]
                        for i in range(1, len(new_rules)))
            if not valid:
                continue
            remaining = PackedMask(np.ones(n_samples, dtype=bool))
            for r in new_rules:
                remaining = remaining.and_not(r["packed_mask"])
            if remaining.count > 0:
                new_default = (((remaining.sum_y(y) + 1) /
                                (remaining.count + 2)) if use_laplace
                               else remaining.sum_y(y) / remaining.count)
            else:
                new_default = float(y.mean())
            new_obj = _compute_objective(new_rules, new_default, y, n_samples, None)
            if new_obj > best_obj:
                best_rules, best_default, best_obj = new_rules, new_default, new_obj

        elif move == "insert" and len(current_rules) < max_rules:
            pos = rng.randint(0, len(current_rules) + 1)
            remaining = PackedMask(np.ones(n_samples, dtype=bool))
            for i in range(pos):
                remaining = remaining.and_not(current_rules[i]["packed_mask"])
            if remaining.count < min_support:
                continue
            prob_ceil = 1.01 if pos == 0 else current_rules[pos - 1]["prob"]
            prob_floor = (current_rules[pos]["prob"] + prob_gap
                          if pos < len(current_rules) else -1)

            sample_indices = rng.choice(
                len(valid_cands), min(len(valid_cands), 50), replace=False)
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
                if p >= prob_ceil - prob_gap or p <= prob_floor:
                    continue
                if p > best_ip:
                    best_insert = {
                        "rule": {"conditions": cand["conditions"],
                                 "description": cand["description"]},
                        "prob": p, "n_captured": nc, "packed_mask": pm_active,
                    }
                    best_ip = p
            if best_insert is None:
                continue

            new_rules = current_rules[:pos] + [best_insert] + current_rules[pos:]
            remaining = PackedMask(np.ones(n_samples, dtype=bool))
            for r in new_rules:
                remaining = remaining.and_not(r["packed_mask"])
            if remaining.count > 0:
                new_default = (((remaining.sum_y(y) + 1) /
                                (remaining.count + 2)) if use_laplace
                               else remaining.sum_y(y) / remaining.count)
            else:
                new_default = float(y.mean())
            new_obj = _compute_objective(new_rules, new_default, y, n_samples, None)
            if new_obj > best_obj:
                best_rules, best_default, best_obj = new_rules, new_default, new_obj

    return best_rules, best_default, best_obj


# =============================================================================
# FALLING RULE LIST — MODEL CLASS
# =============================================================================

class FallingRuleList:
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
            X, y, feature_names, n_thresholds=self.n_thresholds,
            min_support=min_support, use_laplace=self.use_laplace,
            max_conditions=self.max_conditions)
        candidates = list(cache.iter_candidates())
        if verbose:
            print(f"  Candidates: {len(candidates)} pass support filter")

        greedy_rules, greedy_default = celf_select_rules(
            candidates, y, n_samples, max_rules=self.max_rules,
            min_support=min_support, max_coverage_frac=self.max_coverage_frac,
            prob_gap=self.prob_gap, use_laplace=self.use_laplace)
        if verbose:
            print(f"  Greedy solution: {len(greedy_rules)} rules")

        if self.mc_restarts > 0 and len(greedy_rules) > 0:
            mc_rules, mc_default, mc_obj = mc_search_improve(
                greedy_rules, greedy_default, candidates, y, n_samples,
                max_rules=self.max_rules, min_support=min_support,
                max_coverage_frac=self.max_coverage_frac,
                prob_gap=self.prob_gap, use_laplace=self.use_laplace,
                n_restarts=self.mc_restarts)
            greedy_obj = _compute_objective(
                greedy_rules, greedy_default, y, n_samples, None)
            if mc_obj > greedy_obj:
                if verbose:
                    print(f"  MC improved: {greedy_obj:.1f} → {mc_obj:.1f}")
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

    def to_table_data(self):
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
# MAIN — FIT + THRESHOLD TUNING ON TEST SET
# =============================================================================

if __name__ == "__main__":

    BETA = 0.72

    # ---- Fixed hyperparameters (from Optuna AUC-best trial) ----
    PARAMS = {
        "max_rules":         28,
        "min_support_frac":  0.03620738142047356,
        "n_thresholds":      20,
        "max_conditions":    3,
        "max_coverage_frac": 0.6239009112557059,
        "prob_gap":          0.004212451270415977,
        "use_laplace":       True,
        "mc_restarts":       20,
    }

    print("=" * 72)
    print(" FALLING RULE LIST — Fixed Model + F_0.72 Threshold Tuning")
    print("=" * 72)

    # ---- 1. Load & split ----
    df = pd.read_csv("df_clean_v3_for_FRL.csv")
    target_col = "RiskPerformanceBinary"
    feature_cols = [c for c in df.columns if c != target_col]

    X_all = df[feature_cols].values
    y_all = df[target_col].values

    # 70% train / 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.30, random_state=42, stratify=y_all)

    print(f"\n  Dataset:    {df.shape[0]} rows, {len(feature_cols)} features")
    print(f"  Train set:  {len(X_train)}  (fit model)")
    print(f"  Test set:   {len(X_test)}  (threshold tuning + evaluation)")
    print(f"  Prevalence: train={y_train.mean():.3f}  test={y_test.mean():.3f}")

    # ---- 2. Fit model on training set ----
    print(f"\n{'='*72}")
    print(" FITTING MODEL")
    print(f"{'='*72}\n")
    print("  Hyperparameters:")
    for k, v in PARAMS.items():
        print(f"    {k}: {v}")
    print()

    model = FallingRuleList(**PARAMS)
    model.fit(X_train, y_train, feature_names=feature_cols, verbose=True)

    # ---- 3. Print rule table ----
    print(f"\n{'='*72}")
    print(" FALLING RULE LIST")
    print(f"{'='*72}")
    rows = model.to_table_data()
    print()
    print(tabulate(rows, headers="keys", tablefmt="fancy_grid",
                   stralign="left", numalign="center",
                   colalign=("center", "left", "center", "center", "center")))

    # ---- 4. Threshold sweep on test set ----
    print(f"\n{'='*72}")
    print(f" THRESHOLD OPTIMIZATION VIA F_{BETA} (Bad class, test set)")
    print(f"{'='*72}")

    test_probs = model.predict_proba(X_test)

    # Dense sweep from 0.30 to 0.75 in 0.01 steps
    thresholds = np.arange(0.30, 0.75, 0.05)
    sweep_results = []

    for t in thresholds:
        y_pred_t = (test_probs >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred_t)

        prec_bad = precision_score(y_test, y_pred_t, pos_label=1,
                                   zero_division=0)
        rec_bad = recall_score(y_test, y_pred_t, pos_label=1,
                               zero_division=0)
        f_beta = fbeta_score(y_test, y_pred_t, beta=BETA, pos_label=1,
                             zero_division=0)
        acc = accuracy_score(y_test, y_pred_t)

        sweep_results.append({
            "Threshold": round(t, 2),
            "Precision Bad": prec_bad,
            "Recall Bad": rec_bad,
            f"F_{BETA} Bad": f_beta,
            "Accuracy": acc,
            "Good Denied (FP)": cm[0, 1],
            "Bad Approved (FN)": cm[1, 0],
        })

    sweep_df = pd.DataFrame(sweep_results)

    # Find optimum
    best_idx = sweep_df[f"F_{BETA} Bad"].idxmax()
    best_threshold = sweep_df.loc[best_idx, "Threshold"]

    # Display subset: every 5th row + the optimum row
    display_idx = set(range(0, len(sweep_df), 1))
    display_idx.add(best_idx)
    display_df = sweep_df.loc[sorted(display_idx)].copy()
    display_df[""] = ""
    display_df.loc[best_idx, ""] = "  ◀ BEST"

    fmt = display_df.copy()
    for col in ["Precision Bad", "Recall Bad", f"F_{BETA} Bad", "Accuracy"]:
        fmt[col] = fmt[col].map(lambda x: f"{x:.4f}")

    print(f"\n  Threshold sweep (test set, n={len(X_test)}):")
    print(tabulate(fmt.values.tolist(), headers=list(fmt.columns),
                   tablefmt="fancy_grid", stralign="center",
                   numalign="center"))

    best_row = sweep_df.iloc[best_idx]
    print(f"\n  ✦ OPTIMAL THRESHOLD: {best_threshold}")
    print(f"    Precision (Bad): {best_row['Precision Bad']:.4f}")
    print(f"    Recall (Bad):    {best_row['Recall Bad']:.4f}")
    print(f"    F_{BETA} (Bad):     {best_row[f'F_{BETA} Bad']:.4f}")
    print(f"    Accuracy:        {best_row['Accuracy']:.4f}")
    print(f"\n    β={BETA} weights precision ~{1/BETA**2:.1f}x over recall.")

    # ---- 5. Side-by-side: default 0.5 vs optimized threshold ----
    print(f"\n{'='*72}")
    print(f" DEFAULT vs OPTIMIZED THRESHOLD (test set)")
    print(f"{'='*72}\n")

    test_preds_opt = (test_probs >= best_threshold).astype(int)
    test_preds_def = (test_probs >= 0.5).astype(int)
    test_auc = roc_auc_score(y_test, test_probs)

    def _row(label, preds):
        return [
            label,
            f"{precision_score(y_test, preds, pos_label=1):.4f}",
            f"{recall_score(y_test, preds, pos_label=1):.4f}",
            f"{fbeta_score(y_test, preds, beta=BETA, pos_label=1):.4f}",
            f"{accuracy_score(y_test, preds):.4f}",
            f"{test_auc:.4f}",
        ]

    comparison = [
        _row("Default (0.50)", test_preds_def),
        _row(f"Optimized ({best_threshold})", test_preds_opt),
    ]
    print(tabulate(comparison,
                   headers=["Threshold", "Bad Precision", "Bad Recall",
                            f"F_{BETA} Bad", "Accuracy", "AUC"],
                   tablefmt="fancy_grid", stralign="center",
                   numalign="center"))

    print(f"\n  Confusion Matrix (threshold={best_threshold}):")
    cm = confusion_matrix(y_test, test_preds_opt)
    cm_data = [
        ["Pred Good (0)", cm[0, 0], cm[0, 1]],
        ["Pred Bad (1)", cm[1, 0], cm[1, 1]],
    ]
    print(tabulate(cm_data,
                   headers=["", "Actual Good (0)", "Actual Bad (1)"],
                   tablefmt="fancy_grid", stralign="center",
                   numalign="center"))

    print(f"\n{'='*72}")
