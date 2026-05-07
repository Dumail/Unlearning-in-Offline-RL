import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .tmi import bootstrap_auc_ci, compute_tmi_auc


def threshold_mia(
    forget_nlls: np.ndarray,
    negative_nlls: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """Yeom-style threshold MIA.

    NLL < threshold -> predicted as member.
    Default threshold = median of the combined distribution.
    Returns balanced accuracy and the corresponding threshold.
    """
    if threshold is None:
        threshold = float(np.median(np.concatenate([forget_nlls, negative_nlls])))

    # Member prediction: NLL below threshold -> model is confident -> likely member
    forget_pred = (forget_nlls < threshold).astype(int)  # Should be 1 (member)
    negative_pred = (negative_nlls < threshold).astype(int)  # Should be 0 (non-member)

    y_true = np.concatenate([np.ones(len(forget_nlls)), np.zeros(len(negative_nlls))])
    y_pred = np.concatenate([forget_pred, negative_pred])

    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    return {
        "threshold_balanced_accuracy": bal_acc,
        "threshold_value": threshold,
        # Convert balanced accuracy to equivalent AUC scale: 0.5 = random
        "threshold_privacy_gap": abs(bal_acc - 0.5),
    }


def bootstrap_threshold_mia_ci(
    forget_nlls: np.ndarray,
    negative_nlls: np.ndarray,
    threshold: float | None = None,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for threshold MIA balanced accuracy。"""
    if threshold is None:
        threshold = float(np.median(np.concatenate([forget_nlls, negative_nlls])))

    rng = np.random.RandomState(seed)
    n_f, n_n = len(forget_nlls), len(negative_nlls)
    accs = []

    for _ in range(n_bootstrap):
        f_idx = rng.choice(n_f, n_f, replace=True)
        n_idx = rng.choice(n_n, n_n, replace=True)
        f_pred = (forget_nlls[f_idx] < threshold).astype(int)
        n_pred = (negative_nlls[n_idx] < threshold).astype(int)
        y_true = np.concatenate([np.ones(n_f), np.zeros(n_n)])
        y_pred = np.concatenate([f_pred, n_pred])
        accs.append(balanced_accuracy_score(y_true, y_pred))

    accs = np.array(accs)
    return (
        float(np.mean(accs)),
        float(np.percentile(accs, 100 * alpha / 2)),
        float(np.percentile(accs, 100 * (1 - alpha / 2))),
    )


def reference_model_attack(
    forget_nlls: np.ndarray,
    negative_nlls: np.ndarray,
    ref_forget_nlls: np.ndarray,
    ref_negative_nlls: np.ndarray,
) -> dict:
    """Reference-model calibrated attack.

    Score = NLL_reference - NLL_target (higher -> target model is more confident on
    that trajectory -> likely member). Uses the gold-standard retrained model as
    reference to remove trajectory intrinsic difficulty bias.
    """
    # Calibrated score: reference NLL minus target NLL
    # Member trajectories have low NLL on target (more confident), high NLL on
    # reference (not trained on them), so score = ref_NLL - target_NLL is higher for members
    forget_scores = ref_forget_nlls - forget_nlls
    negative_scores = ref_negative_nlls - negative_nlls

    labels = np.concatenate(
        [np.ones(len(forget_scores)), np.zeros(len(negative_scores))]
    )
    scores = np.concatenate([forget_scores, negative_scores])

    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = 0.5

    # Bootstrap CI
    auc_mean, ci_low, ci_high = _bootstrap_auc_with_scores(
        forget_scores, negative_scores
    )

    return {
        "reference_auc": auc,
        "reference_auc_ci_low": ci_low,
        "reference_auc_ci_high": ci_high,
        "reference_privacy_gap": abs(auc - 0.5),
    }


def _bootstrap_auc_with_scores(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap AUC-ROC CI from pre-computed scores。"""
    rng = np.random.RandomState(seed)
    n_m, n_n = len(member_scores), len(nonmember_scores)
    aucs = []

    for _ in range(n_bootstrap):
        m_idx = rng.choice(n_m, n_m, replace=True)
        n_idx = rng.choice(n_n, n_n, replace=True)
        labels = np.concatenate([np.ones(n_m), np.zeros(n_n)])
        scores = np.concatenate([member_scores[m_idx], nonmember_scores[n_idx]])
        try:
            aucs.append(roc_auc_score(labels, scores))
        except ValueError:
            continue

    aucs = np.array(aucs)
    return (
        float(np.mean(aucs)),
        float(np.percentile(aucs, 100 * alpha / 2)),
        float(np.percentile(aucs, 100 * (1 - alpha / 2))),
    )


def nll_variance_attack(
    forget_nlls: np.ndarray,
    negative_nlls: np.ndarray,
) -> dict:
    """Per-trajectory NLL variance attack (only effective with multi-token NLL).

    For data already aggregated to mean NLL, uses absolute deviation as proxy:
    |NLL - population_mean| smaller -> model is more calibrated on that trajectory -> likely member.
    """
    # Use absolute deviation from population mean as non-membership signal
    # Member trajectories are closer to the mean (model calibrated), non-members are farther
    all_nlls = np.concatenate([forget_nlls, negative_nlls])
    pop_mean = all_nlls.mean()

    # Score: -|NLL - mean| (higher = closer to mean = likely member)
    forget_scores = -np.abs(forget_nlls - pop_mean)
    negative_scores = -np.abs(negative_nlls - pop_mean)

    labels = np.concatenate(
        [np.ones(len(forget_scores)), np.zeros(len(negative_scores))]
    )
    scores = np.concatenate([forget_scores, negative_scores])

    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = 0.5

    auc_mean, ci_low, ci_high = _bootstrap_auc_with_scores(
        forget_scores, negative_scores
    )

    return {
        "variance_auc": auc,
        "variance_auc_ci_low": ci_low,
        "variance_auc_ci_high": ci_high,
        "variance_privacy_gap": abs(auc - 0.5),
    }


def z_normalized_nll_attack(
    forget_nlls: np.ndarray,
    negative_nlls: np.ndarray,
) -> dict:
    """Z-normalized NLL attack, removes NLL scale differences.

    Z-normalizes NLL before computing AUC-ROC, to check whether
    cross-architecture comparisons are biased by different NLL scales.
    """
    all_nlls = np.concatenate([forget_nlls, negative_nlls])
    mu, sigma = all_nlls.mean(), all_nlls.std()
    if sigma < 1e-8:
        return {
            "znorm_auc": 0.5,
            "znorm_auc_ci_low": 0.5,
            "znorm_auc_ci_high": 0.5,
            "znorm_privacy_gap": 0.0,
        }

    forget_z = (forget_nlls - mu) / sigma
    negative_z = (negative_nlls - mu) / sigma

    auc = compute_tmi_auc(forget_z, negative_z)
    auc_mean, ci_low, ci_high = bootstrap_auc_ci(forget_z, negative_z)

    return {
        "znorm_auc": auc,
        "znorm_auc_ci_low": ci_low,
        "znorm_auc_ci_high": ci_high,
        "znorm_privacy_gap": abs(auc - 0.5),
    }


def equivalence_test(
    auc: float,
    n_pairs: int,
    margin: float = 0.1,
    alpha: float = 0.05,
) -> dict:
    """TOST (Two One-Sided Tests) equivalence test.

    H0: |AUC - 0.5| >= margin
    H1: |AUC - 0.5| < margin (AUC in [0.5-margin, 0.5+margin])

    Uses asymptotic normal approximation for Wilcoxon AUC:
    SE(AUC) ~ sqrt((AUC*(1-AUC) + (n-1)*(Q1-AUC^2) + (n-1)*(Q2-AUC^2)) / (n*n))
    Simplified to SE ~ sqrt(AUC*(1-AUC)/n) for matched pairs.
    """
    se = np.sqrt(auc * (1 - auc) / max(n_pairs, 1))

    # Upper test: H0: AUC >= 0.5 + margin
    z_upper = (auc - (0.5 + margin)) / max(se, 1e-8)
    from scipy import stats

    p_upper = stats.norm.cdf(z_upper)  # Should be significant (small)

    # Lower test: H0: AUC <= 0.5 - margin
    z_lower = (auc - (0.5 - margin)) / max(se, 1e-8)
    p_lower = 1 - stats.norm.cdf(z_lower)  # Should be significant (small)

    p_tost = max(p_upper, p_lower)
    equivalent = p_tost < alpha

    return {
        "tost_p_value": float(p_tost),
        "tost_equivalent": bool(equivalent),
        "tost_margin": margin,
        "tost_alpha": alpha,
        "tost_se": float(se),
        "auc_in_margin": bool(abs(auc - 0.5) < margin),
    }


def multi_attack_evaluation(
    forget_nlls: np.ndarray,
    negative_nlls: np.ndarray,
    ref_forget_nlls: np.ndarray | None = None,
    ref_negative_nlls: np.ndarray | None = None,
    margin: float = 0.1,
) -> dict:
    """Run all attacks and aggregate results.

    Parameters:
        forget_nlls: target model per-trajectory mean NLL on forget set
        negative_nlls: target model per-trajectory mean NLL on non-member set
        ref_forget_nlls: reference model (gold-standard) NLL on forget set
        ref_negative_nlls: reference model NLL on non-member set
        margin: TOST equivalence test margin

    Returns:
        dict containing all attack results and robustness summary
    """
    forget_nlls = np.asarray(forget_nlls, dtype=np.float64)
    negative_nlls = np.asarray(negative_nlls, dtype=np.float64)
    n_pairs = min(len(forget_nlls), len(negative_nlls))

    # 1. Original NLL-AUC attack
    nll_auc = compute_tmi_auc(forget_nlls, negative_nlls)
    nll_mean, nll_ci_low, nll_ci_high = bootstrap_auc_ci(forget_nlls, negative_nlls)

    # 2. Threshold MIA
    thresh_result = threshold_mia(forget_nlls, negative_nlls)
    thresh_mean, thresh_ci_low, thresh_ci_high = bootstrap_threshold_mia_ci(
        forget_nlls, negative_nlls
    )

    # 3. Z-normalized NLL attack
    znorm_result = z_normalized_nll_attack(forget_nlls, negative_nlls)

    # 4. Reference-model attack (if reference NLLs are provided)
    ref_result = None
    if ref_forget_nlls is not None and ref_negative_nlls is not None:
        ref_forget_nlls = np.asarray(ref_forget_nlls, dtype=np.float64)
        ref_negative_nlls = np.asarray(ref_negative_nlls, dtype=np.float64)
        ref_result = reference_model_attack(
            forget_nlls, negative_nlls, ref_forget_nlls, ref_negative_nlls
        )

    # 5. NLL variance attack
    var_result = nll_variance_attack(forget_nlls, negative_nlls)

    # 6. TOST equivalence test on primary NLL-AUC
    equiv = equivalence_test(nll_auc, n_pairs, margin=margin)

    # Aggregate all gaps
    gaps = {
        "nll_auc": abs(nll_auc - 0.5),
        "threshold": thresh_result["threshold_privacy_gap"],
        "znorm": znorm_result["znorm_privacy_gap"],
        "variance": var_result["variance_privacy_gap"],
    }
    if ref_result is not None:
        gaps["reference"] = ref_result["reference_privacy_gap"]

    # Robustness check: whether all attack gaps are < margin
    all_below_margin = all(g < margin for g in gaps.values())
    # Whether all attack directions are consistent (all indicate member or all near random)
    max_gap = max(gaps.values())
    min_gap = min(gaps.values())

    result = {
        # Original NLL-AUC
        "nll_auc": nll_auc,
        "nll_auc_ci": [nll_ci_low, nll_ci_high],
        "nll_gap": abs(nll_auc - 0.5),
        # Threshold MIA
        "threshold_balanced_acc": thresh_result["threshold_balanced_accuracy"],
        "threshold_balanced_acc_ci": [thresh_ci_low, thresh_ci_high],
        "threshold_gap": thresh_result["threshold_privacy_gap"],
        "threshold_value": thresh_result["threshold_value"],
        # Z-normalized
        "znorm_auc": znorm_result["znorm_auc"],
        "znorm_auc_ci": [
            znorm_result["znorm_auc_ci_low"],
            znorm_result["znorm_auc_ci_high"],
        ],
        "znorm_gap": znorm_result["znorm_privacy_gap"],
        # NLL variance
        "variance_auc": var_result["variance_auc"],
        "variance_auc_ci": [
            var_result["variance_auc_ci_low"],
            var_result["variance_auc_ci_high"],
        ],
        "variance_gap": var_result["variance_privacy_gap"],
        # TOST equivalence
        "tost_p_value": equiv["tost_p_value"],
        "tost_equivalent": equiv["tost_equivalent"],
        "tost_margin": margin,
        # Robustness summary
        "n_pairs": n_pairs,
        "all_gaps_below_margin": all_below_margin,
        "max_gap": max_gap,
        "min_gap": min_gap,
        "gap_range": max_gap - min_gap,
        "n_attacks": len(gaps),
        "attack_names": list(gaps.keys()),
    }

    # If reference attack is available, supplement results
    if ref_result is not None:
        result["reference_auc"] = ref_result["reference_auc"]
        result["reference_auc_ci"] = [
            ref_result["reference_auc_ci_low"],
            ref_result["reference_auc_ci_high"],
        ]
        result["reference_gap"] = ref_result["reference_privacy_gap"]

    return result
