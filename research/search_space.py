"""Focused one-variable-at-a-time hypotheses for night-one research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    family: str
    overrides: dict[str, Any]
    hypothesis: str
    expected_mechanism: str


def jevlet_p_candidates() -> list[Candidate]:
    """Pretrained-backbone ablations. Each changes one axis of ``configs/pretrained_*.json``.

    Grounded in TypeSafe's public description of Jev: questions are evaluated independently
    against one shared state, options are runtime-defined with descriptions, and training
    targets calibrated probabilities (RLCD) rather than rater preference.
    """
    return [
        Candidate(
            "p-baseline",
            "baseline",
            {},
            "A pretrained encoder under the packed block-bidirectional topology is a strong base.",
            "Pretrained semantics plus listwise option comparison inside each isolated branch.",
        ),
        Candidate(
            "p-option-isolated",
            "attention_topology",
            {"model.attention_topology": "option_isolated"},
            "Exact option-order invariance costs little accuracy and removes position bias.",
            "Options share positions and cannot see each other; only [DECIDE] compares them.",
        ),
        Candidate(
            "p-full-leaky",
            "attention_topology",
            {"model.attention_topology": "full"},
            "Sibling isolation does not reduce accuracy relative to full cross-question attention.",
            "If leakage helps, siblings carry shortcuts that isolated Jev-style calls lack.",
        ),
        Candidate(
            "p-option-end",
            "option_representation",
            {"model.option_pool": "end"},
            "The [END_OPTION] boundary state is as good as mean-pooled option tokens.",
            "A learned boundary summary can replace pooling that the encoder was pretrained for.",
        ),
        Candidate(
            "p-head-bilinear",
            "decision_head",
            {"model.decision_head": "bilinear"},
            "A bilinear pointer aligns decision and option subspaces better than a dot product.",
            "An extra learned matrix lets the head reweight semantic directions per decision.",
        ),
        Candidate(
            "p-head-mlp",
            "decision_head",
            {"model.decision_head": "mlp"},
            "A nonlinear compatibility head helps on entailment-style questions.",
            "|decide - option| features expose mismatch directly to the scorer.",
        ),
        Candidate(
            "p-loss-ce-brier",
            "loss",
            {"training.loss": "ce_brier"},
            "Adding Brier to log-loss improves calibration at equal accuracy (an RLCD proxy).",
            "Both are strictly proper scoring rules; Brier bounds the penalty on confident misses.",
        ),
        Candidate(
            "p-loss-smoothing",
            "loss",
            {"training.loss": "label_smooth_ce"},
            "Light label smoothing reduces overconfidence on public-data shift.",
            "Softened targets cap logit growth on easy templated families.",
        ),
        Candidate(
            "p-backbone-bge-small",
            "backbone",
            {"model.backbone": "BAAI/bge-small-en-v1.5"},
            "A deeper retrieval-tuned 33M encoder improves semantic routing per millisecond.",
            "Twelve layers and contrastive pretraining give sharper query-option alignment.",
        ),
    ]


def night_one_candidates() -> list[Candidate]:
    return [
        Candidate(
            "baseline",
            "baseline",
            {},
            "The current shared-state block-causal pointer model is a valid baseline.",
            (
                "It encodes state once, isolates sibling questions, and scores dynamic "
                "options directly."
            ),
        ),
        Candidate(
            "state-mean",
            "state_representation",
            {"model.state_pool": "mean"},
            "Mean state pooling improves robustness over the final state token.",
            "Evidence distributed across the prefix contributes directly to each decision.",
        ),
        Candidate(
            "state-attention",
            "state_representation",
            {"model.state_pool": "attention"},
            "Learned state attention improves selective evidence aggregation.",
            "A learned query emphasizes decision-relevant prefix positions.",
        ),
        Candidate(
            "state-latent",
            "state_representation",
            {"model.state_pool": "latent"},
            "A tiny latent pool preserves evidence with a compact decision interface.",
            "Multiple learned queries compress complementary state features before averaging.",
        ),
        Candidate(
            "option-last",
            "option_representation",
            {"model.option_pool": "last"},
            "The final lexical option token is sufficient for dynamic option scoring.",
            "A cheaper representation may retain the relevant option semantics.",
        ),
        Candidate(
            "option-mean",
            "option_representation",
            {"model.option_pool": "mean"},
            "Mean option pooling is less sensitive to option length.",
            "All lexical bytes contribute equally instead of relying on a boundary state.",
        ),
        Candidate(
            "head-bilinear",
            "decision_head",
            {"model.decision_head": "bilinear"},
            "A bilinear pointer captures useful query-option interactions.",
            "The learned compatibility matrix can align distinct decision and option subspaces.",
        ),
        Candidate(
            "head-mlp",
            "decision_head",
            {"model.decision_head": "mlp"},
            "A nonlinear compatibility head improves complex option comparisons.",
            "Absolute differences plus a hidden layer model interactions beyond a dot product.",
        ),
        Candidate(
            "head-linear",
            "decision_head",
            {"model.decision_head": "linear"},
            "A minimal dynamic linear scorer is competitive at lower complexity.",
            "Direct features may be enough when the transformer performs the comparison.",
        ),
        Candidate(
            "loss-brier",
            "loss",
            {"training.loss": "brier"},
            "Brier-only training improves calibration without unacceptable accuracy loss.",
            "A proper quadratic scoring rule directly penalizes probability error.",
        ),
        Candidate(
            "loss-ce-brier",
            "loss",
            {"training.loss": "ce_brier"},
            "CE plus Brier preserves ranking while improving calibration.",
            "Log-loss supplies sharp gradients while Brier regularizes probability geometry.",
        ),
        Candidate(
            "loss-smoothing",
            "loss",
            {"training.loss": "label_smooth_ce"},
            "Light label smoothing reduces overconfidence under shift.",
            "Softened one-hot targets discourage extreme logits on deterministic tasks.",
        ),
        Candidate(
            "topology-separate",
            "attention_topology",
            {"model.attention_topology": "separate"},
            "Duplicating state per question establishes the quality/compute baseline.",
            "Each question gets a conventional causal sequence without packed-branch masking.",
        ),
        Candidate(
            "topology-shared-leaky",
            "attention_topology",
            {"model.attention_topology": "shared_prefix"},
            "Naive packed causal questions may benefit from cross-question context.",
            (
                "Later decisions can reuse earlier question and option representations, "
                "at leakage risk."
            ),
        ),
    ]


CANDIDATE_SETS = {"night_one": night_one_candidates, "jevlet_p": jevlet_p_candidates}
