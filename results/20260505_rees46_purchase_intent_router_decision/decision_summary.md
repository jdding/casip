# REES46 Purchase-Intent Router Decision

Date: 2026-05-05

## Question

Can the `+852` Hit@10 oracle headroom in REES46 be harvested by a
validation-safe purchase-intent/source router using deployable return-state
features?

## Prototype

Script: `scripts/train_agknet/rees46_purchase_intent_router.py`

Training data:

- validation users from
  `results/20260505_rees46_purchase_event_gate_local/rees46_validation_safe_gate_val_per_user.csv`;
- candidate config labels are whether each source/config hits validation
  holdout `cart` at top 10;
- features are deployable return-state features from the protocol artifact:
  history/context lengths, view/cart mix, repeat rate, seen-product rate,
  category/brand/code overlap, entropy, and candidate-source identity.

Test data:

- full validation feedback as context;
- test purchases as targets;
- no test purchase/popularity labels are used by the router.

Baseline:

`global_purchase_popularity_0.025__context_event_weighted_replay_0.975`

Baseline test Hit@10/50/100/500:

`3044/5319/6392/8816`

## Results

| Prototype | Validation Hit@10 | Test Hit@10 | Test Hit@50 | Test Hit@100 | Test Hit@500 |
|---|---:|---:|---:|---:|---:|
| Sources HGB router | 3704 | 2964 | 5099 | 6085 | 8339 |
| Compact HGB router | 3693 | 2933 | 4785 | 5629 | 7549 |
| Compact logistic router | 3559 | 1925 | 2099 | 2109 | 2112 |
| All-event proxy compact HGB | 8630 | 2766 | 4194 | 4773 | 6186 |

Conservative margin sweep over the sources HGB router:

| Margin | Test Hit@10 | Test Hit@50 | Test Hit@100 | Test Hit@500 | Behavior |
|---:|---:|---:|---:|---:|---|
| 0.00 | 2964 | 5099 | 6085 | 8339 | switches too often |
| 0.02 | 3030 | 5280 | 6339 | 8734 | still below baseline |
| 0.05 | 3040 | 5309 | 6381 | 8802 | still below baseline |
| 0.10 | 3044 | 5318 | 6391 | 8814 | essentially baseline |
| 0.15+ | 3044 | 5319 | 6392 | 8816 | no switching |

## Decision

Stop this prototype line for now.

The oracle headroom is real, but it is not recoverable with the current
deployable return-state features and validation proxies. The failure mode is
not model capacity: HGB reaches high training AUC and still transfers worse
than the fixed rule. The failure mode is proxy mismatch and insufficient
state identifiability: validation cart/all-event targets reward context-driven
sources that do not transfer to test purchases.

## Implication

Do not invest next in a router-only method. For a WWW/KDD-level method, the
next investment must change the supervision or candidate construction, not
only the source-selection model:

1. construct a stronger purchase-aligned pseudo-target inside validation; or
2. add a new candidate generator that uses return-phase sequence semantics
   beyond the existing popularity/replay sources.

Until one of those passes a small gate, the fixed purchase-event rule remains
the strongest deployable REES46 result.
