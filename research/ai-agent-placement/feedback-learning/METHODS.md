# Feedback-aware learning of AI–human player dynamics

## Research question and contribution scope

Can a population-conditioned, transition-constrained neural model forecast human engagement after an AI-placement intervention more reliably than a population-blind kernel or unconstrained neural baselines?

The contribution of this prototype is a CSWIM-linked computational benchmark, a structured neural transition model, controlled ablations, and an evaluation that distinguishes forecasting from placement decisions. It does not establish a novel general-purpose learning algorithm, recover hidden parameters, validate real human behavior, or claim conference-level novelty. The original problem remains AI-player placement and human engagement evolution.

## Preserved mechanism

Let S_t be the human population proportions at levels 0,...,5, with level 0 inactive. For a fixed AI placement ell, y is the corresponding one-hot vector. The opponent distribution among active levels is

    w_j(S_t,ell) = (S_t,j + r*y_j) / (sum_{k=1}^5 S_t,k + r*1{ell>0}).

We fix r=0.3, epsilon=0.02 and p0=0, matching `phase_diagram_compare.m` and the audited data-generating code rather than the paper's discrepant epsilon caption. For i=1,...,4,

    T_i,i-1 = sum_j w_j sigmoid(alpha*(|j-i|-dc)),
    T_i,i+1 = 1-T_i,i-1.

The original boundaries are T_0,0=0.98, T_0,1=0.02, T_5,4=1. AI=0 contributes no opponents. When the active pool is empty the simulator uses the original uniform-weight fallback. These rules are not intended as a universal model of real games; they are the preserved CSWIM assumptions.

## Finite population and observation model

Each period updates N=1000 humans synchronously. Conditional on the current counts and known context, transition counts in row i are multinomial with n_i trials and probabilities T_i,:. Player opponent draws and transitions are conditionally independent; this is not disjoint two-person matching. A batched binomial implementation is exactly equivalent to multinomial sampling because every row has at most two possible destinations.

Observed training data contain population histories and origin-destination transition counts. They are fully synthetic. The exact sigmoid and matching formula are available only to the generator/reference evaluator, not called by the learners. Alpha and dc are known contextual covariates; parameter inference is outside scope.

## Models and matched comparisons

All networks have two width-64 SiLU hidden layers. Inputs are alpha, dc/5, ell/5, 1{ell>0}, plus the six current population shares when allowed. There are no trainable embeddings, transformers, language models or RL agents.

| Name | Kernel / output | Inputs | Parameters |
|---|---|---|---:|
| No population feedback | Four down-probability logits; original legal destinations and boundaries | Context only | 4,740 |
| Direct population MLP | Softmax next population | Context + population | 5,254 |
| Dense transition MLP | Softmax over six destinations for each row 1–4; same fixed boundary rows | Context + population | 6,424 |
| Structured one-step | Four down-probability logits; legal destinations and fixed boundaries | Context + population | 5,124 |
| Structured multi-step | Same architecture, initialized from the corresponding one-step checkpoint | Context + population | 5,124 |

Structured models output q_i=sigmoid(g_theta(features)_i), set T_i,i-1=q_i and T_i,i+1=1-q_i, and evolve S_{t+1}=S_t*T_theta. Non-negativity, row sums and legal support hold by construction. The no-feedback model still propagates current population mass, but its transition kernel cannot adapt to the current opponent population. It is a strong mechanistic baseline, not a constant predictor.

The no-feedback vs structured comparison tests population feedback. Dense vs structured holds transition-count supervision and boundary knowledge fixed while changing legal support and output size. Direct vs structured changes both architecture and granularity of supervision: the direct model sees next-population targets, so its gap cannot be attributed solely to architecture. Model counts are comparable, not identical.

## Objectives and training

Transition models minimize the observed-count negative log likelihood

    L_count = -(1/B) sum_b sum_i,j (C_b,i,j/N)*log T_theta,i,j.

The direct population baseline minimizes cross entropy against next-period empirical proportions. Structured multi-step fine-tuning adds 10 times population MSE over eight recursively predicted periods to the first-step count likelihood. Windows come from the same training trajectories; the generator is not used to supply noiseless training targets.

AdamW: learning rate 0.002, weight decay 1e-5, batch size 1024, gradient clipping at norm 5, maximum 100 epochs. Every baseline selects its checkpoint using validation mean TV over a ten-step rollout, with patience 18. Fine-tuning uses learning rate 0.0005, 12 batches of 128 eight-step windows per epoch, maximum 45 epochs; it can retain the original checkpoint if validation does not improve. Three initialization/training seeds are 11, 29, 47. Hyperparameters were not selected on the test set; the baselines are limited-budget baselines, not exhaustively optimized architectures.

## Splits and shifts

- Train: 128 distinct (alpha,dc) contexts, six fixed AI placements per context, 32 periods; 768 trajectories and 24,576 population-transition observations.
- Validation: 32 new contexts, six placements, 32 periods.
- Test: 48 new contexts, all six placements, 60 periods, in each of three conditions.
- Context sampling: sign(alpha) equiprobable, |alpha| uniform on [0.2,1], dc uniform on [0,5]. The poorly identified/placement-indifferent alpha=0 neighborhood is not studied.
- Ordinary initialization: Dirichlet(2,...,2), converted to exactly N humans by largest-remainder rounding.
- Initial-population shift: Dirichlet(0.2,...,0.2), producing more concentrated states.
- Intervention shift: after 20 transitions, ell changes to (ell+3) mod 6. All individual action values were seen in training, but the changing action schedules were not.

All actions and time points from a parameter context remain in a single split. Test conditions reuse the same 48 parameter contexts, so they are related experiments, not 144 independent contexts. The three training seeds share a single training dataset; they assess optimization variability, not variability across independently generated training datasets. No OOD behavioral-parameter or unseen-AI-level guarantee is claimed.

## Forecasting and decision evaluation

TV(S,S_hat)=0.5*sum_i |S_i-S_hat_i|. We report teacher-forced one-step TV, recursive mean TV over 60 periods, final TV and engagement error. Stochastic-trajectory TV includes irreducible finite-population fluctuations. A separate deterministic-rollout TV metric is saved; a nonlinear mean-field rollout is not automatically the exact expectation of a finite-population process.

For static-placement tests, predict each configuration's finite-horizon objective

    J_H(ell) = E[(1/H) sum_{t=1}^H sum_i i*S_t,i], H=60,

and choose the largest predicted value. Independent Monte Carlo evaluation uses 1024 complete trajectories per context/action, with seeds distinct from training and forecast tests. We report regret relative to the largest of these estimated action means, not an analytically proven optimum. Same evaluation samples are reused across learned methods to make comparisons paired. The action means themselves are estimated and taking their maximum introduces possible selection optimism near ties; exact-best accuracy and tiny regrets must therefore be treated cautiously. Reference mean standard errors are recorded in `manifest.json`.

The within-0.02 rate means estimated regret is at most 0.02 level units. It is not a certified probability bound. Dynamic-switch tests evaluate forecasts only; they do not claim to optimize a dynamic policy.

Paired percentile bootstrap uses 4000 resamples of the 48 independent parameter contexts, keeping six actions together and averaging the three training seeds first. These intervals are conditional on the fixed datasets/reference estimates. They do not include training-data-generation uncertainty or fully propagate Monte Carlo reference uncertainty. They are exploratory intervals without multiple-comparison correction.

## Two useful elementary bounds

These are standard error-propagation observations, not claimed novel theorems.

If the true deterministic update F is L-Lipschitz in L1 on the states visited, and the learned update has uniform one-step L1 error at most eta there, then starting from the same state,

    e_{t+1} <= L*e_t + eta,
    e_t <= eta*sum_{k=0}^{t-1} L^k.

This follows by adding and subtracting F(S_hat_t). It explains why low teacher-forced error alone does not ensure good rollouts. No contraction constant is established for this model, and no claim of equilibrium uniqueness or stability is made.

If all six finite-horizon action values satisfy |J_hat(ell)-J(ell)|<=delta, choosing the maximizer of J_hat has regret at most 2*delta: add and subtract the predicted values at the true and predicted maximizers. This is a conditional bound, not an empirical certification of the networks. Population forecast error and policy regret can differ because correct ranking does not require perfect trajectories.

## Verification and reproducibility

The batched transition implementation is compared to an independently written scalar transcription of `solve.m`, plus one-step aggregate Monte Carlo and explicit per-player opponent sampling. Tests check probability normalization, population conservation, flow row totals and the empty-pool edge case. `verification.json` records measured errors. `audit_results.py` validates checkpoint predictions, output consistency, split separation and preservation of the original solver hash.

Exact CPU timings, package versions, seeds, source hashes and dataset hashes are included. The original `solve.m`, Excel files and paper are untouched. The saved result means and intervals can be recomputed from model forecasts and independent Monte Carlo samples.

## Positioning and references

The motivating connection is policy-induced distribution change, as in [Perdomo et al., Performative Prediction, ICML 2020](https://proceedings.mlr.press/v119/perdomo20a.html). Our finite-horizon forecasting experiment is not a verification of that paper's formal convergence assumptions. Learning population dynamics has precedents such as [Hashimoto et al., Learning Population-Level Diffusions with Generative RNNs, ICML 2016](https://proceedings.mlr.press/v48/hashimoto16.html); that work addresses a different observation regime and application.

The scientific starting point is the user's `CSWIM_Yanghaifei.pdf` and its accompanying MATLAB code. Appropriate claims are an implemented, reproducible synthetic-data extension and observed benchmark results. Establishing publication-level novelty would require a broader literature review, stronger baselines, independently regenerated data and additional environments or real observations.
