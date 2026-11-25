# AI Agent Placement and Dynamic Player Engagement in Online Games

How does placing an AI opponent at a particular rank change human participation over repeated matches? The paper studies a ranked game. A MATLAB parameter sweep and a later Python extension examine the resulting player distributions.

## Explore the project

| File | What it contains |
| --- | --- |
| [Research paper](paper/ai-placement.pdf) | Model, placement regimes and implications for engagement. |
| [Parameter sweep](matlab/phase_diagram_compare.m) | Compares placement policies across response parameters using the equilibrium solver. |
| [Equilibrium solver](matlab/solve.m) | Computes the player distribution for alternative agent placements. |
| [Learning extension](feedback-learning/run_experiment.py) | Experiments on finite-population synthetic trajectories. |
| [Extension results](feedback-learning/results/summary.csv) | One-step error, rollout error and decision regret by model and split. |

## Data and use

The MATLAB scripts and their original result workbooks share a directory so filename-based references are preserved. The Python extension uses simulated player trajectories.
