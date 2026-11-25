# Machine Learning–Driven Risk Stratification for Non-Invasive Prenatal Testing

This mathematical modeling competition project studies the timing of NIPT, grouping by maternal characteristics, and the classification of chromosomal abnormalities. The original analysis combines repeated-measure modeling, optimization and supervised classification.

## Explore the project

| File | What it contains |
| --- | --- |
| [Competition paper (Chinese)](report/competition-paper.docx) | The four research questions, methods and findings. |
| [Repeated measurements](analysis/q1-mixed-effects.R) | A mixed-effects model for gestational week, BMI and Y-chromosome concentration. |
| [Timing by group](analysis/q2-testing-schedule.ipynb) | Optimizes group cutoffs and recommended sampling times. |
| [Multifactor optimization](analysis/q3-multifactor-schedule.ipynb) | Genetic search with feasibility and risk constraints. |
| [Classification](analysis/q4-classification.py) | Model selection and evaluation for the final screening question. |

## Data and use

Participant-level source workbooks are excluded. The scripts retain their original local input paths. Stored notebook outputs containing individual records have been cleared in the portfolio copies.
