# Marketing Mix Modeling and Media Response

Publicis Groupe internship. The R workflow estimates a sales baseline with Prophet and models media carryover and saturation. A Bayesian attribution stage builds a Stan model for media and baseline effects.

## Explore the project

| File | What it contains |
| --- | --- |
| [Prophet functions](R/prophet-functions.R) | Seasonality flags, train/test splitting, evaluation, grid search and holiday handling. |
| [Media response functions](R/media-response.R) | Original geometric adstock, saturation and marginal-response calculations. |
| [Bayesian attribution](R/stan-attribution.R) | Original R code that constructs the Stan specification used in the pipeline. |

## Data and use

The Prophet function library is a full source file. The two other files contain verbatim sections of the internship pipeline, separated from client-specific inputs and reporting paths. The Stan excerpt expects upstream model settings and feature counts.
