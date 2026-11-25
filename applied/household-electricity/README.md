# Household Electricity Consumption Forecasting

UC Davis ECS 171, Group 13. This team project forecasts household power consumption from multivariate meter readings. The repository contains convolutional and recurrent forecasting experiments, cross-validation summaries and a web interface for inspecting a prediction.

## Explore the project

| File | What it contains |
| --- | --- |
| [CNN architecture](CNN/cnn_model_yin.py) | Convolutional forecasting model used by the model loader. |
| [Cross-validation notebook](CNN/cv_demo.ipynb) | Training and evaluation workflow for the CNN. |
| [LSTM experiment](RNN/lstm_v4.ipynb) | Recurrent model for electricity-use sequences. |
| [Model comparison](CNN/performance_summary.csv) | Saved comparison of forecasting performance. |
| [FastAPI application](app.py) | Serves a test window, its predictions and generated recommendations. |
| [Web interface](front/app/page.js) | Client for the forecasting application. |

## Data and use

The original project uses the [UCI household power consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption). The checkpoint referenced by the API is included. The application expects a GPT_KEY in the local environment for its recommendations. The data processor used with the CNN is also available at the import path expected by app.py.
