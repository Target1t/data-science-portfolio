# ChatBI for Business Intelligence

Dr. Jing internship. This Qwen-Agent integration turns an analyst question into an RSL-SQL workflow. One tool orchestrates schema linking and query generation. A second tool executes the resulting query and exports its result.

## Explore the project

| File | What it contains |
| --- | --- |
| [Agent integration excerpts](src/chatbi-tools.py) | Original schema parsing, tool registration, SQL generation and execution classes, ReAct agent setup and UI initialization. |

## Data and use

These are verbatim sections of chatbi.py, including their original decorators. Running them requires the external RSL-SQL project, a configured model and an authorized warehouse connection. The credential-bearing connection block, live table queries, internal configuration and customer exports are excluded.
