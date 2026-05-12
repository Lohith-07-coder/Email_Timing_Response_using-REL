# Email Timing Response

An MLOps project that uses reinforcement learning to recommend the best action for an email based on urgency, sender importance, waiting time, workload, and time of day.

The project trains a Deep Q-Network (DQN) agent to choose one of four actions:

- `reply_now`
- `delay_reply`
- `mark_important`
- `archive`

It includes model training, inference APIs, a Flask web UI, monitoring hooks, Docker deployment files, and CI/CD automation.

## Screenshots

### Web UI Dashboard

![Email triage dashboard](assets/ui-dashboard.png)

### Decision Result

![Email triage decision result](assets/ui-decision-result.png)

### Training Reward Curves

![Reward curve](assets/reward_curve.png)

![DQN reward curve](assets/reward_curve_dqn.png)

## Project Overview

The system models email handling as a reinforcement-learning problem.

| Item | Description |
| --- | --- |
| State | 5 features: priority, sender importance, waiting time, workload, time of day |
| Actions | 4 actions: reply now, delay reply, mark important, archive |
| Agent | Double DQN with replay buffer and target network |
| Reward | Custom reward function that favors timely responses to important emails |
| Serving | FastAPI inference API and Flask web UI |
| Monitoring | Metrics, logs, drift-detection utilities, and MLflow support |

## Architecture

```text
Data sources
    |
    v
Email simulation / feature extraction
    |
    v
RL environment + reward function
    |
    v
DQN training pipeline
    |
    v
Saved model weights
    |
    +--> FastAPI inference API
    |
    +--> Flask web UI
```

Main components:

| Path | Purpose |
| --- | --- |
| `agent/dqn.py` | DQN agent, Q-network, replay buffer, action selection |
| `environment/email_env.py` | Reinforcement-learning environment |
| `environment/reward.py` | Reward calculation logic |
| `simulation/` | Synthetic, Enron, terminal, NLP, and web email sources |
| `training/` | Trainer and evaluator |
| `pipelines/` | Training and inference pipeline entry points |
| `app/main.py` | FastAPI inference service |
| `ui/web_ui.py` | Flask UI server |
| `ui/templates/index.html` | Browser interface |
| `monitoring/` | Logging, metrics, drift detection, MLflow tracking |
| `.github/workflows/ci_cd.yml` | CI/CD workflow |

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Train The Model

Run the training pipeline:

```powershell
python pipelines\training_pipeline.py --episodes 10000
```

For a quicker local DQN run:

```powershell
python pipelines\train_local_dqn.py
```

Trained models are saved in:

```text
models/
```

The default DQN weights path is:

```text
models/dqn_weights.pt
```

## Run The FastAPI Inference API

Start the API:

```powershell
uvicorn app.main:app --reload --port 8000
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

Prediction example:

```powershell
Invoke-RestMethod -Method POST `
  -Uri "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" `
  -Body '{
    "subject": "Urgent meeting",
    "sender": "boss@company.com",
    "priority": 3,
    "sender_importance": 3,
    "waiting_time": 5,
    "workload": 2,
    "time_of_day": 14
  }'
```

## Run The Web UI

The repository also includes a Flask browser UI for trying the agent interactively.

Start the UI:

```powershell
python ui\web_ui.py
```

Then open:

```text
http://127.0.0.1:5000
```

The UI has two modes:

| Mode | Description |
| --- | --- |
| Auto mode | Enter subject and sender. NLP-style feature extraction estimates priority, sender importance, workload, and waiting time. |
| Manual mode | Use sliders to manually provide priority, sender importance, waiting time, workload, and time of day. |

Useful UI endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Render the web UI |
| `GET /agent_status` | Show agent mode and epsilon |
| `GET /debug` | Show session stats, sender memory, and model details |
| `POST /infer` | Live feature preview while typing |
| `POST /decide_nlp` | Auto-mode action recommendation |
| `POST /decide` | Manual-mode action recommendation |

## Run Tests

```powershell
pytest
```

Run with coverage:

```powershell
pytest --cov
```

## Docker

Build the image:

```powershell
docker build -f docker\Dockerfile -t email-timing-response .
```

Run with Docker Compose:

```powershell
docker compose up --build
```

## CI/CD

The GitHub Actions workflow in `.github/workflows/ci_cd.yml` is intended to automate:

- dependency installation
- code quality checks
- tests
- Docker build
- deployment steps

## Typical Workflow

1. Install dependencies.
2. Train the model or use the existing weights in `models/`.
3. Run the FastAPI service for programmatic inference.
4. Run the Flask UI for an interactive demo.
5. Use tests and CI/CD to validate changes.

## Notes

- If the API returns `Model not loaded`, check that `models/dqn_weights.pt` exists.
- If the UI starts slowly, it is usually loading the trained agent.
- The Flask UI performs online learning during interaction, so session metrics can change as you submit more examples.
