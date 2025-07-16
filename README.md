# ML-Deploy-Test: MLOps Best Practices for an Intent Classifier

## 🚀 Project Overview

This project demonstrates a robust and scalable Machine Learning Operations (MLOps) pipeline for an Intent Classifier. It showcases how to build, deploy, and monitor an ML model as a service, emphasizing best practices for reproducibility, automation, and observability.

## 🛡️ MLOps Best Practices Demonstrated

This repository is a practical example of essential MLOps principles:

*   **Model Versioning & Management (Weights & Biases - W&B):**
    *   Models are tracked and versioned using Weights & Biases (W&B).
    *   The `fetch_model.sh` script automatically downloads the latest model artifact from W&B when the application container starts, ensuring the deployed model is always the intended version.
    *   *How it's accomplished:* W&B integration in `tools/intent_classifier.py` for training and artifact logging, and `app/fetch_model.sh` for retrieval.

*   **Containerization (Docker & Docker Compose):**
    *   The FastAPI API and the Streamlit frontend are containerized for consistent environments across development, testing, and deployment.
    *   Docker Compose orchestrates these services, along with the observability stack, for easy local setup and management.
    *   *How it's accomplished:* `app/app.Dockerfile` for the FastAPI service, `app/Streamlit.Dockerfile` for the Streamlit frontend, and `docker-compose.yml` for orchestration.

*   **API Development (FastAPI):**
    *   The ML model is exposed as a RESTful API using FastAPI, providing a modern, high-performance, and automatically documented interface.
    *   *How it's accomplished:* `app/app.py` defines the FastAPI application and its endpoints.

*   **Frontend (Streamlit):**
    *   A simple, interactive web interface built with Streamlit allows for easy interaction with the deployed ML model.
    *   *How it's accomplished:* `app/streamlit.py` implements the Streamlit application.

*   **Observability (OpenTelemetry, LGTM Stack):**
    *   Comprehensive monitoring is integrated using OpenTelemetry for collecting logs, traces, and metrics.
    *   These telemetry signals are sent to a bundled Grafana LGTM (Loki, Grafana, Tempo, Prometheus) stack for visualization and analysis.
    *   *How it's accomplished:* OpenTelemetry SDKs are initialized in `app/observability.py`, and the `docker-compose.yml` includes the `grafana/otel-lgtm` service. Logs are sent to Loki, traces to Tempo, and metrics to Prometheus, all visualized in Grafana.

*   **CI/CD (GitHub Actions):**
    *   Automated workflows ensure code quality, run tests, and manage Docker image publishing.
    *   *How it's accomplished:* Workflows defined in `.github/workflows/test.yml` (for testing) and `.github/workflows/docker-publish.yml` (for Docker Hub publishing).

*   **Database Integration (MongoDB Atlas):**
    *   Prediction requests and results are logged to MongoDB for historical analysis and auditing.
    *   *How it's accomplished:* `db/engine.py` and `db/models.py` for database interaction, and `app/app.py` for logging predictions.

## 📦 Getting Started: Local Development Setup

Follow these steps to get the Intent Classifier API and its observability stack running on your local machine.

### Prerequisites

*   **Docker Desktop:** Ensure Docker Desktop is installed and running on your system.
*   **Python 3.9+:** While the application runs in Docker, you'll need Python for initial setup and running tests.
*   **`pip`:** Python package installer.
*   **`git`:** Version control system.

### Step 1: Clone the Repository

Open your terminal and clone the project:

```bash
git clone https://github.com/adaj/ml-deploy-test.git
cd ml-deploy-test
```

### Step 2: Configure Environment Variables

Create a `.env` file in the root of your project directory (`ml-deploy-test/`) and add the following variables. **Replace the placeholder values with your actual credentials.**

```dotenv
# MongoDB Atlas Connection String (e.g., from your Atlas cluster)
# Example: MONGO_URI="mongodb+srv://<username>:<password>@cluster0.abcde.mongodb.net/myDatabase?retryWrites=true&w=majority"
MONGO_URI="YOUR_MONGODB_ATLAS_CONNECTION_STRING"
MONGO_DB="your_database_name" # e.g., "intent_logs"

# Weights & Biases API Key (for model fetching)
# You can find this in your W&B settings: https://wandb.ai/settings
WANDB_API_KEY="YOUR_WANDB_API_KEY"

# Weights & Biases Model URL (for fetching the trained model artifact)
# Example: WANDB_MODEL_URL="adaj/test_wandb/confusion-clf-v1"
# Ensure this points to the correct project and artifact name in your W&B account.
WANDB_MODEL_URL="YOUR_WANDB_MODEL_URL"

# Environment mode for the application (e.g., "dev" or "prod")
ENV="dev"
```

**Important Notes for MongoDB Atlas:**
*   **Network Access:** For your Docker containers to connect to MongoDB Atlas, you *must* configure Network Access in your Atlas project. For local development, you can temporarily add an IP Access List Entry for `0.0.0.0/0` (Allow Access from Anywhere). **Remember to remove this for production environments.**
*   **Database User:** Ensure the user specified in your `MONGO_URI` has the necessary read/write permissions to the specified database.

### Step 3: Build and Run the Docker Services

This command will build the Docker images (if they haven't been built or if changes are detected) and start all the services defined in `docker-compose.yml` in detached mode.

```bash
docker compose up --build -d
```

*   The `app` service (FastAPI) will be available on `http://localhost:8000`.
*   The `lgtm` service (Grafana, Loki, Tempo, Prometheus) will be available on `http://localhost:3000` (Grafana UI).

### Step 4: Verify Service Status

Check if all containers are running:

```bash
docker ps
```

You should see `intent-classifier-app-1` and `intent-classifier-lgtm-1` listed with a `Status` of `Up`.

### Step 5: Interact with the API

Send a sample prediction request to your FastAPI application. This will generate logs, traces, and metrics that will be sent to your observability stack.

```bash
curl -X POST http://localhost:8000/intents/confusion \
  -H "Content-Type: application/json" \
  -d '{"text": "I need help with my account"}'
```

You should receive a JSON response from the API.

### Step 6: Explore Observability in Grafana

Open your web browser and navigate to `http://localhost:3000`.

*   **Login:** Use default credentials: `admin` / `admin`.
*   **Logs (Loki):**
    *   Go to the **Explore** section (compass icon on the left sidebar).
    *   Select the **Loki** datasource.
    *   Query: `{service_name="intent-classifier-app"}`
    *   You should see all application logs, including `INFO` messages, from your FastAPI service.
*   **Traces (Tempo):**
    *   Go to the **Explore** section.
    *   Select the **Tempo** datasource.
    *   Search by service name: `service.name="intent-classifier-app"`.
    *   You should see traces for each API request, showing the flow and duration of operations.
*   **Metrics (Prometheus):
    *   Go to the **Explore** section.
    *   Select the **Prometheus** datasource.
    *   Query for `prediction_count` or `prediction_latency_seconds`.
    *   You can visualize the number of predictions and their latency over time.

## 📁 Project Structure

```
.
├── .github/                  # GitHub Actions workflows for CI/CD
│   └── workflows/
│       ├── docker-publish.yml  # Workflow to build and publish Docker images
│       └── test.yml            # Workflow for running tests
├── app/                      # FastAPI application and related files
│   ├── app.Dockerfile        # Dockerfile for the FastAPI service
│   ├── app.py                # Main FastAPI application entry point
│   ├── Streamlit.Dockerfile  # Dockerfile for the Streamlit frontend
│   ├── streamlit.py          # Streamlit application
│   ├── fetch_model.sh        # Script to fetch model from W&B
│   └── observability.py      # OpenTelemetry setup and instrumentation
├── db/                       # Database models and connection logic
├── tools/                    # ML model related scripts and artifacts
│   ├── models/               # Trained ML models
│   └── ...
├── tests/                    # Unit and integration tests
├── docker-compose.yml        # Docker Compose orchestration for all services
├── requirements.txt          # Pinned Python dependencies
└── .gitignore                # Git ignore rules
```

## 📚 Resources

*   **FastAPI Docs:** [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
*   **MongoDB Atlas:** [https://www.mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
*   **Weights & Biases:** [https://wandb.ai/](https://wandb.ai/)
*   **Grafana LGTM Stack:** [https://grafana.com/oss/lgtm/](https://grafana.com/oss/lgtm/)

---

📄 License
This project is licensed under the MIT License.