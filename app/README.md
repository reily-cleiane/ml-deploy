# Intent Classifier API

This API provides an interface to classify user intents based on input text. It uses a machine learning model to predict the intent and provides a certainty score for the prediction.

## API Base URL

The API is served under the `/intents` root path. For example, if your application is running on `http://localhost:<port>`, the API endpoints will be accessible under `http://localhost:<port>/intents`.

## Endpoints

### 1. Root

*   **GET /**
    *   **Description:** Returns a welcome message indicating the API is running and the current environment mode.
    *   **Response:**
        ```json
        {
            "message": "Intent Classifier API is running in [dev/prod] mode. Check /redoc for more info."
        }
        ```

### 2. Predict Confusion Intent

*   **POST /confusion**
    *   **Description:** Predicts the intent of a given text string, specifically for the "confusion" model.
    *   **Request Body:**
        ```json
        {
            "text": "Your input text string here"
        }
        ```
    *   **Response (Success - 200 OK):**
        ```json
        {
            "text": "Your input text string here",
            "prediction": "predicted_intent_name",
            "certainty": 0.95
        }
        ```
        Where `prediction` is the most likely intent and `certainty` is the model's confidence score for that prediction (a float between 0 and 1).
    *   **Responses (Error):
        *   `401 Unauthorized` if authentication fails (in production mode).
        *   `500 Internal Server Error` if the model is not available or an unexpected error occurs during prediction.

## Authentication

*   **Development Mode (`ENV=dev`):** Authentication is skipped, and a default user (`dev_user`) is assumed.
*   **Production Mode (`ENV=prod` or not set):** Authentication is required. The API expects a token to be verified (implementation details for token verification are in `app.utils.verify_token`, which is not part of this specific file but is referenced).

## Logging

The API performs detailed logging of requests, responses, and internal operations. Logs are output to both the console and a file named `app.log`.

## Database

Prediction results and relevant information are logged to a MongoDB collection named `intent_logs`.

## API Documentation

Interactive API documentation is available through:

*   **Swagger UI:** `/docs` (e.g., `http://localhost:<port>/intents/docs`)
*   **ReDoc:** `/redoc` (e.g., `http://localhost:<port>/intents/redoc`)

## Key Technologies

*   **FastAPI:** For building the API.
*   **TensorFlow/Keras:** For the machine learning model serving.
*   **MongoDB:** For logging prediction data.
*   **Pydantic:** For data validation (used by FastAPI).

---

This README is used to populate the description field in the FastAPI application's automatic API documentation.
