# ml-deploy-test

Repositório da API para classificação de intenção baseada em texto, usando FastAPI + MongoDB.

```
intent-classifier/
├── app.py
├── Dockerfile
├── docker-compose.yml
├── README.md
├── app/
│   └── classifier_wrapper.py
├── models/
│   └── ...
├── db/
│   ├── models.py
│   └── engine.py
```

## Intent Classifier API

### Endpoints

- `POST /predict`  
  **Body**:
  ```json
  {
    "text": "Olá, tudo bem?"
  }
  ```
  **Response**:
  ```json
    {
    "text": "Olá, tudo bem?",
    "prediction": "saudação",
    "certainty": 0.97
    }
  ```

### Rode localmente

`docker-compose up --build`
