# ml-deploy-test


## 🚀 Features

* Train and predict text intent using a custom classifier
* MongoDB integration for storing predictions
* FastAPI endpoints with automatic Swagger docs
* Dockerized for easy deployment
* Public exposure via ngrok

## 📁 Project Structure

```
intent-classifier/

ml-deploy-test/
├── Dockerfile              # Docker container setup
├── docker-compose.yml      # Compose file for API and MongoDB
├── README.md               # This file
├── app/
│   └── app.py                  # FastAPI app entry point
├── db/
│   ├── models.py           # MongoDB schemas
│   └── engine.py           # MongoDB connection logic
├── tools/
│   ├── models/             # Trained models
│   │   └── ...                 
│   └── classifier_wrapper.py  # IntentClassifier implementation
├── tests/
├── requirements.txt         
└── .gitignore              # Git ignore rules
```

### 📦 Setup Instructions

#### 1. Clone the Repository 📁

```bash
git clone https://github.com/adaj/ml-deploy-test.git
cd ml-deploy-test

# If you are not in a Colab Notebook, create your conda env
conda create -n ml-deploy-test python=3.10

pip install -r requirements.txt
```

#### 2. Train/Load IntentClassifier Model

```bash
python tools/intent_classifier.py train \
  --config="tools/confusion/confusion_config.yml" \
  --examples_file="tools/confusion/confusion_examples.yml" \ --save_model="tools/confusion/confusion-clf-v1.keras"
```

#### 3. Configure MongoDB Atlas 🌱

Create a free MongoDB Atlas cluster: 
1. Sign up at https://www.mongodb.com/cloud/atlas
2. Create a new Shared Cluster (M0)
3. Add your IP to the access list (e.g., 0.0.0.0/0 for testing)
4. Create a database user and password
5. Copy the connection string (e.g., mongodb+srv://<user>:<pass>@cluster.mongodb.net/dbname) 


#### 4. Prepare your environment

Set the connection string as an environment variable MONGO_URI in your `.env`, taken from your MongoDB Atlas cluster.

Set the WANDB_API_KEY to keep track of your model's info.

If you set ENV=prod, create your API token (requires authenticated access to MongoDB cluster):
```bash
# Criar um novo token
python db/tokens.py create --owner="alguem" --expires_in_days=365

# Read the tokens created
python db/tokens.py read_all
```

This token has to be added in the header of API requests.

If you want to run the app locally:

```python
uvicorn app.app:app --host 0.0.0.0 --port 8000
```

Check http://localhost:8000.
#### 6. Optional: Expose API with ngrok 📢

Install ngrok:
```bash
npm install -g ngrok  # or follow instructions at https://ngrok.com/download
```

Authenticate ngrok (replace YOUR_AUTHTOKEN with your token):
```bash
ngrok config add-authtoken YOUR_AUTHTOKEN
```

Start ngrok tunnel:
```bash
ngrok http 8000
```

You’ll receive a public URL like `https://abc123.ngrok.io.

#### 7. Build and Run with Docker 🐳

```bash
docker compose up --build --detach
```

Check the containers and logs:
```bash
docker ps -a
docker compose logs -f
```

After it's up, the API should be available at http://localhost:8000.

#### 8. API Usage 🧪

`POST /confusion`

Make a prediction:
```bash
curl -X POST http://localhost:8000/confusion \
  -H "Content-Type: application/json" \
  -d '{"text": "Não entendo como isso é possível"}'
```

Response:
```json
  {
  "text": "Não entendo como isso é possível",
  "prediction": "confusion",
  "certainty": 0.97
  }
```
Access the interactive Swagger UI at `http://localhost:8000/docs` or via your ngrok URL.


## 🛡️ Boas Práticas de MLOps

Este projeto demonstra algumas práticas essenciais de MLOps para garantir reproduzibilidade e automação:

- **Imagens Docker**: a API e a interface Streamlit são distribuídas em containers isolados. O `Dockerfile.app` e `Dockerfile.streamlit` definem dependências e comandos.
- **Orquestração com Docker Compose**: o arquivo `docker-compose.yml` permite iniciar a API, o Streamlit e o MongoDB de forma integrada.
- **Variáveis de ambiente**: URIs, credenciais e o caminho do modelo são configurados via variáveis de ambiente ou `.env`.
- **Testes automatizados**: os testes em `tests/` validam o `IntentClassifier` e rodam no fluxo de CI.
- **Download automático de modelo**: `fetch_model.sh` obtém o modelo armazenado no Weights & Biases sempre que o container inicia.
- **Logs em MongoDB**: previsões são registradas para análise posterior.

### Workflows do GitHub Actions

A automação do ciclo de vida conta com dois workflows em `.github/workflows`:

1. **ci.yml** – constrói as imagens do `docker-compose.yml` e executa os testes unitários a cada push ou pull request para `main`, usando os segredos como variáveis de ambiente.
2. **docker-publish.yml** – publica a imagem da API no Docker Hub após push para `main` ou execução manual. A imagem é construída com `docker/build-push-action` e enviada usando `DOCKER_USERNAME` e `DOCKER_PASSWORD`.


## 🛠️ Development Notes

* The IntentClassifier can be trained (`IntentClassifier(config, examples_file).train(save_model="model-0.1")`) 
and later, initialized with `load_model="model-0.1"` to load a pre-trained model.

* Training data is loaded from examples_file specified in the config.

* Predictions are stored in MongoDB for later analysis.

## 📚 Resources

* FastAPI Docs: https://fastapi.tiangolo.com/
* MongoDB Atlas: https://www.mongodb.com/cloud/atlas
* ngrok Docs: https://ngrok.com/docs

📄 License
This project is licensed under the MIT License.
