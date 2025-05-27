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
├── requirements.txt         
└── .gitignore              # Git ignore rules
```

### 📦 Setup Instructions

#### 1. Clone the Repository 📁

```bash
git clone https://github.com/adaj/ml-deploy-test.git
cd ml-deploy-test

conda create -n mlops python=3.10
```

#### 2. Train/Load IntentClassifier Model

```bash
pip install - requirements.txt
```


```bash
cd tools
python intent_classifier.py train --config="config.yml" \
    --examples_file="confusion_examples.yml" \
    --save_model="confusion-clf-v1.keras"
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

Create your API token (requires authenticated access to MongoDB cluster):
```bash
# Criar um novo token
python db/tokens.py create --owner="alguem" --expires_in_days=365

# Read the tokens created
python db/tokens.py read_all
```


#### 6. Build and Run with Docker 🐳

```bash
docker-compose up --build
```

The API will be available at http://localhost:8000.

#### 7. Expose API with ngrok 📢

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

#### 8. API Usage 🧪

`POST /predict`

Make a prediction:
```bash
curl -X POST http://localhost:8000/predict \
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
