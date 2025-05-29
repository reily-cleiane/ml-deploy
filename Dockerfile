FROM python:3.11

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir fastapi[all] pymongo python-dotenv

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
