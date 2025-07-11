FROM python:3.11

WORKDIR /ml-deploy-test

# Instala dependências do sistema (se precisar de libs para TF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgrpc++1 libstdc++6 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Ensure the models are accessible (adjust path if necessary)
# If your models are large and not checked into git, you might need a different strategy for getting them into the image.
COPY app ./app
COPY app/fetch_model.sh /usr/local/bin/fetch_model.sh
RUN chmod +x /usr/local/bin/fetch_model.sh
ENTRYPOINT ["fetch_model.sh"]

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# # Command to run the application
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
# Alternatively:
# Better utilization of multiple CPUs and fault isolation — if a worker crashes, Gunicorn restarts only that one.
# Can handle more simultaneous requests (via internal round-robin).
# CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", \
#     "-w", "2", \
#     "-b", "0.0.0.0:8000", \
#     "app.app:app"]

