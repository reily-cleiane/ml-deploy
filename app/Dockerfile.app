FROM python:3.11-slim

WORKDIR /ml-deploy-test

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .
# Ensure the models are accessible (adjust path if necessary)
# If your models are large and not checked into git, you might need a different strategy for getting them into the image.

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
