# Use the official Python image as the base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Set the entry point for the container
CMD ["python", "PredictionAPI/run.py"]
