# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install MLflow
RUN pip install mlflow

# Expose the port that MLflow will run on
EXPOSE 5000

# Define environment variable
ENV MLFLOW_HOME /usr/src/app

# Run MLflow server when the container launches
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]