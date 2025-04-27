# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install ssh-keyscan (part of the openssh-client package)
RUN apt-get update && apt-get install -y openssh-client

# Create .ssh folder and scan host key
RUN mkdir -p /root/.ssh 
# && \
#     ssh-keyscan -H 195.201.98.37 >> /root/.ssh/known_hosts && \
#     chmod 644 /root/.ssh/known_hosts

COPY id_rsa /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa

# Install MLflow
RUN pip install mlflow==2.19.0 paramiko==3.5.0 pysftp==0.2.9

# Expose the port that MLflow will run on
EXPOSE 5000

# Define environment variable
ENV MLFLOW_HOME /usr/src/app

# By default, we can store artifacts in /usr/src/app/mlruns (inside the container)
# Also specify a backend store. For a quick test, you can use sqlite.
CMD ["mlflow", "server", \
    "--backend-store-uri", "sqlite:///mlflow.db", \
    "--default-artifact-root", "sftp://root@195.201.98.37:/usr/src/app/mlruns", \
    "--host", "0.0.0.0", \
    "--port", "5000"]