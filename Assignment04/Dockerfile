# 1. Use an official Python runtime as a parent image
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your app files and the trained model into the container
COPY app.py .
COPY score.py .
COPY best_model.joblib .

# 5. Make port 5000 available to the world outside this container
EXPOSE 5000

# 6. Run app.py when the container launches
CMD ["python", "app.py"]