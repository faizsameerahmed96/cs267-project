FROM apache/airflow:latest

# Install custom Python packages from requirements.txt
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
