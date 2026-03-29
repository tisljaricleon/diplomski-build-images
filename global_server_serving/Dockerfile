FROM python:3.10-slim

WORKDIR /home

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy serving script and model directory
COPY global_server_serving.py ./
COPY model/ ./model/

EXPOSE 8000

CMD ["python", "global_server_serving.py"]
