FROM nvcr.io/nvidia/pytorch:23.11-py3-igpu

WORKDIR /home
COPY requirements.txt /home/

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY . /home/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["uvicorn", "client_serving:app", "--host", "0.0.0.0", "--port", "8000"]