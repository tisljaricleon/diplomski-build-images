FROM python:3.10-slim-trixie

WORKDIR /home
COPY requirements.txt /home/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://www.piwheels.org/simple -r requirements.txt

COPY . /home/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "global_server.py"]

