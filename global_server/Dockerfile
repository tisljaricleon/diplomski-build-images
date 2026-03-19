FROM python:3.10-slim-trixie

WORKDIR /home
COPY requirements.txt /home/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://www.piwheels.org/simple -r requirements.txt --no-deps

COPY torch-2.1.2-cp310-cp310-linux_aarch64.whl /tmp/
COPY torchvision-0.16.2-cp310-cp310-linux_aarch64.whl /tmp/

RUN pip install /tmp/torch-2.1.2-cp310-cp310-linux_aarch64.whl /tmp/torchvision-0.16.2-cp310-cp310-linux_aarch64.whl

COPY . /home/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "global_server.py"]

