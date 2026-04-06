FROM nvcr.io/nvidia/pytorch:23.11-py3-igpu

WORKDIR /workspace

# Copy your code into the container
COPY . /workspace

# Install dependencies
RUN pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Set the entrypoint (adjust as needed)
CMD ["python", "mps.py"]
