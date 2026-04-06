# MPS Container Example

This folder contains an example setup for running PyTorch with NVIDIA MPS in a Docker container.

## Usage

1. **Build the Docker image:**

   ```sh
   docker build -t my-mps-test .
   ```

2. **Run two containers to test MPS:**

   ```sh
   docker run --rm my-mps-test
   docker run --rm my-mps-test

   docker run --rm --gpus all -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 my-mps-test
   docker run --rm --gpus all -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 my-mps-test
   ```

   Run these in two terminals to test MPS.

3. **Add your code:**
   - Place your `mps.py` and any other files in this folder.
   - List dependencies in `requirements.txt`.

## Notes

- Make sure the NVIDIA driver and Docker setup support MPS and the `nvcr.io` images.
- You may need to start the MPS control daemon on the host:

  ```sh
  sudo nvidia-cuda-mps-control -d
  ```
