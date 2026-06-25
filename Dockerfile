# Use Ubuntu base image for arm64 (works on M1)
FROM ubuntu:24.04

# Avoid tzdata prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install compilers, MPI, OpenMP runtime
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenmpi-dev \
    openmpi-bin \
    libomp-dev \
    gdb \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
