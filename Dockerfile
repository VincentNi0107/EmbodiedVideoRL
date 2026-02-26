FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    git wget ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# PyTorch + flash-attn (slow to build, cache this layer)
RUN pip install --no-cache-dir torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir packaging ninja \
    && pip install --no-cache-dir flash-attn==2.7.0.post2 --no-build-isolation

# Copy project files (before SAM3 clone, so code changes don't invalidate SAM3 layer)
COPY pyproject.toml requirements-lint.txt env_setup.sh ./
COPY fastvideo/ fastvideo/
COPY server/ server/
COPY scripts/ scripts/
COPY tools/ tools/
COPY tests/ tests/
COPY assets/ assets/
COPY data/rl_train/ data/rl_train/
COPY .env.example .

# Install project dependencies
RUN pip install --no-cache-dir -r requirements-lint.txt \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir ml-collections absl-py inflect==6.0.4 openai

# Clone and install SAM3
RUN git clone https://github.com/facebookresearch/sam3.git \
    && cd sam3 && pip install --no-cache-dir -e .

# Checkpoints are mounted at runtime, not baked into the image
# docker run -v /path/to/ckpts:/app/ckpts ...
VOLUME ["/app/ckpts"]

# Default: bash (user chooses training or inference command)
CMD ["bash"]
