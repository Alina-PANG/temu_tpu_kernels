# syntax=docker/dockerfile:1.6
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      git \
      libgomp1 \
      libgoogle-perftools4 \
      procps \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf "$(which python)" /usr/bin/python3

RUN pip install --no-cache-dir uv

# Install the pinned TorchTPU + libtpu + Pallas/Tokamax stack from public PyPI
# and GitHub so any user can build the container from a fresh git clone.
ARG TOKAMAX_COMMIT=0d3e82b41ddc63e5fdd5d9c02c76894377cc9a18
RUN uv pip install --system --no-cache \
      "setuptools<81" \
      "numpy>=2.1" \
      "absl-py>=2.3.0" \
      "portpicker>=1.6" \
      "frozendict>=2.0" \
      "immutabledict>=4.0" \
      "pyzmq>=27.0" \
      "jax==0.12.0.1" \
      "jaxlib==0.12.0.1" \
      "einshape" \
      "jaxtyping>=0.3" \
      "pydantic>=2.11.0" \
      "qwix>=0.1.2" \
      "tqdm" \
      "tensorboardx" \
      "einops>=0.8.0" \
      "chex" \
      "tpu-info" \
      "xprof==2.24.2" \
      "tensorboard>=2.20" \
      "tensorboard-plugin-profile" \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      --index-strategy unsafe-best-match \
    && uv pip install --system --no-cache \
      "torch==2.15.0+cpu" \
      "libtpu==0.0.53" \
      "torch-tpu==0.3.0" \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      --index-strategy unsafe-best-match \
    && uv pip install --system --no-cache --no-deps \
      "git+https://github.com/openxla/tokamax.git@${TOKAMAX_COMMIT}" \
      "vllm-torchtpu-core==0.33.0" \
      "jax-flash-attn2==0.0.3"

ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
ENV PYTHONPATH="/workspace/temu_kbench"
ENV PYTHONUNBUFFERED="1"
ENV JAX_PLATFORMS="cpu"

WORKDIR /workspace/temu_kbench
COPY . /workspace/temu_kbench/

RUN chmod +x /workspace/temu_kbench/run_all.sh \
      /workspace/temu_kbench/torchtpu_kernel_bench.py \
      /workspace/temu_kbench/gcs_upload.py \
      /workspace/temu_kbench/verify_build.py \
      /workspace/temu_kbench/test_torchtpu_bench.py \
      /workspace/temu_kbench/examples/standalone_custom_pallas_op.py \
    && ln -sf /workspace/temu_kbench/run_all.sh /usr/local/bin/run_all.sh \
    && ln -sf /workspace/temu_kbench/gcs_upload.py \
      /usr/local/bin/gcs_upload.py \
    && ln -sf /workspace/temu_kbench/torchtpu_kernel_bench.py \
      /usr/local/bin/torchtpu_kernel_bench.py

RUN python3 /workspace/temu_kbench/verify_build.py \
    && python3 /workspace/temu_kbench/test_torchtpu_bench.py -v

CMD ["/bin/bash"]
