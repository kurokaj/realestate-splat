FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG NS_REF="v1.1.4"
ARG BUILD_JOBS=8

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/workspace/pixi/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

ENV TCNN_CUDA_ARCHITECTURES=75\;86\;89
ENV TORCH_CUDA_ARCH_LIST=7.5\;8.6\;8.9
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates \
    build-essential cmake ninja-build pkg-config \
    gcc-11 g++-11 \
    python3 python3-pip python3-venv \
    ffmpeg \
    libgl1 libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    libgomp1 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11
ENV CUDAHOSTCXX=/usr/bin/g++-11

RUN mkdir -p /workspace/pixi \
    && curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/workspace/pixi bash

WORKDIR /workspace/opt

RUN git clone https://github.com/nerfstudio-project/nerfstudio.git \
    && cd nerfstudio \
    && git checkout "${NS_REF}"

WORKDIR /workspace/opt/nerfstudio

RUN pixi run post-install

RUN pixi run python -c "import torch; print(torch.__version__, torch.version.cuda)" \
    && pixi run python -c "import nerfstudio; print('nerfstudio OK')" \
    && pixi run ns-train --help >/tmp/ns_train_help.txt \
    && pixi run ns-process-data images --help >/tmp/ns_process_data_images_help.txt

WORKDIR /workspace

CMD ["/workspace/pixi/bin/pixi", "run", "--manifest-path", "/workspace/opt/nerfstudio/pixi.toml", "ns-train", "--help"]
