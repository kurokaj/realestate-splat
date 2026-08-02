FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG CUDA_ARCHS="75;86;89"
ARG BUILD_JOBS=8
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates gnupg \
    build-essential cmake ninja-build pkg-config ccache \
    gcc-10 g++-10 \
    python3 python3-pip python3-venv awscli \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev libboost-filesystem-dev \
    libeigen3-dev libopenimageio-dev openimageio-tools libopenexr-dev \
    libmetis-dev libgoogle-glog-dev libgflags-dev \
    libsqlite3-dev libglew-dev libcgal-dev \
    libsuitesparse-dev libcurl4-openssl-dev libssl-dev libtbb-dev \
    libblas-dev liblapack-dev libopenblas-dev libopenblas-openmp-dev \
    && mkdir -p /usr/include/opencv4 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
    | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" \
      > /etc/apt/sources.list.d/kitware.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cudss-cuda-12 \
    && rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb

ENV CC=/usr/bin/gcc-10
ENV CXX=/usr/bin/g++-10
ENV CUDAHOSTCXX=/usr/bin/g++-10

WORKDIR /opt/src

RUN git clone --recurse-submodules https://github.com/ceres-solver/ceres-solver.git \
    && cd ceres-solver \
    && mkdir build \
    && cd build \
    && cmake .. -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/ceres-cuda \
      -DBUILD_TESTING=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DUSE_CUDA=ON \
      -DSUITESPARSE=ON \
      -DBLA_VENDOR=OpenBLAS \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    && ninja -j"${BUILD_JOBS}" \
    && ninja install \
    && cd /opt/src \
    && rm -rf ceres-solver

RUN git clone https://github.com/colmap/colmap.git \
    && cd colmap \
    && mkdir build \
    && cd build \
    && cmake .. -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/colmap-cuda \
      -DCOLMAP_CUDA=ON \
      -DCASPAR_ENABLED=ON \
      -DGUI_ENABLED=OFF \
      -DCeres_DIR=/opt/ceres-cuda/lib/cmake/Ceres \
      -DBLA_VENDOR=OpenBLAS \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    && ninja -j"${BUILD_JOBS}" \
    && ninja install \
    && cd /opt/src \
    && rm -rf colmap

ENV PATH=/opt/colmap-cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/ceres-cuda/lib:/opt/colmap-cuda/lib:${LD_LIBRARY_PATH}

RUN ldconfig \
    && ldd /opt/colmap-cuda/bin/colmap > /tmp/colmap_ldd.txt \
    && ! grep -q "not found" /tmp/colmap_ldd.txt \
    && colmap -h >/tmp/colmap_help.txt \
    && colmap global_mapper -h >/tmp/colmap_global_mapper_help.txt \
    && colmap view_graph_calibrator -h >/tmp/colmap_view_graph_calibrator_help.txt \
    && aws --version >/tmp/aws_version.txt

WORKDIR /workspace
CMD ["colmap", "-h"]
