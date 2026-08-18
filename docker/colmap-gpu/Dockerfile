FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ARG CUDA_ARCHS="75;86;89"
ARG BUILD_JOBS=8
ARG COLMAP_REF=4.0.4
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates gnupg \
    build-essential cmake ninja-build pkg-config ccache \
    gcc-10 g++-10 \
    python3 python3-dev python3-pip \
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
    && apt-get install -y --no-install-recommends cudss-cuda-12 cudnn9-cuda-12 \
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

RUN git clone --branch "${COLMAP_REF}" https://github.com/colmap/colmap.git \
    && cd colmap \
    && mkdir build \
    && cd build \
    && cmake .. -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/colmap-cuda \
      -DCOLMAP_CUDA=ON \
      -DONNX_ENABLED=ON \
      -DCASPAR_ENABLED=ON \
      -DGUI_ENABLED=OFF \
      -DCeres_DIR=/opt/ceres-cuda/lib/cmake/Ceres \
      -DBLA_VENDOR=OpenBLAS \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    && ninja -j"${BUILD_JOBS}" \
    && ninja install

# COLMAP's PyCOLMAP build generates type stubs with ruff. Keep this tool in
# the builder only; it is not required by the runtime image. Install it in a
# standard executable directory because the isolated pip build invokes ruff
# directly from generate_stubs.sh.
RUN python3 -m pip install --no-cache-dir --prefix=/usr/local ruff==0.6.9 \
    && command -v ruff

RUN cd /opt/src/colmap \
    && test -f pyproject.toml \
    && mkdir -p /opt/pycolmap \
    && CMAKE_ARGS="-Dcolmap_DIR=/opt/colmap-cuda/share/colmap/cmake -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}" \
      python3 -m pip install --no-cache-dir --target /opt/pycolmap . \
    && cd /opt/src \
    && rm -rf colmap


FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    awscli git ca-certificates curl \
    python3 python3-venv \
    libboost-program-options1.74.0 libboost-graph1.74.0 libboost-system1.74.0 libboost-filesystem1.74.0 \
    libopenimageio2.2 openimageio-tools \
    libmetis5 libgoogle-glog0v5 libgflags2.2 \
    libsqlite3-0 libglew2.2 libopengl0 \
    libcgal-dev libgmp10 libmpfr6 \
    libsuitesparseconfig5 libcholmod3 libspqr2 libcamd2 libccolamd2 libcolamd2 libamd2 \
    libcurl4 libssl3 libtbb12 \
    libblas3 liblapack3 libopenblas0-openmp \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cudss-cuda-12 cudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb

COPY --from=builder /opt/ceres-cuda /opt/ceres-cuda
COPY --from=builder /opt/colmap-cuda /opt/colmap-cuda
COPY --from=builder /opt/pycolmap /opt/pycolmap

ENV PATH=/opt/colmap-cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/ceres-cuda/lib:/opt/colmap-cuda/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/opt/pycolmap

RUN set -eux; \
    ldconfig; \
    ldd /opt/colmap-cuda/bin/colmap | tee /tmp/colmap_ldd.txt; \
    ! grep -q "not found" /tmp/colmap_ldd.txt; \
    if [ -f /opt/colmap-cuda/lib/libonnxruntime_providers_cuda.so ]; then \
      ldd /opt/colmap-cuda/lib/libonnxruntime_providers_cuda.so | tee /tmp/onnxruntime_cuda_provider_ldd.txt; \
      ! grep -q "not found" /tmp/onnxruntime_cuda_provider_ldd.txt; \
    fi; \
    find /usr /opt -name "libcudnn.so.9*" -print | tee /tmp/cudnn_paths.txt; \
    test -s /tmp/cudnn_paths.txt; \
    colmap -h >/tmp/colmap_help.txt; \
    colmap global_mapper -h >/tmp/colmap_global_mapper_help.txt; \
    colmap view_graph_calibrator -h >/tmp/colmap_view_graph_calibrator_help.txt; \
    python3 -c 'import pycolmap; assert hasattr(pycolmap, "match_image_pairs"); assert hasattr(pycolmap, "match_vocabtree"); print(pycolmap.__version__)' >/tmp/pycolmap_version.txt; \
    aws --version >/tmp/aws_version.txt

# Future hardening option: preload COLMAP's ALIKED/LightGlue ONNX models into
# the image and pass explicit model paths from scripts/run_colmap.py. For now
# pods download those small model files on first ALIKED/LightGlue use, matching
# COLMAP's default behavior and avoiding build-time GitHub/release dependency.

WORKDIR /workspace
CMD ["colmap", "-h"]
