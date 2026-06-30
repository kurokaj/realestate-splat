#!/usr/bin/env bash
set -euo pipefail

apt update
apt install -y \
  libblas3 \
  liblapack3 \
  libgomp1 \
  libgl1 \
  libglib2.0-0t64 \
  libglew2.2 \
  libboost-program-options1.83.0 \
  libcholmod5 \
  libceres4t64 \
  libgoogle-glog0v6t64 \
  libopenimageio2.4t64 \
  libopenimageio-dev \
  openimageio-tools \
  libqt5widgets5t64 \
  libqt5gui5t64 \
  libqt5core5t64 \
  libsqlite3-0 \
  libfreeimage3 \
  libflann1.9 \
  libmetis5 \
  libsuitesparseconfig7 

# Nerfstudio related
apt install -y gcc-11 g++-11

ldd /workspace/opt/colmap-install/bin/colmap | grep "not found" || true