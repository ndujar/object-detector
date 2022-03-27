# Dockerfile for building an image with a source FiftyOne install atop a
# Debian-based Linux distribution.
#
# By default, Ubuntu 20.04 and Python 3.8 are used, but these can be customized
# via ARGs.
#
# ARGs::
#
#   BASE_IMAGE (ubuntu:20.04): The Debian-based image to build from
#   PYTHON_VERSION (3.8): The Python version to install
#   ROOT_DIR (/fiftyone): The name of the directory within the container that
#       should be mounted when running
#
# Example usage::
#
#   # Build
#   make python
#   docker build -t voxel51/fiftyone .
#
#   # Run
#   SHARED_DIR=/path/to/shared/dir
#   docker run \
#       -v ${SHARED_DIR}:/fiftyone \
#       -p 5151:5151 \
#       -it voxel51/fiftyone
#
# Copyright 2017-2022, Voxel51, Inc.
# voxel51.com
#

# The base image to build from; must be Debian-based (eg Ubuntu)
ARG BASE_IMAGE=ubuntu:20.04
FROM $BASE_IMAGE

# The Python version to install
ARG PYTHON_VERSION=3.8

#
# Install system packages
#

RUN apt -y update \
    && apt -y --no-install-recommends install software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt -y update \
    && apt -y upgrade \
    && apt -y --no-install-recommends install tzdata \
    && TZ=Etc/UTC \
    && apt -y --no-install-recommends install \
        build-essential \
        ca-certificates \
        cmake \
        cmake-data \
        pkg-config \
        libcurl4 \
        libsm6 \
        libxext6 \
        libssl-dev \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        unzip \
        curl \
        wget \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        ffmpeg \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -s /usr/local/lib/python${PYTHON_VERSION} /usr/local/lib/python \
    && curl https://bootstrap.pypa.io/get-pip.py | python \
    && rm -rf /var/lib/apt/lists/*

#
# Install Python dependencies
#
# Other packages you might want:
#   torch torchvision: Torch model training/zoo datasets
#   tensorflow tensorflow-datasets: TF model training/zoo datasets
#   pycocotools: COCO-style evaluation
#   notebook>=5.3 ipywidgets>=7.5: Jupyter notebooks
#   flash>=0.4: Lightning Flash integration
#   apache_beam: Apache Beam integration
#   labelbox: Labelbox integration
#   shapely: Polyline evaluation
#   rasterio: GeoTIFF images
#   pydicom: DICOM images
#

RUN pip --no-cache-dir install --upgrade pip setuptools wheel ipython

RUN pip install fiftyone
RUN pip install torch torchvision
RUN pip install pycocotools
RUN pip install jupyter

COPY scripts /scripts