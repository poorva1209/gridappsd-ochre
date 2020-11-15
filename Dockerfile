FROM ubuntu:20.04

WORKDIR /tmp
RUN export DEBIAN_FRONTEND=noninteractive && \
	apt-get -y update && apt-get install -y tzdata && \
# set your timezone
	ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get install -y wget \
    libboost-filesystem1.67.0 libboost-program-options1.67.0 \
    libboost-test1.67.0 libzmq5 libtool libxerces-c-dev git build-essential \
    libboost-dev \
    libboost-test-dev \
    libzmq5-dev \
    python3-dev \
    swig \
    cmake \
    git 

ENV PYTHONPATH /usr/local/python:/app
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda && \
    /miniconda/bin/conda init bash

COPY ./environment.yml /app/environment.yml
COPY src/ /app/src/
WORKDIR /app
RUN /miniconda/bin/conda env create

WORKDIR /tmp
RUN git clone --single-branch https://github.com/GMLC-TDC/HELICS
WORKDIR /tmp/HELICS/build
RUN cmake \
  -DBUILD_PYTHON_INTERFACE=OFF \
  -DHELICS_BUILD_CXX_SHARED_LIB=ON \
  ..
RUN make -j6 && make install

WORKDIR /tmp
RUN git clone -b develop --single-branch https://github.com/gridlab-d/gridlab-d.git
WORKDIR /tmp/gridlab-d
RUN autoreconf -if
RUN ./configure \
    --prefix=/usr/local \
    --with-helics=/tmp/HELICS/install \
    --enable-silent-rules \
    "CFLAGS=-g -O0 -w" \
    "CXXFLAGS=-g -O0 -w -std=c++14" \
    "LDFLAGS=-g -O0 -w" && \
    make -j6 && make install && ldconfig

COPY . /app
WORKDIR /app