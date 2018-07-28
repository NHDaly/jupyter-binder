FROM andrewosh/binder-base
#FROM binder-project/binder:binder/images/base/Dockerfile

MAINTAINER Nathan Daly <nhdaly@gmail.com>


USER root

# Install TensorFlow
# Ubuntu/Linux 64-bit, CPU only, Python 2.7
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl --ignore-installed
#RUN b.gcr.io/tensorflow/tensorflow:latest
RUN pip install prettytensor

RUN apt-get update

# Add Julia dependencies
RUN apt-get install -y ca-certificates
# Fix curl
RUN mkdir -p /etc/pki/tls/certs
RUN cp /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt

# Install Julia
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/0.6/julia-0.6.4-linux-x86_64.tar.gz
RUN tar -xzf julia-0.6.4-linux-x86_64.tar.gz

RUN ln -s "$(pwd)/"julia-*/bin/julia /usr/local/bin/julia

USER main

# Install Julia kernel
RUN julia -E 'Pkg.add("IJulia")'
# Install other Julia packages
RUN julia -E 'Pkg.add("Plots"); Pkg.add("Blink")'


# Install OpenAI's gym
USER root
RUN apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
USER main

RUN pip install gym


# Playing with FFMPEG / Jupyter
# Install FFMPEG
RUN wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
RUN tar xvfJ ffmpeg-release-64bit-static.tar.xz
USER root
# Note that we don't know exactly which version of ffmpeg will come down.
RUN ln ffmpeg-*-64bit-static/ffmpeg /usr/local/bin/ffmpeg

