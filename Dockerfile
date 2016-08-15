FROM andrewosh/binder-base
#FROM binder-project/binder:binder/images/base/Dockerfile

MAINTAINER Nathan Daly <nhdaly@gmail.com>


USER root

# Install TensorFlow
# Ubuntu/Linux 64-bit, CPU only, Python 2.7
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl --ignore-installed
#RUN b.gcr.io/tensorflow/tensorflow:latest
RUN pip install prettytensor

# Add Julia dependencies
RUN apt-get update
RUN apt-get install -y julia libnettle4 && apt-get clean

USER main

# Install Julia kernel
RUN julia -e 'Pkg.add("IJulia")'
RUN julia -e 'Pkg.add("Gadfly")' && julia -e 'Pkg.add("RDatasets")'


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
RUN ln ffmpeg-3.1.2-64bit-static/ffmpeg /usr/local/bin/ffmpeg

