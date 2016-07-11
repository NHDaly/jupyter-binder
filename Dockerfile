FROM andrewosh/binder-base
#FROM binder-project/binder:binder/images/base/Dockerfile

MAINTAINER Nathan Daly <nhdaly@gmail.com>


USER root

# Install TensorFlow
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl --ignore-installed
#RUN b.gcr.io/tensorflow/tensorflow:latest

# Add Julia dependencies
RUN apt-get update
RUN apt-get install -y julia libnettle4 && apt-get clean

USER main

# Install Julia kernel
RUN julia -e 'Pkg.add("IJulia")'
RUN julia -e 'Pkg.add("Gadfly")' && julia -e 'Pkg.add("RDatasets")'


# Install OpenAI's gym
USER root
RUN apt-get install -y python-numpy python-dev python-opengl

# Set up x11vnc for gui access over vnc
run     apt-get install -y x11vnc xvfb
run     mkdir ~/.vnc
# Setup a password
run     x11vnc -storepasswd 1234 ~/.vnc/passwd

USER main

RUN pip install gym


# DELETE THIS BEFORE PUSHING TO GITHUB:
# Autostart notebook (might not be the best way to do it, but it does the trick)
run     bash -c 'echo "./start-notebook.sh '--ip=0.0.0.0'" >> /.bashrc'


