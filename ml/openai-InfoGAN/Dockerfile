FROM andrewosh/binder-base
#FROM binder-project/binder:binder/images/base/Dockerfile

MAINTAINER Nathan Daly <nhdaly@gmail.com>


USER root

# Install TensorFlow
# Ubuntu/Linux 64-bit, CPU only, Python 2.7
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl --ignore-installed
RUN pip install prettytensor

RUN pip install -r requirements.txt

RUN pip install scipy Pillow
