FROM binder-project/binder:binder/images/base/Dockerfile



MAINTAINER Nathan Daly <nhdaly@gmail.com>

USER root
#
## Add Julia dependencies
#RUN apt-get update
#RUN apt-get install -y julia libnettle4 && apt-get clean
#

RUN b.gcr.io/tensorflow/tensorflow

#USER main
