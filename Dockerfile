FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
LABEL maintainer="otavio.b.gomes@gmail.com"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt -yq install git graphviz

WORKDIR /workspace
ADD requirements.txt .
RUN pip install -r requirements.txt

ADD . plant-segmentation
RUN pip install -e plant-segmentation/module
WORKDIR /workspace/plant-segmentation

RUN chmod -R o+rw .
