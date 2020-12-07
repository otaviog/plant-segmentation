FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
LABEL maintainer="otavio.b.gomes@gmail.com"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt -yq install git graphviz libgtk2.0

WORKDIR /workspace
ADD requirements.txt .
RUN pip install -r requirements.txt

RUN pip uninstall -y opencv-python-headless
RUN pip install opencv-python==4.1.1.26

ADD . plant-segmentation
RUN pip install -e plant-segmentation/module
WORKDIR /workspace/plant-segmentation

RUN chmod -R o+rw .
