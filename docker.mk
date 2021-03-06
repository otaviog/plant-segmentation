about:
	@echo "Docker tasks"

build:
	docker build . -t otaviog/plant-segmentation

start:
	docker run --gpus all\
		--volume=`pwd`/dataset:/workspace/plant-segmentation/dataset\
		--user=`id -u`:`id -g`\
		--env="DISPLAY"\
		--env=NVIDIA_DRIVER_CAPABILITIES=all\
		--env=XAUTHORITY\
		--volume="/etc/group:/etc/group:ro"\
		--volume="/etc/passwd:/etc/passwd:ro"\
		--volume="/etc/shadow:/etc/shadow:ro"\
		--volume="/etc/sudoers.d:/etc/sudoers.d:ro"\
		--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"\
		--ipc=host\
		--volume=${HOME}:${HOME}\
		-it otaviog/plant-segmentation /bin/bash

push:
	docker tag otaviog/plant-segmentation otaviog/plant-segmentation:latest
	docker push otaviog/plant-segmentation:latest
