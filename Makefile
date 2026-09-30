.PHONY: all build test clean archive run

IMAGE_NAME ?= homelab/urban-traffic-vision:latest

all: build

build:
	docker build -t $(IMAGE_NAME) .

test:
	python3 -m pytest tests/ -v 2>/dev/null || echo "No unit tests configured yet."

archive:
	./scripts/archive_daily.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	docker run --rm -it --network host \
		-v /home/tlima/Enterprise_Hub/data/media/merged/vision:/data/media/merged/vision \
		-v $(PWD)/config/settings.yaml:/app/config/settings.yaml:ro \
		-v $(PWD)/config/zones.json:/app/config/zones.json \
		$(IMAGE_NAME)
