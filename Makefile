.PHONY: setup check test lint types data bench replay annotate sim report demo demo-sim serve

setup:            ## runtime, training and dev dependencies
	uv sync --all-extras

check: lint types test

lint:
	uv run ruff check src tests

types:
	uv run mypy src

test:
	uv run pytest

data:             ## CNRPark+EXT patches, splits and full frames (about 1.5 GB), then the cache
	uv run spark data download --full
	uv run spark data cache

bench:            ## the full benchmark behind docs/benchmark.md (about an hour on an M1 Max)
	uv run spark bench run

replay:           ## end-to-end check on whole camera frames, shipped model and baselines
	uv run spark bench replay --model models/default.onnx models/sparknet-w0.25-r32-int8.onnx \
		models/hog-linear.npz models/edge-energy.npz

annotate:         ## the camera frame in the README
	uv run spark bench annotate --frame 2015-12-22_1022

sim:              ## pricing policy comparison behind docs/pricing.md
	uv run spark sim --days 28 --seeds 5 --out results/pricing.json

report:
	uv run spark bench report

demo:             ## replay CNRPark+EXT camera 8 through the full system
	uv run spark demo

demo-sim:         ## simulated cars, no dataset needed
	uv run spark demo --synthetic

serve:
	uv run spark serve
