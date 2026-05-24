.PHONY: test test-rust test-extractor test-ml-pipeline coverage coverage-extractor coverage-ml-pipeline fmt lint

SEGMENTATOR := segmentator
EXTRACTOR := extractor/yolo_raw_extractor
ML_PIPELINE := segmentator/ml_pipeline

test: test-rust test-extractor test-ml-pipeline

test-rust:
	cd $(SEGMENTATOR) && cargo test

test-extractor:
	cd $(EXTRACTOR) && uv run --group dev pytest

test-ml-pipeline:
	cd $(ML_PIPELINE) && uv run --group dev pytest

coverage: coverage-extractor coverage-ml-pipeline

coverage-extractor:
	cd $(EXTRACTOR) && uv run --group dev pytest --cov --cov-report=term-missing --cov-report=html --cov-report=xml

coverage-ml-pipeline:
	cd $(ML_PIPELINE) && uv run --group dev pytest --cov --cov-report=term-missing --cov-report=html --cov-report=xml

fmt:
	cd $(SEGMENTATOR) && cargo fmt

lint:
	cd $(SEGMENTATOR) && cargo clippy -- -D warnings
