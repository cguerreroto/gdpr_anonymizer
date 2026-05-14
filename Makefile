.PHONY: test
test:
	cd segmentator && cargo test
	cd extractor/yolo_raw_extractor && uv run --group dev pytest
