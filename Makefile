.PHONY: install lint test e2e docker serve lock

install:
	pip install -e ".[dev]"
lint:
	ruff check src tests
test:
	pytest -q
e2e:
	tinyforge pipeline --preset micro --steps 300
docker:
	docker build -t tinyforge:latest .
serve:
	tinyforge serve
lock:
	./requirements/lock.sh
