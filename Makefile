all:
	source ./venv/bin/activate && python nn.py
$(filename):
	source ./venv/bin/activate && python $(filename).py

