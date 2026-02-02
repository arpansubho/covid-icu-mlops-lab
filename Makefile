install:
	python -m pip install -r requirements.txt

clean:
	python -m src.clean_data

train:
	python -m src.train

test:
	python -m pytest

mlflow:
	python -m mlflow ui
