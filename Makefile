model_path := ./prediction_model
vocab_path := ./vocab.txt
data_dir := '../tagged'

run:
	MODEL_PATH=$(model_path) \
	VOCAB_PATH=$(vocab_path) \
	python app.py

train:
	DATA_DIR=$(data_dir) \
	MODEL_PATH=$(model_path) \
	VOCAB_PATH=$(vocab_path) \
	BATCH=8 \
	EPOCHS=125 \
	python trainer.py

infer:
	DATA_DIR=$(data_dir) \
	MODEL_PATH=$(model_path) \
	VOCAB_PATH=$(vocab_path) \
	python inferer.py

test:
	pytest -vvv .

.PHONY: train infer run test