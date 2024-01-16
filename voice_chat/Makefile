.PHONY: run_tgi run_ocr run_api 

MODE := quiet
HOME := /home/mtman/Documents/Repos/yakwith.ai

run_tgi:
	echo 'Starting Dockerized TGI on port 8080 for Mistal-7B-OpenOrca for locally hosted'
	cd /home/mtman/Documents/Repos/yakwith.ai/voice_chat
	. ../.venv/bin/activate
	docker run --rm --gpus all -p 8080:80 -v  $(HOME)models:/data \
	--pull always ghcr.io/huggingface/text-generation-inference:latest \
	--model-id /data/Mistral-7B-OpenOrca \
	--max-input-length 2000 --max-total-tokens 4000

run_tgi_zephyr:
	echo 'Run TGI with local copy of zephyr 7B'
	cd /home/mtman/Documents/Repos/yakwith.ai/voice_chat
	. ../.venv/bin/activate
	docker run --rm --gpus all -p 8080:80 -v  $(HOME)/models:/data \
	--pull always ghcr.io/huggingface/text-generation-inference:latest \
	--model-id /data/zephyr \
	--max-input-length 2000 --max-total-tokens 4000
	
run_ocr:
	echo 'Run Tesseract OCR in docker on port 8885'
	echo 'Run tesseract-server on port 8884 https://github.com/hertzg/tesseract-server'
	docker run -p 8885:8884 hertzg/tesseract-server:latest

run_api:
	echo 'Run yak'' chat_api on port 8884'
	python $(HOME)/voice_chat/chat_api.py

run_backend:
	echo run TGI, OCR, MongoDB and start the api (port 8884)
	echo ===================================================
	docker-compose up -d 
	python $(HOME)/voice_chat/chat_api.py
