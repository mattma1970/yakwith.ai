run_tgi:
	echo 'Starting Dockerized TGI on port 8080 for Mistal-7B-OpenOrca for locally hosted'
	cd /home/mtman/Documents/Repos/yakwith.ai/voice_chat
	. .venv/bin/activate
	docker run --rm --gpus all -p 8080:80 -v  /home/mtman/Documents/Repos/yakwith.ai/models:/data --pull always ghcr.io/huggingface/text-generation-inference:latest --model-id /data/Mistral-7B-OpenOrca 
run_ocr:
	echo 'Run tesseract-server on port 8884 https://github.com/hertzg/tesseract-server'
	docker run -p 8884:8884 hertzg/tesseract-server:latest

