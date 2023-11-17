.PHONY: run_tgi run_ocr run_api run_chat run_local_chat run_local_web_chat

MODE := quiet

run_tgi:
	echo 'Starting Dockerized TGI on port 8080 for Mistal-7B-OpenOrca for locally hosted'
	cd /home/mtman/Documents/Repos/yakwith.ai/voice_chat
	. .venv/bin/activate
	docker run --rm --gpus all -p 8080:80 -v  /home/mtman/Documents/Repos/yakwith.ai/models:/data \
	--pull always ghcr.io/huggingface/text-generation-inference:latest \
	--model-id /data/Mistral-7B-OpenOrca \
	--max-input-length 2000 --max-total-tokens 4000

run_tgi_zephyr:
	echo 
	cd /home/mtman/Documents/Repos/yakwith.ai/voice_chat
	. .venv/bin/activate
	docker run --rm --gpus all -p 8080:80 -v  /home/mtman/Documents/Repos/yakwith.ai/models:/data \
	--pull always ghcr.io/huggingface/text-generation-inference:latest \
	--model-id /data/zephyr \
	--max-input-length 2000 --max-total-tokens 4000
run_ocr:
	echo 'Run tesseract-server on port 8884 https://github.com/hertzg/tesseract-server'
	docker run -p 8884:8884 hertzg/tesseract-server:latest

run_api:
	echo 'Run ya api on port 8884'
	python /home/mtman/Documents/Repos/yakwith.ai/voice_chat/chat_api.py

run_local_web_chat: voice_chat/local_web_chat_client.py
	@echo 'Start voice chat via browser - client and server co-located and so no TURN server is needed.'
	. .venv/bin/activate; \
	streamlit run voice_chat/local_web_chat_client.py -- --mode $(MODE) --local

run_remote_web_chat: voice_chat/local_web_chat_client.py
	@echo 'Start voice chat via remote browser - requires the https connection; use ssl-proxy'
	. .venv/bin/activate; \
	voice_chat/Ssl/ssl-proxy -from 0.0.0.0:8502 -to 0.0.0.0:8501 & \
	streamlit run voice_chat/local_web_chat_client.py --  --mode $(MODE) --stream_type web
run_local_chat:
	@echo 'Run voice chat with audio source from local machine. Uses PuAudio and requires that a sound server be running (e.g. PulseAudio )'
	. .venv/bin/activate; \
	streamlit run voice_chat/chat_client.py -- --task chat_with_agent
kill_ssl:
	@echo 'stop ssl-proxy'
	ps -aux | grep ssl-proxy | kill -9 $$(awk '(NR==1){print $$2}' )

install: requirements.txt
	# ffmpeg requried from PyAv (its a pythonic binding to FFMpeg)
	sudo apt install ffmpeg; \
	pip install -r requirements.txt
