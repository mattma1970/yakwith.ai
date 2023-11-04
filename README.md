# yakwith.ai

Locally served voicebots
vooicebot requires custom version of griptape. 
Use:
```
git clone https://github.com/mattma1970/griptape/tree/yak_dev
pip install -e . 
```
This branch (at 2/11/23) contains the HF InferenceClient prompt driver for locally server HF models. 

Model serving uses HuggingFace TGI. Using the docker to run TGI 
TGI Offical Docker installation [https://huggingface.co/docs/text-generation-inference/quicktour]
Note that model-id can be set to a locally saved model.

** Voicebot **
``` 
# from root directory 
source .venv/bin/activate
make run_tgi  # this will launch TGI and expose port 8080 for model serving.

cd voice_chat
python chat_api.py # start fastAPi webserver listentin on port 8884

... WIP



```

