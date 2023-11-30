# yakwith.ai

A fastAPI backend for locally hosted voicebots. Its designed to work with the (react.js front-end)[https://github.com/mattma1970/yak_react_frontend].

As at 30/11/23 it is configured to use:
* Azure Congnitive Services for Text-to-Speech;
* a custom version of (griptape.ai)[http://griptape.ai] modified to support locally hosted huggingface models (see below)
* models served locally using Huggingfacen's TGI LLM inference server running in docker. 

See the Makefile for details of the relevant ports.

### Custom version of gripetape.
Griptape.ai is an enterprise agent framework that is rapidly developing. The customized version below implements Huggingface's InferenceClient which is superceeding their InferenceAPI, and makes it possible to host the models locally. 
As at v0.21, gripetape has updated their model drivers with InferenceClient and this will remove the need to continne using the custom version once this api server has been updated too. 

Use:
```
git clone https://github.com/mattma1970/griptape/tree/yak_dev
pip install -e . 
```
This branch (at 2/11/23) contains the HF InferenceClient prompt driver for locally server HF models. 

### Model Inference

Model serving uses HuggingFace TGI. Using the docker to run TGI 
TGI Offical Docker installation [https://huggingface.co/docs/text-generation-inference/quicktour]
Note that model-id can be set to the path of a local copy of the model files which can be cloned from Huggingface using git lfs. 
This api has been tested on OpenOrca and Zephry, both 7B models running on a 24G consumer GPU card. 

While locally hosted models have been our target, it is easy to replace local model servering with a commercial api.
To do this, change the voice_chat/yak_agent/yak_agent.py package. Replace the definition of agent with the appropriate griptape driver. For example, 
```
            self.agent = Agent(prompt_driver=OpenAiChatPromptDriver(model='gpt-3.5-turbo', stream=self.stream),
                            logger_level=logging.ERROR,
                            rules= self.rules,
                            stream=self.stream,)
```

Will replace local model inference with gpt-3.5-turbo. (You'll need to set the OPENAI_API_KEY environment key in the .env file)

### Environment variables

The following environment variables need to be set in the .env file in the root. Environment variable loading is handled by dotenv

OPENAI_API_KEY (if using OpenAi models)
HUGGING_FACE_API_TOKEN  (if using HF models hosted by them)
assembly_ai_api (used for speech to text)
AZURE_SPEECH_SERVICES_KEY (text to speech)
AZURE_SPEECH_REGION  (text to speech)

Note: assembly.ai (STT) if required here as a temporary token for use on the front end needs to be generated.

``` 
# from root directory 
source .venv/bin/activate
make run_tgi  # this will launch TGI and expose port 8080 for model serving.
```

In a seperate terminal run
```
make run_api
```

