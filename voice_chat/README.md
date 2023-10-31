** A streamlit app for voice chat with an LLM **

Uses webRTC custom component for streamlit [https://github.com/whitphx/streamlit-webrtc] for sending voice to the server where Assembly.ai is used for STT. This is then used to chat with the LLM. 

Designed to work with HF models served by TGI. 

** Installation **
Requires custom verison of griptape.ai from https://github.com/mattma1970/griptape/tree/main. This fork contains HuggingFaceInferenceClientPromptDriver which can be used with HF models served from arbitrary endpoints. 



