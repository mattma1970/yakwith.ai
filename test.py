import requests

from pydantic import BaseModel
from typing import Optional
import json
import time


url = 'http://localhost:8884/create_agent_session'
response = requests.get(url)
session=response.content.decode('utf-8')
session = json.loads(session)
session_id=session['session_id']
data = {"user_input":"hello","session_id":'1'}
data2 = {"user_input":"what is the time in Turkey?","session_id":'2'}
#data
print(data)

url_chat = 'http://localhost:8884/chat_with_agent'
with requests.post(url_chat, json=data, stream=True) as response:
    for x in response.iter_content(chunk_size=None):
        print (x.decode('utf-8'), end="", flush=True)

print(' ********************* DONE ******************************')


""" url = 'http://localhost:8884/create_agent_session'
response = requests.get(url)
session=response.content.decode('utf-8')
session = json.loads(session)
session_id=session['session_id']
data = {"user_input":"hello","session_id":session_id}

print(' ********************* DONE ******************************') """

url_chat = 'http://localhost:8884/chat_with_agent'
with requests.post(url_chat, json=data2, stream=True) as response2:
    for x in response2.iter_content(chunk_size=None):
        print (x.decode('utf-8'), end="", flush=True)