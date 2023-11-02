import requests

data = {"user_input":"hello","session_id":'1'}

url_chat = 'http://localhost:8888/chat_with_agent'
with requests.post(url_chat,json=data, stream=True) as response:
    for x in response.iter_content(chunk_size=None):
        print (x.decode('utf-8'), end="", flush=True)


""" url_chat = 'http://localhost:8888/chat_non_streaming'
response=requests.post(url_chat,json=data, stream=False)
print (response) """