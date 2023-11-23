"""
Voice interface for chatting with Llama2 installed on the local machine only. 
For voice interface on LLama2 hosted on a remote server, see local_web_chat_client.py
(c) mattma1970@gmail.com

"""

import websockets
import asyncio
import base64
import json
import os
from dotenv import load_dotenv

import streamlit as st
import requests
import re
import argparse
from typing import List, Union

from data_classes.chat_data_classes import SessionStart, ApiUserMessage

import pyaudio

load_dotenv()

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()

# start recorder

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER,
)

if "run" not in st.session_state:
    st.session_state["run"] = False


def start_listening():
    st.session_state["run"] = True


def stop_listening():
    st.session_state["run"] = False


st.title("YakChat")
start, stop = st.columns(2)

start.button("Start", on_click=start_listening)
stop.button("Stop", on_click=stop_listening)

URL = f"wss://api.assemblyai.com/v2/realtime/ws?sample_rate={RATE}"


def create_agent(args) -> str:
    if "session_id" not in st.session_state:
        payload = SessionStart(
            cafe_id=args.cafe_id,
            rule_names={
                "agent": args.agent_rules,
                "restaurant": args.restaurant_rules,
            },
            stream=args.stream,
        )
        response = requests.get(
            url=f'{args.llm_endpoint.strip("/")}/create_agent_session',
            json=payload.dict(),
        )
        if response.status_code == 200:
            session_id = json.loads(response.content)["session_id"]
            st.session_state["session_id"] = session_id
            print(session_id)
        else:
            print(f"Failed to create new agent: Status Code: {response.status_code}")


async def send_receive(args):
    """
    Async send and recieve with speech to text
    """
    print(f"Connecting websocket to url ${URL}")

    async with websockets.connect(
        URL,
        extra_headers=(("Authorization", os.environ["assembly_ai_api"]),),
        ping_interval=5,
        ping_timeout=20,
    ) as _ws:
        r = await asyncio.sleep(0.1)
        print("Receiving SessionBegins ...")

        session_begins = await _ws.recv()
        print(session_begins)
        print("Sending messages ...")

        async def send(args):
            while st.session_state["run"]:
                try:
                    data = stream.read(FRAMES_PER_BUFFER)
                    data = base64.b64encode(data).decode("utf-8")
                    json_data = json.dumps({"audio_data": str(data)})
                    r = await _ws.send(json_data)

                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    assert e.code == 4008
                    break

                except Exception as e:
                    assert False, "Not a websocket 4008 error"

                r = await asyncio.sleep(0.01)

            return True

        async def receive(args):
            while "run" in st.session_state and st.session_state["run"]:
                try:
                    result_str = await _ws.recv()
                    if (
                        json.loads(result_str)["message_type"] == "FinalTranscript"
                        and json.loads(result_str)["text"] != ""
                    ):
                        st.markdown(json.loads(result_str)["text"] + "*")
                        st.markdown("sending to chatbot...")
                        user_input = ApiUserMessage(
                            user_input=f"{json.loads(result_str)['text']}",
                            session_id=st.session_state.session_id,
                            user_id=None,
                        )
                        r = requests.post(
                            url=f'{args.llm_endpoint.strip("/")}/{args.task}',
                            json=user_input.dict(),
                            stream=args.stream,
                        )

                        if args.stream:
                            streaming_text = st.empty()
                            message = ""
                            for chunk in r:
                                message += chunk.decode("utf-8")
                                streaming_text.text(message)
                        else:
                            data = r.json()
                            chat_response = data["data"]
                            st.markdown(chat_response)

                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    assert e.code == 4008
                    break

                except Exception as e:
                    assert False, "Not a websocket 4008 error"

        send_result, receive_result = await asyncio.gather(send(args), receive(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm_endpoint",
        type=str,
        default="http://localhost:8884/",
        help="URL for REST API serving LLM",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="chat_with_agent",
        help="Task to perform, also name of endpoint",
    )
    parser.add_argument("--cafe_id", type=str, default="Twist", help="UID for cafe.")
    parser.add_argument(
        "--agent_rules",
        type=str,
        default="average_agent",
        help="rule set on server side for configuring agent personality",
    )
    parser.add_argument(
        "--restaurant_rules",
        type=str,
        default="json_menu",
        help="rule that holds menu and menu rules",
    )
    parser.add_argument(
        "--stream",
        type=bool,
        default=True,
        help="Whether responses from api should be streamed back.",
    )

    args = parser.parse_args()
    print("Creating agent .. ", create_agent(args))
    asyncio.run(send_receive(args))
