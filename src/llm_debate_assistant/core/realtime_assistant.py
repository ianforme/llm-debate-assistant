import base64
import json
import queue
import socket
import threading
import time

import pyaudio
import socks
import websocket

from llm_debate_assistant.config import app_config


class RealtimeAssistant:
    def __init__(self, ws_url="wss://api.openai.com/v1/realtime?model=gpt-realtime"):
        self.temperature = app_config.realtime_config.temperature
        self.max_response_token = app_config.realtime_config.max_response_token

        # Set up SOCKS5 proxy
        socket.socket = socks.socksocket
        self.WS_URL = ws_url
        self.API_KEY = app_config.api_keys.openai_api_key

        self.audio_buffer = bytearray()
        self.mic_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.CHUNK_SIZE = app_config.realtime_config.CHUNK_SIZE
        self.RATE = app_config.realtime_config.RATE
        self.FORMAT = app_config.realtime_config.FORMAT

        self.mic_active = None

        self.is_playing = False
        self.assistant_speeches = []
        self.human_speeches = []

        self.user_session_total_time = 0
        self.user_session_start_time = None

    def clear_audio_buffer(self):
        self.audio_buffer = bytearray()
        print("🔵 Audio buffer cleared.")

    # Function to stop audio playback
    def stop_audio_playback(self):
        self.is_playing = False
        print("🔵 Stopping audio playback.")

    # Function to handle microphone input and put it into a queue
    def mic_callback(self, in_data, frame_count, time_info, status):
        if not self.mic_active:
            print("🎙️🟢 Mic active")
            self.mic_active = True
        self.mic_queue.put(in_data)

        return (None, pyaudio.paContinue)

    # Function to send microphone audio data to the WebSocket
    def send_mic_audio_to_websocket(self, ws):
        try:
            while not self.stop_event.is_set():
                if not self.mic_queue.empty():
                    mic_chunk = self.mic_queue.get()
                    encoded_chunk = base64.b64encode(mic_chunk).decode("utf-8")
                    message = json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": encoded_chunk,
                        }
                    )
                    try:
                        ws.send(message)
                    except Exception as e:
                        print(f"Error sending mic audio: {e}")
        except Exception as e:
            print(f"Exception in send_mic_audio_to_websocket thread: {e}")
        finally:
            print("Exiting send_mic_audio_to_websocket thread.")

    # Function to handle audio playback callback
    def speaker_callback(self, in_data, frame_count, time_info, status):
        bytes_needed = frame_count * 2
        current_buffer_size = len(self.audio_buffer)

        if current_buffer_size >= bytes_needed:
            audio_chunk = bytes(self.audio_buffer[:bytes_needed])
            self.audio_buffer = self.audio_buffer[bytes_needed:]
        else:
            audio_chunk = bytes(self.audio_buffer) + b"\x00" * (bytes_needed - current_buffer_size)
            self.audio_buffer.clear()

        return (audio_chunk, pyaudio.paContinue)

    # Function to receive audio data from the WebSocket and process events
    def receive_audio_from_websocket(
        self, ws, instruction, user_time_in_seconds, ai_start_first=True
    ):
        try:
            while not self.stop_event.is_set():
                try:
                    if (user_time_in_seconds is None) or (
                        self.user_session_total_time <= user_time_in_seconds
                    ):
                        message = ws.recv()
                        if not message:  # Handle empty message (EOF or connection close)
                            print("🔵 Received empty message (possibly EOF or WebSocket closing).")
                            break

                        # Now handle valid JSON messages only
                        message = json.loads(message)
                        event_type = message["type"]
                        print(f"⚡️ Received WebSocket event: {event_type}")

                        if event_type == "session.created":
                            self.send_fc_session_update(ws)

                            # if AI needs to speak first
                            self.start_conversation(ws, instruction, ai_start_first)

                        elif event_type == "error":
                            print(message["error"])

                        elif event_type == "response.audio.delta":
                            audio_content = base64.b64decode(message["delta"])
                            self.audio_buffer.extend(audio_content)
                            # print(f'🔵 Received {len(audio_content)} bytes, total buffer size: {len(self.audio_buffer)}')

                        elif event_type == "input_audio_buffer.speech_started":
                            print("🔵 Speech started, clearing buffer and stopping playback.")
                            self.clear_audio_buffer()
                            self.stop_audio_playback()
                            self.user_session_start_time = time.time()

                        elif event_type == "input_audio_buffer.speech_stopped":
                            self.create_responses_with_speech_history(ws, instruction)
                            speech_time = round(time.time() - self.user_session_start_time, 0)
                            self.user_session_total_time += speech_time
                            print(
                                f"🔵 Speech stopped, creating new responses; Speech time: {speech_time} seconds; Total time: {self.user_session_total_time}"
                            )

                        elif event_type == "response.audio.done":
                            print("🔵 AI finished speaking.")

                        elif event_type == "conversation.item.input_audio_transcription.completed":
                            print(f"User Input: {message['transcript']}")
                            self.human_speeches.append("用户:\n" + message["transcript"])

                        elif event_type == "response.audio_transcript.done":
                            print(f"AI output: {message['transcript']}")
                            self.assistant_speeches.append("AI助手:\n" + message["transcript"])
                    else:
                        print("Total user time is up. Exiting..")
                        break

                except Exception as e:
                    print(f"Error receiving audio: {e}")
        except Exception as e:
            print(f"Exception in receive_audio_from_websocket thread: {e}")
        finally:
            print("Exiting receive_audio_from_websocket thread.")
            self.stop_event.set()

    def start_conversation(self, ws, instruction, ai_start_first):
        if ai_start_first:
            start_queue = "直接开始发言。"
        else:
            start_queue = "邀请用户开始质询"

        start_prompt = (
            instruction.strip()
            + f"\n\n【以上是本次辩论的发言背景和规则要求, 请你基于以上信息，{start_queue}】"
        )

        response_create = {
            "type": "response.create",
            "response": {"modalities": ["audio", "text"], "instructions": start_prompt},
        }

        try:
            ws.send(json.dumps(response_create))
            print("✅ response.create 已发送（已要求模型基于背景开始发言）。")
        except Exception as e:
            print(f"Failed to send response.create: {e}")

    def create_responses_with_speech_history(self, ws, context_instruction):
        # assume assistant starts first
        speech_history = [
            x for pair in zip(self.assistant_speeches, self.human_speeches) for x in pair
        ]
        response_create = {
            "type": "response.create",
            "response": {
                "modalities": ["audio", "text"],
                "instructions": context_instruction
                + "\n\n"
                + f"【当前环节双方发言】\n{speech_history}",
            },
        }

        ws.send(json.dumps(response_create))
        print("✅ response.create 已发送（基于本环节双方发言）。")

    # Function to send session configuration updates to the server
    def send_fc_session_update(self, ws):
        """
        - base_system_instruction -> session.instructions（稳定的系统设定）
        - context_instruction     -> response.create.instructions（本轮上下文/背景/任务）
        """
        # ① 系统层（稳定设定）：放角色、风格、语言等“长期有效”的规则
        base_system_instruction = """
    【你是谁】
    - 你是一名专业的辩手，你正在参加一场中文辩论比赛，你即将和对方辩友展开交锋。
    """

        # ② 发送 session.update：把系统层设定写入 instructions
        session_config = {
            "type": "session.update",
            "session": {
                # 仅放“长期规则”，不放大段背景
                "instructions": base_system_instruction,
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.7,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                    "create_response": False,
                },
                "voice": "alloy",
                "temperature": self.temperature,
                "max_response_output_tokens": self.max_response_token,
                "modalities": ["text", "audio"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1", "language": "zh"},
            },
        }

        try:
            ws.send(json.dumps(session_config))
            print("✅ session.update 已发送（系统设定生效）。")
        except Exception as e:
            print(f"Failed to send session update: {e}")

    # Function to create a WebSocket connection using IPv4
    def create_connection_with_ipv4(self, *args, **kwargs):
        # Enforce the use of IPv4
        original_getaddrinfo = socket.getaddrinfo

        def getaddrinfo_ipv4(host, port, family=socket.AF_INET, *args):
            return original_getaddrinfo(host, port, socket.AF_INET, *args)

        socket.getaddrinfo = getaddrinfo_ipv4
        try:
            return websocket.create_connection(*args, **kwargs)
        finally:
            # Restore the original getaddrinfo method after the connection
            socket.getaddrinfo = original_getaddrinfo

    # Function to establish connection with OpenAI's WebSocket API
    def connect_to_openai(self, instruction, user_time_in_seconds, ai_start_first=True):
        ws = None
        try:
            ws = self.create_connection_with_ipv4(
                self.WS_URL,
                header=[
                    f"Authorization: Bearer {self.API_KEY}",
                    "OpenAI-Beta: realtime=v1",
                ],
            )
            print("Connected to OpenAI WebSocket.")

            # Start the recv and send threads
            receive_thread = threading.Thread(
                target=self.receive_audio_from_websocket,
                args=(
                    ws,
                    instruction,
                    user_time_in_seconds,
                    ai_start_first,
                ),
            )
            receive_thread.start()

            mic_thread = threading.Thread(target=self.send_mic_audio_to_websocket, args=(ws,))
            mic_thread.start()

            # Wait for stop_event to be set
            while not self.stop_event.is_set():
                time.sleep(0.1)

            # Send a close frame and close the WebSocket gracefully
            print("Sending WebSocket close frame.")
            ws.send_close()

            receive_thread.join()
            mic_thread.join()

            print("WebSocket closed and threads terminated.")
        except Exception as e:
            print(f"Failed to connect to OpenAI: {e}")
        finally:
            if ws is not None:
                try:
                    ws.close()
                    print("WebSocket connection closed.")
                except Exception as e:
                    print(f"Error closing WebSocket connection: {e}")

    def run(self, instruction, user_time_in_seconds, ai_start_first=True):
        p = pyaudio.PyAudio()

        mic_stream = p.open(
            format=self.FORMAT,
            channels=1,
            rate=self.RATE,
            input=True,
            stream_callback=self.mic_callback,
            frames_per_buffer=self.CHUNK_SIZE,
        )

        speaker_stream = p.open(
            format=self.FORMAT,
            channels=1,
            rate=self.RATE,
            output=True,
            stream_callback=self.speaker_callback,
            frames_per_buffer=self.CHUNK_SIZE,
        )

        try:
            mic_stream.start_stream()
            speaker_stream.start_stream()

            self.connect_to_openai(instruction, user_time_in_seconds, ai_start_first=ai_start_first)

        except KeyboardInterrupt:
            print("Gracefully shutting down...")
            self.stop_event.set()

        finally:
            # get the conversation history as output
            convo_history = [
                x for pair in zip(self.assistant_speeches, self.human_speeches) for x in pair
            ]
            mic_stream.stop_stream()
            mic_stream.close()
            speaker_stream.stop_stream()
            speaker_stream.close()

            # reset status
            self.audio_buffer = bytearray()
            self.mic_queue = queue.Queue()
            self.stop_event = threading.Event()

            self.mic_active = None

            self.is_playing = False
            self.assistant_speeches = []
            self.human_speeches = []

            self.user_session_total_time = 0
            self.user_session_start_time = None

            p.terminate()
            print("Audio streams stopped and resources released. Exiting.")

            return convo_history
