import ggwave
import pyaudio
import ollama
import threading
import queue
import time
import json
import os
import re
import logging
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("audio_chat.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AudioChat")

@dataclass
class Message:
    role: str
    content: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class Config:
    sample_rate: int = 48000
    chunk_size: int = 1024
    max_payload_size: int = 140  # Match ggwave limit
    protocol_id: int = 2
    volume: int = 20
    reflex_model: str = "tinyllama"
    reasoning_model: str = "deepseek-r1:7b"
    conversational_model: str = "llama3.2"
    history_file: str = "databank.json"
    chunk_delay: float = 0.02
    timeout: float = 1.0
    max_history: int = 10
    assistant_name: str = "LLaMA"
    reasoning_delay: float = 0.5  # Delay before reasoning starts
    complexity_threshold: float = 0.5

class AudioChatSystem:
    def __init__(self, config: Config):
        self.config = config
        self.receive_queue = queue.Queue()
        self.send_queue = queue.Queue()
        self.reasoning_queue = queue.Queue()
        self.history: List[Message] = []
        self.shutdown_event = threading.Event()
        self.history_lock = threading.Lock()
        self.reasoning_result = None
        self.reasoning_ready = threading.Event()
        
        self.p = pyaudio.PyAudio()
        self.ggwave_instance = ggwave.init()
        
        self._load_history()
        
    def _load_history(self) -> None:
        try:
            if os.path.exists(self.config.history_file):
                with open(self.config.history_file, "r", encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        raw_history = json.loads(content)
                        self.history = [Message(**msg) for msg in raw_history]
                        logger.info(f"Loaded {len(self.history)} messages from history")
                    else:
                        logger.info("History file is empty")
            else:
                logger.info("No history file found")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse history: {e}")
            self.history = []
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            self.history = []

    def _save_history(self) -> None:
        with self.history_lock:
            try:
                with open(self.config.history_file, "w", encoding='utf-8') as f:
                    json.dump([asdict(msg) for msg in self.history], f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(self.history)} messages to history")
            except Exception as e:
                logger.error(f"Failed to save history: {e}")

    def receiver_thread(self) -> None:
        logger.info("Receiver thread started")
        try:
            input_stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
            while not self.shutdown_event.is_set():
                try:
                    data = input_stream.read(self.config.chunk_size, exception_on_overflow=False)
                    res = ggwave.decode(self.ggwave_instance, data)
                    if res is not None:
                        received_text = res.decode("utf-8")
                        logger.info(f"Received signal: {received_text}")
                        self.receive_queue.put(received_text)
                except Exception as e:
                    logger.error(f"Error in receiver: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize input stream: {e}")
        finally:
            if 'input_stream' in locals():
                input_stream.stop_stream()
                input_stream.close()
            logger.info("Receiver thread stopped")

    def reflex_thread(self) -> None:
        logger.info("Reflex thread started")
        while not self.shutdown_event.is_set():
            try:
                received_text = self.receive_queue.get(timeout=self.config.timeout)
                logger.info(f"Reflex processing: {received_text}")
                
                if received_text.lower().strip() == "clear history":
                    with self.history_lock:
                        self.history = []
                    self._save_history()
                    self.send_queue.put("History cleared!")
                    self.receive_queue.task_done()
                    continue
                
                with self.history_lock:
                    self.history.append(Message(role="user", content=received_text))
                
                reflex_messages = self._build_reflex_prompt(received_text)
                reflex_response = ollama.chat(model=self.config.reflex_model, messages=reflex_messages)
                reflex_text = reflex_response["message"]["content"].strip()
                logger.info(f"Reflex response: {reflex_text}")
                
                route = self._route_conversation(received_text)
                
                if route == "reflex_only":
                    self._send_chunks(reflex_text)
                    with self.history_lock:
                        self.history.append(Message(role="assistant", content=reflex_text))
                else:
                    self._send_chunks(reflex_text)  # Send initial acknowledgment
                    with self.history_lock:
                        self.history.append(Message(role="assistant", content=reflex_text))
                    
                    if route == "conversational":
                        threading.Thread(target=self._generate_conversational_response, 
                                       args=(received_text, reflex_text), daemon=True).start()
                    elif route == "reasoning":
                        self.reasoning_queue.put((received_text, reflex_text))
                        if not self.reasoning_ready.is_set():
                            threading.Thread(target=self._reasoning_thread, daemon=True).start()
                
                self.receive_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in reflex thread: {e}")
                self.send_queue.put("Oops, hang on!")

    def _reasoning_thread(self) -> None:
        logger.info("Reasoning thread started")
        self.reasoning_ready.set()
        while not self.shutdown_event.is_set():
            try:
                received_text, reflex_text = self.reasoning_queue.get(timeout=self.config.timeout)
                reasoning_messages = self._build_reasoning_prompt(received_text)
                reasoning_response = ollama.chat(model=self.config.reasoning_model, messages=reasoning_messages)
                self.reasoning_result = reasoning_response["message"]["content"]
                logger.info(f"Reasoning monologue: {self.reasoning_result}")
                
                # Pass reasoning result to conversational model
                threading.Thread(target=self._generate_conversational_response,
                               args=(received_text, reflex_text, self.reasoning_result), daemon=True).start()
                self.reasoning_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in reasoning thread: {e}")

    def _route_conversation(self, input_text: str) -> str:
        input_text = input_text.lower().strip()
        basic_inputs = {"hey", "hi", "hello", "yo"}
        
        if input_text in basic_inputs:
            return "reflex_only"
        
        # Check for tasks requiring reasoning or complex responses
        if any(keyword in input_text for keyword in ["code", "write", "how", "why", "explain"]):
            return "reasoning"
        
        # Default to conversational for anything else
        return "conversational"

    def _build_reflex_prompt(self, current_input: str) -> List[Dict[str, str]]:
        system_prompt = (
            "You are a friendly assistant. Give a short, casual acknowledgment (max 10 words) to the user’s input. "
            "For anything beyond greetings, say 'Let me check that!'"
        )
        messages = [{"role": "system", "content": system_prompt}]
        with self.history_lock:
            start_idx = max(0, len(self.history) - self.config.max_history - 1)
            recent_history = self.history[start_idx:-1]
            for msg in recent_history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": current_input})
        return messages

    def _build_reasoning_prompt(self, current_input: str) -> List[Dict[str, str]]:
        system_prompt = (
            "You are a reasoning assistant. Analyze the user’s input and provide a concise internal monologue "
            "on how to respond helpfully. For code-related tasks, outline the solution approach."
        )
        messages = [{"role": "system", "content": system_prompt}]
        with self.history_lock:
            start_idx = max(0, len(self.history) - self.config.max_history)
            recent_history = self.history[start_idx:]
            for msg in recent_history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": current_input})
        return messages

    def _send_chunks(self, text: str) -> None:
        buffer = text
        while buffer:
            buffer_bytes = buffer.encode("utf-8")
            if len(buffer_bytes) > self.config.max_payload_size:
                last_space = buffer.rfind(" ", 0, self.config.max_payload_size)
                if last_space == -1:
                    last_space = self.config.max_payload_size
                chunk = buffer[:last_space]
                buffer = buffer[last_space:].strip()
            else:
                chunk = buffer
                buffer = ""
            logger.info(f"Transmitting chunk: {chunk}")
            self.send_queue.put(chunk)

    def _generate_conversational_response(self, received_text: str, reflex_text: str, reasoning_text: Optional[str] = None) -> None:
        conversational_messages = self._build_conversational_prompt(received_text, reflex_text, reasoning_text)
        response_stream = ollama.chat(model=self.config.conversational_model, messages=conversational_messages, stream=True)
        
        full_response = ""
        buffer = ""
        
        for chunk in response_stream:
            chunk_text = chunk["message"]["content"]
            full_response += chunk_text
            buffer += chunk_text
            
            buffer_bytes = buffer.encode("utf-8")
            if len(buffer_bytes) >= self.config.max_payload_size - 10:
                last_space = buffer.rfind(" ")
                if last_space != -1 and last_space > 0:
                    send_text = buffer[:last_space]
                    buffer = buffer[last_space + 1:]
                else:
                    send_text = buffer
                    buffer = ""
                logger.info(f"Streaming chunk to send: {send_text}")
                self.send_queue.put(send_text)
        
        if buffer:
            logger.info(f"Streaming final chunk to send: {buffer}")
            self.send_queue.put(buffer)
        
        clean_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
        name_pattern = rf"^{re.escape(self.config.assistant_name)}:\s+"
        clean_response = re.sub(name_pattern, "", clean_response)
        logger.info(f"Full conversational response: {clean_response}")
        
        with self.history_lock:
            if len(self.history) > 1 and self.history[-1].role == "assistant":
                self.history[-1] = Message(role="assistant", content=clean_response)
            else:
                self.history.append(Message(role="assistant", content=clean_response))
        self._save_history()

    def _build_conversational_prompt(self, current_input: str, reflex_text: str, reasoning_text: Optional[str]) -> List[Dict[str, str]]:
        reasoning_part = f"Use this reasoning: '{reasoning_text}'." if reasoning_text else "Respond naturally."
        system_prompt = (
            f"You are {self.config.assistant_name}, a helpful and friendly AI assistant. "
            f"The reflex response was '{reflex_text}'. Build on it with a thoughtful, casual continuation. "
            f"{reasoning_part} For code requests, provide the actual code or explanation."
        )
        messages = [{"role": "system", "content": system_prompt}]
        with self.history_lock:
            start_idx = max(0, len(self.history) - self.config.max_history)
            recent_history = self.history[start_idx:]
            for msg in recent_history:
                messages.append({"role": msg.role, "content": msg.content})
        return messages

    def sender_thread(self) -> None:
        logger.info("Sender thread started")
        try:
            output_stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.config.sample_rate,
                output=True,
                frames_per_buffer=4096
            )
            while not self.shutdown_event.is_set():
                try:
                    droid_response = self.send_queue.get(timeout=self.config.timeout)
                    waveform = ggwave.encode(
                        droid_response,
                        protocolId=self.config.protocol_id,
                        volume=self.config.volume
                    )
                    output_stream.write(waveform, len(waveform) // 4)
                    time.sleep(self.config.chunk_delay)
                    self.send_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in sender: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize output stream: {e}")
        finally:
            if 'output_stream' in locals():
                output_stream.stop_stream()
                output_stream.close()
            logger.info("Sender thread stopped")

    def start(self) -> None:
        threads = [
            threading.Thread(target=self.receiver_thread, daemon=True),
            threading.Thread(target=self.reflex_thread, daemon=True),
            threading.Thread(target=self.sender_thread, daemon=True)
        ]
        for thread in threads:
            thread.start()
        
        logger.info("Audio Chat System started. Press Ctrl+C to stop.")
        try:
            while all(thread.is_alive() for thread in threads):
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown initiated...")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        logger.info("Shutting down...")
        self.shutdown_event.set()
        time.sleep(1)
        ggwave.free(self.ggwave_instance)
        self.p.terminate()
        logger.info("Shutdown complete")

def parse_args():
    parser = argparse.ArgumentParser(description="Audio Communication System with Layered Thinking")
    parser.add_argument("--reflex-model", type=str, default="tinyllama", help="Ollama reflex model")
    parser.add_argument("--reasoning-model", type=str, default="deepseek-r1:7b", help="Ollama reasoning model")
    parser.add_argument("--conversational-model", type=str, default="llama3.2", help="Ollama conversational model")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Audio sample rate")
    parser.add_argument("--history", type=str, default="conversation_history.json", help="History file path")
    parser.add_argument("--volume", type=int, default=20, help="Audio transmission volume")
    parser.add_argument("--max-history", type=int, default=10, help="Max messages in context")
    parser.add_argument("--name", type=str, default="LLaMA", help="Assistant name")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--clear-history", action="store_true", help="Clear history on startup")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    config = Config(
        sample_rate=args.sample_rate,
        reflex_model=args.reflex_model,
        reasoning_model=args.reasoning_model,
        conversational_model=args.conversational_model,
        history_file=args.history,
        volume=args.volume,
        max_history=args.max_history,
        assistant_name=args.name
    )
    
    system = AudioChatSystem(config)
    
    if args.clear_history and os.path.exists(config.history_file):
        try:
            os.remove(config.history_file)
            logger.info(f"Cleared conversation history file: {config.history_file}")
            system.history = []
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
    
    system.start()

if __name__ == "__main__":
    main()