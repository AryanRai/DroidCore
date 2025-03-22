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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("audio_chat.log"),
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
    max_payload_size: int = 128
    protocol_id: int = 2
    volume: int = 20
    reasoning_model: str = "llama3.2"  # Model for reasoning
    conversational_model: str = "deepseek-r1:1.5b"  # Changed to LLaMA 3.2 3B
    history_file: str = "databank.json"
    chunk_delay: float = 0.05
    timeout: float = 1.0
    max_history: int = 10
    assistant_name: str = "LLaMA"  # Changed to reflect LLaMA
    
class AudioChatSystem:
    def __init__(self, config: Config):
        self.config = config
        self.receive_queue = queue.Queue()
        self.send_queue = queue.Queue()
        self.history: List[Message] = []
        self.shutdown_event = threading.Event()
        
        # Initialize audio
        self.p = pyaudio.PyAudio()
        self.ggwave_instance = ggwave.init()
        
        # Load history if it exists
        self._load_history()
        
    def _load_history(self) -> None:
        try:
            if os.path.exists(self.config.history_file):
                with open(self.config.history_file, "r") as f:
                    raw_history = json.load(f)
                    self.history = [Message(**msg) for msg in raw_history]
                logger.info(f"Loaded {len(self.history)} messages from history")
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            self.history = []
    
    def _save_history(self) -> None:
        try:
            with open(self.config.history_file, "w") as f:
                json.dump([asdict(msg) for msg in self.history], f, indent=2)
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
    
    def processor_thread(self) -> None:
        logger.info("Processor thread started")
        
        while not self.shutdown_event.is_set():
            try:
                received_text = self.receive_queue.get(timeout=self.config.timeout)
                logger.info(f"Processing: {received_text}")
                
                # Handle special commands
                if received_text.lower().strip() == "clear history":
                    self.history = []
                    self._save_history()
                    self.send_queue.put("Conversation history cleared.")
                    self.receive_queue.task_done()
                    continue
                
                # Add user message to history
                self.history.append(Message(role="user", content=received_text))
                
                # Step 1: Reasoning with DeepSeek
                reasoning_messages = self._build_reasoning_prompt(received_text)
                reasoning_response = ollama.chat(model=self.config.reasoning_model, messages=reasoning_messages)
                reasoning_text = reasoning_response["message"]["content"]
                logger.info(f"Reasoning monologue: {reasoning_text}")
                
                # Step 2: Generate conversational response with LLaMA 3.2
                conversational_messages = self._build_conversational_prompt(received_text, reasoning_text)
                response = ollama.chat(model=self.config.conversational_model, messages=conversational_messages)
                raw_response = response["message"]["content"]
                
                # Process response (remove thinking blocks if present)
                think_pattern = r"<think>.*?</think>"
                think_content = "\n".join(re.findall(think_pattern, raw_response, re.DOTALL))
                if think_content:
                    logger.info(f"Conversational think monologue: {think_content}")
                droid_response = re.sub(think_pattern, "", raw_response, flags=re.DOTALL).strip()
                
                # Remove assistant name prefix if present
                name_pattern = rf"^{re.escape(self.config.assistant_name)}:\s+"
                droid_response = re.sub(name_pattern, "", droid_response)
                
                logger.info(f"Generated response: {droid_response}")
                
                # Add assistant response to history
                self.history.append(Message(role="assistant", content=droid_response))
                
                # Save history
                self._save_history()
                
                # Queue for sending
                self.send_queue.put(droid_response)
                
                self.receive_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processor: {e}")
                self.send_queue.put("Sorry, I’m having trouble right now. Please try again.")
    
    def _build_reasoning_prompt(self, current_input: str) -> List[Dict[str, str]]:
        system_prompt = (
            "You are a reasoning assistant. Analyze the user’s input and provide a concise internal monologue "
            "on how to respond helpfully and as a friend. Do not generate the final response, just the reasoning."
        )
        messages = [{"role": "system", "content": system_prompt}]
        
        start_idx = max(0, len(self.history) - self.config.max_history - 1)
        recent_history = self.history[start_idx:-1]
        for msg in recent_history:
            role = "user" if msg.role == "user" else "assistant"
            messages.append({"role": role, "content": msg.content})
        
        messages.append({"role": "user", "content": current_input})
        return messages
    
    def _build_conversational_prompt(self, current_input: str, reasoning_text: str) -> List[Dict[str, str]]:
        system_prompt = (
            f"You are {self.config.assistant_name}, a helpful and friendly AI assistant. "
            f"Use the following reasoning to craft a brief, conversational response: '{reasoning_text}'. "
            f"Respond directly as if in a casual chat with a friend, without repeating the reasoning."
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        start_idx = max(0, len(self.history) - self.config.max_history - 1)
        recent_history = self.history[start_idx:-1]
        for msg in recent_history:
            role = "user" if msg.role == "user" else "assistant"
            messages.append({"role": role, "content": msg.content})
        
        messages.append({"role": "user", "content": current_input})
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
                    logger.info(f"Preparing to transmit: {droid_response}")
                    
                    response_bytes = droid_response.encode("utf-8")
                    if len(response_bytes) > self.config.max_payload_size:
                        chunks = [response_bytes[i:i + self.config.max_payload_size] 
                                for i in range(0, len(response_bytes), self.config.max_payload_size)]
                    else:
                        chunks = [response_bytes]
                    
                    for i, chunk in enumerate(chunks):
                        chunk_text = chunk.decode("utf-8", errors="ignore")
                        logger.info(f"Transmitting chunk {i+1}/{len(chunks)}: {chunk_text}")
                        waveform = ggwave.encode(
                            chunk_text, 
                            protocolId=self.config.protocol_id, 
                            volume=self.config.volume
                        )
                        output_stream.write(waveform, len(waveform) // 4)
                        time.sleep(self.config.chunk_delay)
                    
                    logger.info("Transmission complete")
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
            threading.Thread(target=self.processor_thread, daemon=True),
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
    parser = argparse.ArgumentParser(description="Audio Communication System with Hybrid Model")
    parser.add_argument("--reasoning-model", type=str, default="deepseek-r1:1.5b",
                        help="Ollama reasoning model (default: deepseek-r1:1.5b)")
    parser.add_argument("--conversational-model", type=str, default="llama3.2",
                        help="Ollama conversational model (default: llama3.2:3b)")
    parser.add_argument("--sample-rate", type=int, default=48000,
                        help="Audio sample rate (default: 48000)")
    parser.add_argument("--history", type=str, default="conversation_history.json",
                        help="History file path (default: conversation_history.json)")
    parser.add_argument("--volume", type=int, default=20,
                        help="Audio transmission volume (default: 20)")
    parser.add_argument("--max-history", type=int, default=10,
                        help="Maximum number of messages to keep in context (default: 10)")
    parser.add_argument("--name", type=str, default="LLaMA",
                        help="Name for the conversational assistant (default: LLaMA)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--clear-history", action="store_true",
                        help="Clear conversation history on startup")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    config = Config(
        sample_rate=args.sample_rate,
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