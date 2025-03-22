import os
import time
import queue
import threading
import numpy as np
import pyaudio
import whisper
import webrtcvad
import collections
import ggwave
import ollama
import logging
import argparse
import torch
import wave  # Add missing wave import
import psutil  # For system performance metrics
from datetime import datetime
from TTS.api import TTS
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("combined_system.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CombinedSystem")

# Create a separate performance logger
perf_logger = logging.getLogger("PerformanceMetrics")
perf_logger.setLevel(logging.INFO)
perf_handler = logging.FileHandler("performance_metrics.log", encoding='utf-8')
perf_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
perf_logger.addHandler(perf_handler)

@dataclass
class SystemConfig:
    # Speech recognition settings
    vad_mode: int = 1
    whisper_model: str = "base"  # Base model for speech recognition
    chunk_duration_ms: int = 30 
    silence_threshold: int = 2 
    max_speech_chunks: int = 100
    min_speech_chunks: int = 10
    
    # Audio settings
    channels: int = 1
    rate: int = 16000
    sample_rate: int = 48000
    chunk_size: int = 1024
    
    # Droid communication settings
    max_payload_size: int = 140
    protocol_id: int = 2
    volume: int = 20
    
    # Language model settings
    reflex_model: str = "gemma3:1b"        # Smaller, faster model for quick responses
    reasoning_model: str = "deepseek-r1:7b"     # Medium-sized reasoning model
    conversational_model: str = "phi3:mini" # Smaller model for standard responses
    complex_model: str = "llama3.2"        # Larger model for complex queries
    
    skip_reasoning: bool = True            # Skip reasoning for simple queries
    parallel_inference: bool = True        # Run models in parallel when possible
    max_processing_time: int = 30          # Maximum time to wait for model response in seconds
    
    # Message queue settings
    max_queue_size: int = 3                # Maximum messages to store in text queue
    drop_older_messages: bool = True       # Drop older messages when queue overflows
    
    # TTS settings
    tts_model: str = "tts_models/en/jenny/jenny"
    use_gpu: bool = True
    max_tts_length: int = 100             # Reduced for faster processing
    max_sentence_length: int = 50         # Maximum length per TTS chunk
    tts_batch_size: int = 3               # Process multiple sentences at once
    tts_worker_threads: int = 2           # Number of TTS worker threads
    
    # Queue settings
    max_concurrent_tts: int = 2           # Maximum concurrent TTS operations
    tts_queue_size: int = 5               # Maximum TTS queue size
    response_timeout: int = 15            # Maximum time to wait for responses
    
    # Performance optimization
    batch_size: int = 1                   # Batch size for whisper processing
    use_8bit: bool = True                 # Use 8-bit precision for faster inference
    
    # System settings
    output_mode: str = "both"  # "tts", "droid", or "both"
    history_file: str = "conversation_history.json"
    max_history: int = 10
    
    # Performance settings
    log_performance: bool = True
    perf_interval: int = 30  # seconds between performance logs
    
    # Cache settings
    use_cache: bool = True
    cache_dir: str = "model_cache"
    
    # Memory settings
    memory_file: str = "conversation_memory.json"
    max_memory_entries: int = 100
    memory_context_window: int = 5  # Number of recent memories to include in context
    episodic_memory: bool = True    # Enable episodic memory for better context
    semantic_memory: bool = True    # Enable semantic memory for concept understanding
    memory_refresh_interval: int = 300  # Seconds between memory consolidation
    multi_input: bool = True       # Enable multi-input (ggwave + speech)
    max_msg_length: int = 200       # Maximum message length for processing
    filler_response: bool = True   # Enable filler responses
    multi_response: bool = False    # Enable multi-response generation

class PerformanceTracker:
    """Tracks performance metrics and processing times"""
    
    def __init__(self, log_enabled=True):
        self.log_enabled = log_enabled
        self.metrics = {
            "audio_capture": [],
            "speech_recognition": [],
            "language_processing": [],
            "tts_generation": [],
            "output_processing": []
        }
        self.last_system_metrics_time = 0
    
    def start_timer(self, component):
        """Start timing a component"""
        if not self.log_enabled:
            return None
        return time.time()
    
    def end_timer(self, component, start_time):
        """End timing and log the duration"""
        if not self.log_enabled or start_time is None:
            return
        
        duration = time.time() - start_time
        self.metrics[component].append(duration)
        
        # Keep only the last 100 measurements
        if len(self.metrics[component]) > 100:
            self.metrics[component].pop(0)
        
        # Log individual timing for slow operations
        if (component == "speech_recognition" and duration > 1.0) or \
           (component == "language_processing" and duration > 2.0) or \
           (component == "tts_generation" and duration > 1.0):
            perf_logger.warning(f"{component} took {duration:.2f}s")
    
    def log_system_metrics(self, interval=30):
        """Log system-wide performance metrics periodically"""
        if not self.log_enabled:
            return
        
        current_time = time.time()
        if current_time - self.last_system_metrics_time < interval:
            return
        
        self.last_system_metrics_time = current_time
        
        # Get CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Get GPU utilization if available
        gpu_util = "N/A"
        try:
            if torch.cuda.is_available():
                # Note: This requires nvidia-smi to work
                gpu_util = f"{torch.cuda.utilization(0)}%"
        except:
            pass
        
        # Calculate average processing times
        avg_metrics = {}
        for component, times in self.metrics.items():
            if times:
                avg_metrics[component] = sum(times) / len(times)
            else:
                avg_metrics[component] = 0
        
        # Log the metrics
        perf_logger.info(f"SYSTEM: CPU: {cpu_percent:.1f}%, RAM: {memory.percent:.1f}%, GPU: {gpu_util}")
        perf_logger.info(f"AVG_TIMES: " + ", ".join([f"{c}: {t:.3f}s" for c, t in avg_metrics.items()]))
        
        # Clear metrics after logging
        for component in self.metrics:
            self.metrics[component] = []

class ModelCache:
    """Cache for model responses to avoid repeated queries"""
    
    def __init__(self, cache_dir="model_cache", max_size=100):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.cache = {}
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
    def get_key(self, model, messages):
        """Generate a cache key from model and messages"""
        message_str = "|".join([f"{m['role']}:{m['content'][:100]}" for m in messages])
        return f"{model}:{message_str}"
    
    def get(self, model, messages):
        """Get cached response if available"""
        key = self.get_key(model, messages)
        return self.cache.get(key)
    
    def put(self, model, messages, response):
        """Cache a response"""
        key = self.get_key(model, messages)
        self.cache[key] = response
        
        # Trim cache if it exceeds max size
        if len(self.cache) > self.max_size:
            # Remove oldest entries
            keys = list(self.cache.keys())
            for k in keys[:len(keys) - self.max_size]:
                del self.cache[k]

class ConversationMemory:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.episodic_memories = []
        self.semantic_memories = {}
        self.memory_lock = threading.Lock()
        self.last_consolidation = time.time()
        self._load_memories()
    
    def _load_memories(self):
        try:
            if os.path.exists(self.config.memory_file):
                with open(self.config.memory_file, "r", encoding='utf-8') as f:
                    memories = json.load(f)
                    self.episodic_memories = memories.get("episodic", [])
                    self.semantic_memories = memories.get("semantic", {})
                logger.info(f"Loaded {len(self.episodic_memories)} episodic and {len(self.semantic_memories)} semantic memories")
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
            self.episodic_memories = []
            self.semantic_memories = {}
    
    def save_memories(self):
        try:
            with open(self.config.memory_file, "w", encoding='utf-8') as f:
                json.dump({
                    "episodic": self.episodic_memories,
                    "semantic": self.semantic_memories
                }, f, indent=2, ensure_ascii=False)
            logger.info("Memories saved successfully")
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")

    def add_memory(self, text: str, role: str, importance: float = 0.5):
        with self.memory_lock:
            memory = {
                "text": text,
                "role": role,
                "timestamp": time.time(),
                "importance": importance
            }
            self.episodic_memories.append(memory)
            
            # Trim old memories if exceeding max size
            if len(self.episodic_memories) > self.config.max_memory_entries:
                # Keep most important memories
                self.episodic_memories.sort(key=lambda x: x["importance"], reverse=True)
                self.episodic_memories = self.episodic_memories[:self.config.max_memory_entries]
            
            # Periodically consolidate memories
            if time.time() - self.last_consolidation > self.config.memory_refresh_interval:
                self._consolidate_memories()
                self.last_consolidation = time.time()
    
    def _consolidate_memories(self):
        """Convert important episodic memories into semantic memories"""
        if not self.config.semantic_memory:
            return
            
        try:
            # Group related memories
            conversation_groups = []
            current_group = []
            
            for memory in self.episodic_memories:
                if not current_group or time.time() - memory["timestamp"] < 300:  # 5 minute window
                    current_group.append(memory)
                else:
                    if len(current_group) > 1:
                        conversation_groups.append(current_group)
                    current_group = [memory]
            
            if current_group:
                conversation_groups.append(current_group)
            
            # Extract key concepts from conversations
            for group in conversation_groups:
                conversation = " ".join(m["text"] for m in group)
                if len(conversation) > 50:  # Only process substantial conversations
                    # Extract concepts using the reflex model
                    concept_prompt = f"Extract 2-3 key concepts from this conversation: {conversation}"
                    concepts_response = ollama.chat(
                        model=self.config.reflex_model,
                        messages=[{"role": "user", "content": concept_prompt}]
                    )
                    concepts = concepts_response["message"]["content"].split("\n")
                    
                    # Store concepts in semantic memory
                    for concept in concepts:
                        if concept.strip():
                            self.semantic_memories[concept.strip()] = {
                                "last_updated": time.time(),
                                "frequency": self.semantic_memories.get(concept.strip(), {}).get("frequency", 0) + 1
                            }
            
            logger.info(f"Memory consolidation complete. {len(self.semantic_memories)} concepts stored")
            self.save_memories()
            
        except Exception as e:
            logger.error(f"Error in memory consolidation: {e}")
    
    def get_relevant_context(self, current_text: str) -> str:
        """Get relevant memories for the current conversation"""
        with self.memory_lock:
            context_parts = []
            
            # Add recent episodic memories
            recent_memories = sorted(
                self.episodic_memories[-self.config.memory_context_window:],
                key=lambda x: x["timestamp"]
            )
            if recent_memories:
                context_parts.append("Recent conversation:")
                for memory in recent_memories:
                    context_parts.append(f"{memory['role']}: {memory['text']}")
            
            # Add relevant semantic memories
            if self.semantic_memories:
                relevant_concepts = []
                for concept in self.semantic_memories:
                    if any(word.lower() in current_text.lower() for word in concept.split()):
                        relevant_concepts.append(
                            f"Concept: {concept} (mentioned {self.semantic_memories[concept]['frequency']} times)"
                        )
                if relevant_concepts:
                    context_parts.append("\nRelevant concepts:")
                    context_parts.extend(relevant_concepts[:3])  # Limit to top 3 concepts
            
            return "\n".join(context_parts)

class UnifiedAudioSystem:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.setup_queues()
        self.setup_audio()
        self.setup_models()
        self.setup_events()
        self.perf_tracker = PerformanceTracker(log_enabled=config.log_performance)
        self.memory_system = ConversationMemory(config)
        
        # Add queue stats tracking
        self.queue_stats_last_time = time.time()
        
        # Setup model cache if enabled
        if self.config.use_cache:
            self.model_cache = ModelCache(cache_dir=self.config.cache_dir)
        else:
            self.model_cache = None
        
        # Setup TTS worker pool if TTS is enabled
        self.tts_pool = None if config.output_mode not in ["tts", "both"] else TTSWorkerPool(config, self.device)

    def setup_queues(self):
        # Communication queues with size limits to prevent backlog
        self.audio_queue = queue.Queue(maxsize=3) # Reduced maximum
        self.text_queue = queue.Queue(maxsize=1)  # Only allow one text processing at a time
        self.response_queue = queue.Queue(maxsize=2)
        self.droid_queue = queue.Queue(maxsize=2)
        self.tts_queue = queue.Queue(maxsize=3)
        self.droid_input_queue = queue.Queue(maxsize=2)  # Queue for droid text input
        
        # Track active processing to allow cancellation
        self.processing_lock = threading.Lock()
        self.current_text_id = 0
        self.active_text_id = None
        
        # Add locks for thread safety
        self.queue_locks = {
            "audio": threading.Lock(),
            "text": threading.Lock(),
            "response": threading.Lock(),
            "droid": threading.Lock(),
            "tts": threading.Lock(),
            "droid_input": threading.Lock()
        }

    def setup_audio(self):
        # Initialize audio components
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(self.config.vad_mode)
        self.pyaudio_instance = pyaudio.PyAudio()
        if self.config.output_mode in ["droid", "both"]:
            self.ggwave_instance = ggwave.init()
        self.speech_buffer = collections.deque()
        self.silence_counter = 0
        # Initialize ggwave for both input and output if multi-input is enabled
        if self.config.multi_input or self.config.output_mode in ["droid", "both"]:
            self.ggwave_instance = ggwave.init()

    def setup_models(self):
        # Check GPU availability
        self.device = "cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu"
        logger.info(f"Using device: {self.device}")
        if self.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            # Set torch multiprocessing variables for better performance
            torch.set_float32_matmul_precision('high')
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            
            # Set CUDA kernel launch blocking for better debugging
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            
        # Initialize AI models
        logger.info(f"Loading Whisper model: {self.config.whisper_model}")
        self.whisper_model = whisper.load_model(
            self.config.whisper_model,
            device=self.device
        )
        
        # Optimize whisper model for inference
        if hasattr(self.whisper_model, 'eval'):
            self.whisper_model.eval()
            
        if self.device == "cuda" and torch.cuda.is_available():
            # Clear CUDA cache
            torch.cuda.empty_cache()
        
        if self.config.output_mode in ["tts", "both"]:
            logger.info(f"Loading TTS model: {self.config.tts_model}")
            self.tts_model = TTS(
                self.config.tts_model,
                gpu=self.device == "cuda"
            )
            # Properly set the device for TTS model
            if self.device == "cuda" and hasattr(self.tts_model, 'to'):
                self.tts_model.to(self.device)

    def setup_events(self):
        self.shutdown_event = threading.Event()
        self.ready_event = threading.Event()

    def start_input_stream(self):
        return self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=self.config.channels,
            rate=self.config.rate,
            input=True,
            frames_per_buffer=int(self.config.rate * self.config.chunk_duration_ms / 1000)
        )

    def start_output_stream(self):
        return self.pyaudio_instance.open(
            format=pyaudio.paFloat32,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            output=True,
            frames_per_buffer=4096
        )

    def log_queue_stats(self):
        """Log queue sizes to help identify backups"""
        current_time = time.time()
        if current_time - self.queue_stats_last_time < self.config.perf_interval:
            return
        
        self.queue_stats_last_time = current_time
        
        stats = {
            "audio_queue": self.audio_queue.qsize(),
            "text_queue": self.text_queue.qsize(),
            "response_queue": self.response_queue.qsize(),
            "tts_queue": self.tts_queue.qsize() if hasattr(self, 'tts_queue') else 0
        }
        
        perf_logger.info(f"QUEUE_SIZES: " + ", ".join([f"{q}: {s}" for q, s in stats.items()]))
        
    def audio_capture_thread(self):
        logger.info("Starting audio capture thread...")
        try:
            stream = self.start_input_stream()
            while not self.shutdown_event.is_set():
                try:
                    start_time = self.perf_tracker.start_timer("audio_capture")
                    
                    chunk_size = int(self.config.rate * self.config.chunk_duration_ms / 1000)
                    data = stream.read(chunk_size, exception_on_overflow=False)
                    is_speech = self.vad.is_speech(data, self.config.rate)
                    
                    if is_speech:
                        self.speech_buffer.append(data)
                        self.silence_counter = 0
                    else:
                        self.silence_counter += 1
                        self.process_speech_buffer()
                        
                    self.perf_tracker.end_timer("audio_capture", start_time)
                    self.log_queue_stats()
                    
                except Exception as e:
                    logger.error(f"Error in audio capture: {e}")
        finally:
            stream.stop_stream()
            stream.close()

    def process_speech_buffer(self):
        """Process the speech buffer if it meets criteria"""
        if len(self.speech_buffer) >= self.config.min_speech_chunks:
            if (self.silence_counter >= self.config.silence_threshold or 
                len(self.speech_buffer) >= self.config.max_speech_chunks):
                
                # Clear all previous in-progress processing when new speech is detected
                self._cancel_previous_processing()
                
                # Process collected speech only if audio queue isn't full
                if self.audio_queue.qsize() < 3:
                    audio_data = b''.join(self.speech_buffer)
                    try:
                        self.audio_queue.put_nowait(audio_data)
                    except queue.Full:
                        logger.warning("Audio queue full, dropping speech data")
                else:
                    logger.warning("Audio queue full, skipping speech processing")
                
                self.speech_buffer.clear()
    
    def _cancel_previous_processing(self):
        """Cancel any previous text processing to prevent hallucinations"""
        with self.processing_lock:
            # Increment the text ID to invalidate previous processing
            self.current_text_id += 1
            
            # Clear the text queue to avoid processing old inputs
            with self.queue_locks["text"]:
                while not self.text_queue.empty():
                    try:
                        self.text_queue.get_nowait()
                        self.text_queue.task_done()
                    except queue.Empty:
                        break
            
            # Only clear response queue if we have new textual input coming
            if self.silence_counter < 2:
                # Clear the response queue to avoid delivering old responses
                with self.queue_locks["response"]:
                    while not self.response_queue.empty():
                        try:
                            self.response_queue.get_nowait()
                            self.response_queue.task_done()
                        except queue.Empty:
                            break
                logger.info("Cleared in-progress responses due to new speech")

    def speech_recognition_thread(self):
        """Process audio data into text using Whisper"""
        logger.info("Starting speech recognition thread...")
        while not self.shutdown_event.is_set():
            try:
                # Use timeout to avoid blocking forever
                audio_data = self.audio_queue.get(timeout=1.0)
                
                start_time = self.perf_tracker.start_timer("speech_recognition")
                
                # Process audio in batches if multiple chunks available
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Move audio data to GPU if available and use with torch.no_grad for faster inference
                with torch.no_grad():
                    if self.device == "cuda":
                        audio_tensor = torch.from_numpy(audio_np).to(self.device)
                    else:
                        audio_tensor = audio_np
                        
                    # Set specific options to speed up transcription
                    transcription_options = {
                        "language": "en",
                        "task": "transcribe",
                        "fp16": self.device == "cuda",  # Use fp16 for faster GPU inference
                        "beam_size": 3,  # Smaller beam size for faster results
                        "without_timestamps": True  # Skip timestamp generation
                    }
                    
                    result = self.whisper_model.transcribe(audio_tensor, **transcription_options)
                
                if result["text"].strip():
                    text = result["text"].strip()
                    logger.info(f"Recognized: {text}")
                    
                    # Only add to queue if not already processing the same text
                    if not self.check_duplicate_text(text):
                        try:
                            self.text_queue.put_nowait(text)
                        except queue.Full:
                            logger.warning("Text queue full, dropping recognized text")
                
                self.audio_queue.task_done()
                self.perf_tracker.end_timer("speech_recognition", start_time)
                
                # Clear GPU memory after processing if on CUDA
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in speech recognition: {e}")
                import traceback
                logger.error(traceback.format_exc())

    def check_duplicate_text(self, text):
        """Check if the text is similar to what's already in the queue"""
        # Simple approach - check if the exact text is in the queue
        # For a more sophisticated approach, you could use text similarity metrics
        with self.queue_locks["text"]:
            # Create a list of all items in the queue without removing them
            queue_items = []
            for _ in range(self.text_queue.qsize()):
                try:
                    item = self.text_queue.get_nowait()
                    queue_items.append(item)
                    self.text_queue.task_done()
                except queue.Empty:
                    break
            
            # Check for duplicates
            is_duplicate = text in queue_items
            
            # Put all items back in the queue
            for item in queue_items:
                self.text_queue.put(item)
            
            return is_duplicate

    def is_simple_query(self, text):
        """Determine if a query needs complex reasoning"""
        simple_phrases = ["hello", "hi", "hey", "thanks", "thank you", "goodbye", "bye", 
                          "what's up", "whats up", "how are you", "what is up"]
        text_lower = text.lower().strip()
        
        # Check for greetings and simple phrases
        if any(phrase in text_lower for phrase in simple_phrases):
            return True
            
        # Check for short questions
        if len(text_lower.split()) < 5 and not any(w in text_lower for w in ["why", "how", "explain", "code"]):
            return True
            
        return False

    def language_processing_thread(self):
        """Process recognized text into responses using LLMs"""
        logger.info("Starting language processing thread...")
        while not self.shutdown_event.is_set():
            try:
                text = self.text_queue.get(timeout=1.0)
                
                # Limit message length if configured
                if len(text) > self.config.max_msg_length:
                    text = text[:self.config.max_msg_length]
                    logger.info(f"Input text truncated to max length ({self.config.max_msg_length})")
                
                # Assign a unique ID to this text processing
                with self.processing_lock:
                    processing_id = self.current_text_id
                    self.active_text_id = processing_id
                
                logger.info(f"Processing text [ID={processing_id}]: {text}")
                
                # Add user input to memory
                self.memory_system.add_memory(text, "user", importance=0.6)
                
                # Check if this is a simple query
                query_type = self._classify_query(text)
                logger.info(f"Query classified as: {query_type}")
                
                # Send filler response if enabled
                if self.config.filler_response:
                    filler_text = self._get_filler_response(query_type)
                    filler_response = {
                        "text": filler_text,
                        "timestamp": time.time(),
                        "complete": True
                    }
                    try:
                        self.response_queue.put_nowait(filler_response)
                    except queue.Full:
                        logger.warning("Response queue full, dropping filler response")
                
                # Get initial reflex response
                reflex_text = self._get_reflex_response(text, processing_id)
                if not reflex_text:  # If processing was cancelled
                    self.text_queue.task_done()
                    continue
                
                # Select appropriate processing based on query type
                if query_type == "simple":
                    # For simple queries, just use the reflex response
                    response_text = reflex_text
                else:
                    # For more complex queries, determine if reasoning is needed
                    if query_type == "complex" and not self.config.skip_reasoning:
                        # Get reasoning and then final response
                        reasoning = self._get_reasoning(text, processing_id)
                        if reasoning and processing_id == self.current_text_id:
                            response_text = self._get_final_response(text, reasoning, processing_id)
                        else:
                            # Fallback to reflex if reasoning was interrupted or failed
                            response_text = reflex_text
                    else:
                        # For conversational queries without reasoning
                        response_text = self._get_conversational_response(text, reflex_text, processing_id)
                
                # Final check if this response is still relevant
                with self.processing_lock:
                    if processing_id != self.current_text_id:
                        logger.info(f"Abandoning final response for outdated processing [ID={processing_id}]")
                        self.text_queue.task_done()
                        continue
                
                # Send complete response
                if response_text:
                    logger.info(f"Final response [ID={processing_id}]: {response_text[:50]}...")
                    
                    # Prepare complete response
                    complete_response = self._prepare_complete_response(response_text)
                    
                    try:
                        self.response_queue.put_nowait(complete_response)
                    except queue.Full:
                        logger.warning("Response queue full, dropping response")
                
                # Add assistant response to memory
                if response_text:
                    self.memory_system.add_memory(response_text, "assistant", importance=0.6)
                    
                    # Generate follow-up if multi-response is enabled and query is complex
                    if self.config.multi_response and query_type == "complex":
                        threading.Thread(target=self._generate_follow_up,
                                        args=(text, response_text, processing_id),
                                        daemon=True).start()
                
                self.text_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in language processing: {e}")
                import traceback
                logger.error(traceback.format_exc())

    def _prepare_complete_response(self, text):
        """Prepare a complete response object, ensuring it won't be split inappropriately"""
        # Clean up the text
        text = text.strip()
        
        # Create a complete response object
        response = {
            "text": text,
            "timestamp": time.time(),
            "complete": True  # Mark this as a complete response
        }
        
        # For TTS, we need to include a processed version
        if self.config.output_mode in ["tts", "both"]:
            tts_text = text
            if len(text) > self.config.max_tts_length:
                tts_text = text[:self.config.max_tts_length] + "... (Response truncated for voice output)"
            response["tts_text"] = tts_text
        
        return response
    
    def _get_reasoning(self, text, processing_id):
        """Get reasoning for complex queries"""
        try:
            reasoning_messages = [{"role": "user", "content": text}]
            
            # Check cache for reasoning
            reasoning_response = None
            if self.model_cache:
                reasoning_response = self.model_cache.get(self.config.reasoning_model, reasoning_messages)
            
            if not reasoning_response:
                # Check if request is still valid
                with self.processing_lock:
                    if processing_id != self.current_text_id:
                        logger.info(f"Skipping reasoning for outdated processing [ID={processing_id}]")
                        return None
                
                reasoning_response = ollama.chat(
                    model=self.config.reasoning_model,
                    messages=reasoning_messages,
                    options={"num_gpu": 1}
                )
                
                if self.model_cache:
                    self.model_cache.put(self.config.reasoning_model, reasoning_messages, reasoning_response)
            
            return reasoning_response["message"]["content"]
        except Exception as e:
            logger.error(f"Reasoning error: {e}")
            return None

    def output_thread(self):
        """Handle all output (TTS and droid communication)"""
        logger.info("Starting output thread...")
        if self.config.output_mode in ["droid", "both"]:
            output_stream = self.start_output_stream()
        
        while not self.shutdown_event.is_set():
            try:
                response_obj = self.response_queue.get(timeout=1.0)
                
                # Check if this is a complete response
                is_complete = response_obj.get("complete", False)
                
                # Handle all output in one synchronized block
                if is_complete:
                    self._process_complete_response(response_obj, output_stream)
                else:
                    logger.warning("Received incomplete response, skipping")
                
                self.response_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in output processing: {e}")
    
    def _process_complete_response(self, response_obj, output_stream):
        """Process a complete response (both TTS and droid) in a synchronized way"""
        try:
            response = response_obj["text"]
            
            # First, send the entire text via droid for immediate feedback
            if self.config.output_mode in ["droid", "both"]:
                try:
                    self._send_droid_response(response, output_stream)
                except Exception as e:
                    logger.error(f"Error in droid output: {e}")
            
            # Then generate TTS (will be played when ready)
            if self.config.output_mode in ["tts", "both"]:
                tts_text = response_obj.get("tts_text", response)
                try:
                    self._process_tts_response(tts_text)
                except Exception as e:
                    logger.error(f"Error in TTS processing: {e}")
                    
        except Exception as e:
            logger.error(f"Error in complete response processing: {e}")

    def _send_droid_response(self, response, output_stream):
        """Send complete droid response"""
        chunks = [response[i:i + self.config.max_payload_size] 
                for i in range(0, len(response), self.config.max_payload_size)]
        
        # Pre-encode all chunks
        encoded_chunks = []
        for chunk in chunks:
            waveform = ggwave.encode(
                chunk,
                protocolId=self.config.protocol_id,
                volume=self.config.volume
            )
            encoded_chunks.append(waveform)
        
        # Send chunks with appropriate delays
        chunk_delay = max(0.05, min(0.1, 5.0 / len(chunks)))
        for waveform in encoded_chunks:
            output_stream.write(waveform, len(waveform) // 4)
            time.sleep(chunk_delay)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for TTS processing"""
        # First clean the text
        text = text.strip()
        if not text:
            return []
            
        # Split on sentence endings
        sentences = []
        current = []
        
        # Improved sentence splitting with emoji and punctuation handling
        for char in text:
            current.append(char)
            
            # Check for sentence endings
            if char in '.!?' and len(current) > 1:
                # Look ahead to handle ellipsis and other cases
                if char == '.' and text[len(''.join(current)):].startswith('..'):
                    continue
                    
                sentence = ''.join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []
                
        # Add any remaining text as a sentence
        if current:
            sentence = ''.join(current).strip()
            if sentence:
                sentences.append(sentence)
        
        # Post-process sentences
        processed = []
        for sentence in sentences:
            # Handle emojis and special cases
            parts = sentence.split()
            if len(parts) == 1 and all(ord(c) > 127 for c in sentence):  # Single emoji/symbol
                if processed:
                    processed[-1] = f"{processed[-1]} {sentence}"
                else:
                    processed.append(sentence)
            else:
                processed.append(sentence)
        
        return processed

    def _process_tts_response(self, text: str):
        """Process complete TTS response with proper sentence handling"""
        try:
            # Split into sentences
            sentences = self._split_into_sentences(text)
            if not sentences:
                logger.warning("No valid sentences to process for TTS")
                return
            
            # Group sentences into chunks
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                if current_length + sentence_length > self.config.max_sentence_length and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            # Add remaining chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            # Process each chunk with TTS
            for chunk in chunks:
                if chunk.strip():
                    wav_file = self.tts_pool.generate_speech(chunk)
                    if wav_file:
                        self.tts_queue.put(wav_file)
                        logger.debug(f"TTS chunk queued: {chunk[:50]}...")
                        
        except Exception as e:
            logger.error(f"Error in TTS processing: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def tts_playback_thread(self):
        """Play TTS audio files"""
        logger.info("Starting TTS playback thread...")
        while not self.shutdown_event.is_set():
            try:
                wav_file = self.tts_queue.get(timeout=1.0)
                if os.path.exists(wav_file):
                    # Use pyaudio to play the wav file
                    with wave.open(wav_file, 'rb') as wf:
                        stream = self.pyaudio_instance.open(
                            format=self.pyaudio_instance.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True
                        )
                        
                        # Read data in chunks and play
                        chunk_size = 4096  # Larger chunks for more efficient playback
                        data = wf.readframes(chunk_size)
                        while data and not self.shutdown_event.is_set():
                            stream.write(data)
                            data = wf.readframes(chunk_size)
                            
                        stream.stop_stream()
                        stream.close()
                    
                    # Clean up the temporary file
                    try:
                        os.remove(wav_file)
                    except:
                        pass
                
                self.tts_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in TTS playback: {e}")

    def start(self):
        logger.info(f"Starting system with configuration: {asdict(self.config)}")
        
        # Log initial system info
        logger.info(f"CPU: {psutil.cpu_count(logical=True)} logical cores")
        logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB total")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
        
        # Start performance monitoring thread
        perf_thread = threading.Thread(
            target=self._performance_monitoring_thread,
            daemon=True
        )
        perf_thread.start()
        
        threads = [
            threading.Thread(target=self.audio_capture_thread, daemon=True),
            threading.Thread(target=self.speech_recognition_thread, daemon=True),
            threading.Thread(target=self.language_processing_thread, daemon=True),
            threading.Thread(target=self.output_thread, daemon=True)
        ]
        
        # Add droid input thread if multi_input is enabled
        if self.config.multi_input:
            threads.append(
                threading.Thread(target=self.droid_input_thread, daemon=True)
            )
        
        if self.config.output_mode in ["tts", "both"]:
            threads.append(
                threading.Thread(target=self.tts_playback_thread, daemon=True)
            )
        
        for thread in threads:
            thread.start()
        
        logger.info("System started. Press Ctrl+C to stop.")
        try:
            while all(thread.is_alive() for thread in threads):
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown initiated...")
        finally:
            self.shutdown()

    def _performance_monitoring_thread(self):
        """Periodically log system performance"""
        while not self.shutdown_event.is_set():
            try:
                self.perf_tracker.log_system_metrics(self.config.perf_interval)
                time.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")

    def shutdown(self):
        logger.info("Shutting down...")
        self.shutdown_event.set()
        self.memory_system.save_memories()  # Save memories before shutdown
        if self.tts_pool:
            self.tts_pool.shutdown()
        time.sleep(1)
        if hasattr(self, 'ggwave_instance'):
            ggwave.free(self.ggwave_instance)
        self.pyaudio_instance.terminate()
        logger.info("Shutdown complete")

    def droid_input_thread(self):
        """Process ggwave input signals as an alternative to voice"""
        logger.info("Starting droid input thread...")
        try:
            # Create separate input stream for ggwave signals
            input_stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=4096
            )
            
            while not self.shutdown_event.is_set():
                try:
                    data = input_stream.read(4096, exception_on_overflow=False)
                    # Decode ggwave data
                    decoded = ggwave.decode(self.ggwave_instance, data)
                    
                    if decoded is not None:
                        # We received droid input
                        droid_text = decoded.decode("utf-8")
                        logger.info(f"Received droid text input: {droid_text}")
                        
                        # Process the droid text like we would process speech
                        self._cancel_previous_processing()
                        
                        # Add the droid text directly to the text queue, bypassing STT
                        try:
                            self.text_queue.put_nowait(droid_text)
                            logger.info(f"Added droid text to processing queue: {droid_text}")
                        except queue.Full:
                            logger.warning("Text queue full, dropping droid input")
                except Exception as e:
                    logger.error(f"Error in droid input processing: {e}")
                    time.sleep(0.1)  # Avoid tight loop on error
                    
        except Exception as e:
            logger.error(f"Critical error in droid input thread: {e}", exc_info=True)
        finally:
            if 'input_stream' in locals():
                input_stream.stop_stream()
                input_stream.close()
            logger.info("Droid input thread stopped")

    def _classify_query(self, text):
        """Improved query classification for better model selection"""
        text_lower = text.lower().strip()
        words = text_lower.split()
        word_count = len(words)
        
        # Simple greetings and common phrases
        simple_phrases = ["hello", "hi", "hey", "thanks", "thank you", "goodbye", "bye", 
                          "what's up", "whats up", "how are you", "what is up"]
        
        # Complex indicators (usually need reasoning)
        complex_indicators = ["why", "how", "explain", "describe", "analyze", "compare", 
                             "difference", "similarities", "pros and cons", "advantages", 
                             "disadvantages", "code", "script", "program", "algorithm"]
        
        # Check for simple queries first
        if any(phrase in text_lower for phrase in simple_phrases) and word_count < 5:
            return "simple"
            
        # Check for complex queries
        if any(indicator in text_lower for indicator in complex_indicators):
            return "complex"
            
        # Check based on length and structure
        if word_count <= 3:
            return "simple"
        elif word_count <= 8:
            return "conversational"
        else:
            return "complex"
    
    def _get_filler_response(self, query_type):
        """Get appropriate filler response based on query type"""
        if query_type == "simple":
            return "One moment..."
        elif query_type == "complex":
            return "Hmm, let me think about that..."
        else:
            return "Let me check that for you..."
    
    def _get_reflex_response(self, text, processing_id):
        """Get quick initial response from reflex model"""
        try:
            # Check cache first if enabled
            reflex_messages = [{"role": "user", "content": text}]
            reflex_response = None
            
            if self.model_cache:
                reflex_response = self.model_cache.get(self.config.reflex_model, reflex_messages)
            
            if not reflex_response:
                reflex_response = ollama.chat(
                    model=self.config.reflex_model,
                    messages=reflex_messages,
                    options={"num_gpu": 1, "timeout": 5}  # Shorter timeout for reflex
                )
                if self.model_cache:
                    self.model_cache.put(self.config.reflex_model, reflex_messages, reflex_response)
            
            reflex_text = reflex_response["message"]["content"]
            logger.info(f"Reflex response [ID={processing_id}]: {reflex_text[:50]}...")
            
            # Check if processing request is still valid
            with self.processing_lock:
                if processing_id != self.current_text_id:
                    logger.info(f"Abandoning outdated reflex processing [ID={processing_id}]")
                    return None
                    
            return reflex_text
            
        except Exception as e:
            logger.error(f"Error in reflex response: {e}", exc_info=True)
            return "I'm thinking about that."
    
    def _get_conversational_response(self, text, reflex_text, processing_id):
        """Get conversational response using context from reflex"""
        try:
            # Prepare message history with reflex context
            conv_messages = [
                {"role": "system", "content": f"You are a helpful assistant. The quick response was: '{reflex_text}'. Build on this with a natural, complete response."},
                {"role": "user", "content": text}
            ]
            
            # Check if request is still valid
            with self.processing_lock:
                if processing_id != self.current_text_id:
                    logger.info(f"Abandoning outdated conversational processing [ID={processing_id}]")
                    return reflex_text  # Fall back to reflex response
            
            # Get response from conversational model
            final_response = ollama.chat(
                model=self.config.conversational_model,
                messages=conv_messages,
                options={"num_gpu": 1}
            )
            
            response_text = final_response["message"]["content"]
            logger.info(f"Conversational response [ID={processing_id}]: {response_text[:50]}...")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error in conversational response: {e}", exc_info=True)
            return reflex_text  # Fall back to reflex response
    
    def _get_final_response(self, text, reasoning, processing_id):
        """Generate final response using reasoning context"""
        try:
            # Prepare messages with reasoning context
            final_messages = [
                {"role": "system", "content": f"Use this reasoning as context: {reasoning}"},
                {"role": "user", "content": text}
            ]
            
            # Check if request is still valid
            with self.processing_lock:
                if processing_id != self.current_text_id:
                    logger.info(f"Abandoning outdated final processing [ID={processing_id}]")
                    return None
            
            # Get response from model
            final_response = ollama.chat(
                model=self.config.complex_model,
                messages=final_messages,
                options={"num_gpu": 1}
            )
            
            response_text = final_response["message"]["content"]
            logger.info(f"Final reasoned response [ID={processing_id}]: {response_text[:50]}...")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error in final response: {e}", exc_info=True)
            return "I'm having trouble processing that. Could you try again?"
    
    def _generate_follow_up(self, original_text, primary_response, processing_id):
        """Generate a follow-up response when appropriate"""
        try:
            # Check if still valid
            with self.processing_lock:
                if processing_id != self.current_text_id:
                    return
                    
            # Construct follow-up prompt
            follow_up_messages = [
                {"role": "system", "content": "Determine if a follow-up is needed to add valuable information to the previous response. If yes, provide the follow-up. If not, respond with 'NO_FOLLOW_UP_NEEDED'."},
                {"role": "user", "content": original_text},
                {"role": "assistant", "content": primary_response},
                {"role": "user", "content": "Is a follow-up needed? If yes, what would you add?"}
            ]
            
            # Get follow-up decision
            follow_up_response = ollama.chat(
                model=self.config.reflex_model,
                messages=follow_up_messages
            )
            
            follow_up_text = follow_up_response["message"]["content"]
            
            # Check if follow-up is actually needed
            if "NO_FOLLOW_UP_NEEDED" in follow_up_text:
                logger.info("No follow-up needed")
                return
                
            # Clean up the follow-up text
            follow_up_text = follow_up_text.replace("NO_FOLLOW_UP_NEEDED", "").strip()
            
            # Check if still valid
            with self.processing_lock:
                if processing_id != self.current_text_id:
                    return
                    
            # Only send if meaningful content exists
            if len(follow_up_text) > 20:
                logger.info(f"Sending follow-up: {follow_up_text[:50]}...")
                
                follow_up_response = {
                    "text": follow_up_text,
                    "timestamp": time.time(),
                    "complete": True,
                    "is_follow_up": True
                }
                
                try:
                    self.response_queue.put(follow_up_response)
                    # Add to memory
                    self.memory_system.add_memory(follow_up_text, "assistant", importance=0.4)
                except queue.Full:
                    logger.warning("Response queue full, dropping follow-up")
        except Exception as e:
            logger.error(f"Error generating follow-up: {e}")

class TTSWorkerPool:
    """Manages a pool of TTS workers for parallel processing"""
    
    def __init__(self, config: SystemConfig, device: str):
        self.config = config
        self.device = device
        self.workers = []
        self.tts_queue = queue.Queue(maxsize=config.tts_queue_size)
        self.result_queue = queue.Queue()
        self.shutdown_event = threading.Event()
        self.tts_model = None  # Single shared TTS model
        self.model_lock = threading.Lock()
        
        # Initialize TTS model once
        self._initialize_tts_model()
        
        # Initialize worker threads
        for _ in range(config.tts_worker_threads):
            worker = threading.Thread(target=self._tts_worker, daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def _initialize_tts_model(self):
        """Initialize TTS model with proper error handling"""
        try:
            with self.model_lock:
                if self.tts_model is None:
                    logger.info("Initializing shared TTS model...")
                    self.tts_model = TTS(self.config.tts_model)
                    if self.device == "cuda":
                        self.tts_model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {e}")
            raise
    
    def _tts_worker(self):
        """Worker thread for TTS processing"""
        try:
            while not self.shutdown_event.is_set():
                try:
                    batch = self.tts_queue.get(timeout=1.0)
                    if batch is None:
                        break
                    
                    text, temp_file = batch
                    success = False
                    error_msg = None
                    
                    try:
                        # Use shared model with lock
                        with self.model_lock:
                            with torch.no_grad():
                                self.tts_model.tts_to_file(
                                    text=text,
                                    file_path=temp_file,
                                    speaker=None
                                )
                        success = True
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"TTS generation error: {e}")
                    finally:
                        self.tts_queue.task_done()
                        self.result_queue.put((success, temp_file if success else error_msg))
                        
                except queue.Empty:
                    continue
                    
        except Exception as e:
            logger.error(f"TTS worker failed: {e}")
    
    def generate_speech(self, text: str) -> Optional[str]:
        """Generate speech for text, returns filename or None on failure"""
        try:
            # Create unique temp file
            temp_file = f"temp_speech_{time.time_ns()}.wav"
            
            # Add to queue with timeout
            try:
                self.tts_queue.put((text, temp_file), timeout=self.config.response_timeout)
            except queue.Full:
                logger.warning("TTS queue full, dropping request")
                return None
            
            # Wait for result with timeout
            try:
                success, result = self.result_queue.get(timeout=self.config.response_timeout)
                if success:
                    return result
                else:
                    logger.error(f"TTS generation failed: {result}")
                    return None
            except queue.Empty:
                logger.warning("TTS generation timed out")
                return None
            finally:
                self.result_queue.task_done()
                
        except Exception as e:
            logger.error(f"Error in generate_speech: {e}")
            return None
    
    def shutdown(self):
        """Shutdown the worker pool"""
        self.shutdown_event.set()
        for _ in self.workers:
            try:
                self.tts_queue.put(None)  # Signal workers to stop
            except:
                pass
        for worker in self.workers:
            worker.join(timeout=1.0)
        # Clear queues
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except:
                pass
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
                self.result_queue.task_done()
            except:
                pass

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Audio Communication System")
    parser.add_argument("--output-mode", type=str, choices=["tts", "droid", "both"],
                       default="both", help="Output mode selection")
    parser.add_argument("--whisper-model", type=str, default="base",
                       help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--use-gpu", action="store_true", default=True,
                       help="Use GPU for models (default: True)")
    parser.add_argument("--force-cpu", action="store_true",
                       help="Force CPU usage even if GPU is available")
    parser.add_argument("--perf-log", action="store_true", 
                        help="Enable detailed performance logging")
    parser.add_argument("--perf-interval", type=int, default=10,
                        help="Interval between performance logs in seconds")
    parser.add_argument("--skip-reasoning", action="store_true", default=True,
                       help="Skip reasoning for simple queries")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Enable parallel inference")
    parser.add_argument("--cache", action="store_true", default=True,
                       help="Enable response caching")
    parser.add_argument("--max-tts", type=int, default=200,
                       help="Maximum text length for TTS")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Maximum processing time per request (seconds)")
    parser.add_argument("--memory-file", type=str, default="conversation_memory.json",
                       help="Path to memory storage file")
    parser.add_argument("--max-memories", type=int, default=100,
                       help="Maximum number of memories to retain")
    parser.add_argument("--multi-input", action="store_true", default=False,
                       help="Enable multi-input (ggwave + speech)")
    parser.add_argument("--max-msg-length", type=int, default=200,
                       help="Maximum message length for processing")
    parser.add_argument("--filler-response", action="store_true", default=False,
                       help="Enable filler responses")
    parser.add_argument("--multi-response", action="store_true", default=False,
                       help="Enable multi-response generation")
    return parser.parse_args()

def main():
    args = parse_args()
    config = SystemConfig(
        whisper_model=args.whisper_model,
        output_mode=args.output_mode,
        use_gpu=not args.force_cpu and args.use_gpu,
        log_performance=args.perf_log,
        perf_interval=args.perf_interval,
        skip_reasoning=args.skip_reasoning,
        parallel_inference=args.parallel,
        use_cache=args.cache,
        max_tts_length=args.max_tts,
        max_processing_time=args.timeout,
        memory_file=args.memory_file,
        max_memory_entries=args.max_memories,
        multi_input=args.multi_input,
        max_msg_length=args.max_msg_length,
        filler_response=args.filler_response,
        multi_response=args.multi_response
    )
    
    system = UnifiedAudioSystem(config)
    system.start()

if __name__ == "__main__":
    main()
