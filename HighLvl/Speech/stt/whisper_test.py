import pyaudio
import whisper
import threading
import queue
import numpy as np
import webrtcvad
import collections

# Initialize VAD
vad = webrtcvad.Vad()
vad.set_mode(1)  # Aggressiveness mode (0-3)

# Initialize Whisper model (switching to "small" for faster processing)
model = whisper.load_model("small")  # "small" is faster than "medium"

# Audio settings
CHUNK_DURATION_MS = 30  # 30 ms chunks
CHUNK_SIZE = int(16000 * CHUNK_DURATION_MS / 1000)  # Samples per chunk
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Buffer settings
SILENCE_THRESHOLD = 5  # Reduced to 150 ms (5 chunks) of silence to trigger transcription
MAX_SPEECH_CHUNKS = 100  # ~3 seconds max before forced transcription
MIN_SPEECH_CHUNKS = 10  # ~300 ms minimum speech length

# Queue for audio data
transcription_queue = queue.Queue()

# Speech buffer
speech_buffer = collections.deque()
silence_counter = 0

# Audio capture thread with VAD
def capture_thread():
    global speech_buffer, silence_counter
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)
    print("Listening...")
    while True:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        is_speech = vad.is_speech(data, RATE)
        if is_speech:
            speech_buffer.append(data)
            silence_counter = 0
        else:
            silence_counter += 1
            if len(speech_buffer) >= MIN_SPEECH_CHUNKS and silence_counter >= SILENCE_THRESHOLD:
                # Send audio for transcription after 150 ms of silence
                audio_data = b''.join(speech_buffer)
                transcription_queue.put(audio_data)
                speech_buffer.clear()
            elif len(speech_buffer) >= MAX_SPEECH_CHUNKS:
                # Force transcription if buffer hits 3 seconds
                audio_data = b''.join(list(speech_buffer)[:MAX_SPEECH_CHUNKS])
                transcription_queue.put(audio_data)
                for _ in range(MAX_SPEECH_CHUNKS):
                    speech_buffer.popleft()

# Transcription processing thread
def processing_thread():
    while True:
        audio_data = transcription_queue.get()
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        result = model.transcribe(audio_np, language="en")
        print("Transcription:", result['text'])
        transcription_queue.task_done()

# Start threads
capture_t = threading.Thread(target=capture_thread, daemon=True)
processing_t = threading.Thread(target=processing_thread, daemon=True)
capture_t.start()
processing_t.start()

try:
    while True:
        threading.Event().wait()
except KeyboardInterrupt:
    print("\nStopping...")