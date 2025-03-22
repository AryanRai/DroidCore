# -*- coding: utf-8 -*-
import torch
from whisperspeech.pipeline import Pipeline
import keyboard  # For capturing keyboard input in real-time
import time
import pyaudio  # For audio playback
import wave  # To handle WAV files
import os

# Check if GPU is available
if not torch.cuda.is_available():
    raise BaseException("This script requires CUDA. Please run it on a machine with a GPU.")

# Initialize the WhisperSpeech pipeline
pipe = Pipeline(
    t2s_ref='whisperspeech/whisperspeech:t2s-v1.95-small-8lang.model',
    s2a_ref='whisperspeech/whisperspeech:s2a-v1.95-medium-7lang.model'
)

def play_audio(file_path):
    """Play audio from a WAV file using PyAudio."""
    # Open the WAV file
    wf = wave.open(file_path, 'rb')

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a stream
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )

    # Read and play the audio data
    chunk = 1024
    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)

    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()

def generate_speech(text, lang='en', cps=10):
    """Generate and play speech from text."""
    # Generate audio from text
    audio = pipe.generate(text, lang=lang, cps=cps)
    
    # Save the audio to a temporary WAV file
    temp_file = "temp_speech.wav"
    audio.save(temp_file)  # Assuming WhisperSpeech's generate() returns an object with a save method
    
    # Play the audio
    play_audio(temp_file)
    
    # Clean up the temporary file
    os.remove(temp_file)

def real_time_speech():
    print("Type your text and press Enter to hear it spoken. Press 'Ctrl+C' to exit.")
    buffer = ""
    
    while True:
        try:
            # Get input character by character
            char = keyboard.read_key()
            
            if char == "enter":
                if buffer.strip():  # Only process non-empty input
                    generate_speech(buffer.strip())
                buffer = ""  # Clear buffer after speaking
            elif char == "backspace":
                buffer = buffer[:-1]  # Remove last character
            elif char and len(char) == 1:  # Only append single characters
                buffer += char
            
            # Optional: Display current buffer
            print(f"Current input: {buffer}", end='\r')
            time.sleep(0.01)  # Small delay to avoid high CPU usage
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    real_time_speech()