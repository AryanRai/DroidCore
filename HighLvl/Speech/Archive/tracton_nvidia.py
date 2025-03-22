import os
import numpy as np
import torch
import sounddevice as sd
import soundfile as sf
import whisper
from TTS.api import TTS
import wave
import pyaudio
import time
from pathlib import Path

class EmotionalVoiceSystem:
    def __init__(self):
        # Initialize TTS
        print("Initializing TTS system...")
        self.tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True)
        
        # Initialize Whisper for STT
        print("Loading Whisper model (this may take a moment)...")
        self.whisper_model = whisper.load_model("base")
        
        # Available emotions for VITS model
        self.emotions = {
            "happy": {"pitch": 1.2, "speed": 1.1},
            "sad": {"pitch": 0.8, "speed": 0.9},
            "angry": {"pitch": 1.1, "speed": 1.2},
            "calm": {"pitch": 1.0, "speed": 1.0},
            "surprised": {"pitch": 1.3, "speed": 1.15},
            "fearful": {"pitch": 0.9, "speed": 1.2}
        }
        
        # Create output directory if it doesn't exist
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Audio recording parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Whisper expects 16kHz
        self.chunk = 1024
        self.audio = pyaudio.PyAudio()
        
        # Available voices
        self.available_speakers = self.tts.speakers
        if self.available_speakers:
            print(f"Available voices: {', '.join(self.available_speakers)}")
            self.default_speaker = self.available_speakers[0]
        else:
            self.default_speaker = None
            
    def text_to_speech(self, text, emotion="neutral", speaker=None, output_file="output/tts_output.wav"):
        """
        Convert text to speech with emotional expression
        
        Args:
            text (str): Text to convert to speech
            emotion (str): Emotion to express (happy, sad, angry, calm, surprised, fearful)
            speaker (str, optional): Speaker ID if available
            output_file (str): Output file path
        """
        if emotion not in self.emotions and emotion != "neutral":
            print(f"Emotion '{emotion}' not available. Using neutral.")
            emotion = "neutral"
        
        # Apply emotion parameters
        if emotion != "neutral":
            params = self.emotions[emotion]
            pitch_modifier = params["pitch"]
            speed_modifier = params["speed"]
        else:
            pitch_modifier = 1.0
            speed_modifier = 1.0
            
        # Generate speech
        print(f"Generating speech with emotion: {emotion}")
        
        speaker_kwargs = {}
        if speaker and self.available_speakers and speaker in self.available_speakers:
            speaker_kwargs["speaker"] = speaker
        elif self.default_speaker:
            speaker_kwargs["speaker"] = self.default_speaker
        
        self.tts.tts_to_file(
            text=text,
            file_path=output_file,
            speed=speed_modifier,
            **speaker_kwargs
        )
        
        # Apply pitch modification if needed
        if pitch_modifier != 1.0:
            self._modify_pitch(output_file, pitch_modifier)
            
        print(f"Speech generated and saved to {output_file}")
        
        # Play the audio
        self.play_audio(output_file)
        
    def _modify_pitch(self, file_path, pitch_factor):
        """Apply pitch modification to the audio file"""
        # For simplicity, this is a placeholder - in a production system
        # you would use a library like librosa to properly modify pitch
        print(f"Applied pitch factor: {pitch_factor}")
        
    def play_audio(self, file_path):
        """Play audio file using sounddevice"""
        try:
            data, fs = sf.read(file_path)
            sd.play(data, fs)
            sd.wait()
        except Exception as e:
            print(f"Error playing audio: {e}")
            
    def record_audio(self, duration=5, output_file="output/recorded_audio.wav"):
        """
        Record audio from microphone
        
        Args:
            duration (int): Recording duration in seconds
            output_file (str): Output file path
        """
        print(f"Recording for {duration} seconds...")
        
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        frames = []
        for i in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)
            
        stream.stop_stream()
        stream.close()
        
        # Save the recorded audio
        wf = wave.open(output_file, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"Recording saved to {output_file}")
        return output_file
        
    def speech_to_text(self, audio_file=None, record_duration=5):
        """
        Convert speech to text using Whisper
        
        Args:
            audio_file (str, optional): Audio file path. If None, record new audio
            record_duration (int): Recording duration if recording new audio
            
        Returns:
            str: Transcribed text
        """
        if audio_file is None:
            audio_file = self.record_audio(duration=record_duration)
            
        print("Transcribing audio with Whisper...")
        result = self.whisper_model.transcribe(audio_file)
        transcribed_text = result["text"].strip()
        print(f"Transcription: {transcribed_text}")
        return transcribed_text
    
    def interactive_mode(self):
        """Run an interactive session where you can speak and get responses"""
        print("\n========== INTERACTIVE MODE ==========")
        print("Speak into your microphone and the system will respond.")
        print("Say 'exit' or 'quit' to end the session.")
        
        while True:
            print("\nListening...")
            user_text = self.speech_to_text(record_duration=5)
            
            if user_text.lower() in ["exit", "quit", "stop", "end"]:
                print("Ending interactive session.")
                break
                
            # Simple response logic - in a real app, you might connect to an LLM here
            if "hello" in user_text.lower() or "hi" in user_text.lower():
                response = "Hello there! How can I help you today?"
                emotion = "happy"
            elif "how are you" in user_text.lower():
                response = "I'm doing well, thank you for asking. How about yourself?"
                emotion = "calm"
            elif "sad" in user_text.lower() or "unhappy" in user_text.lower():
                response = "I'm sorry to hear that. Is there anything I can do to help?"
                emotion = "sad"
            elif "angry" in user_text.lower() or "mad" in user_text.lower():
                response = "I understand you're feeling frustrated. Let's try to work through this together."
                emotion = "calm"
            elif "joke" in user_text.lower():
                response = "Why don't scientists trust atoms? Because they make up everything!"
                emotion = "happy"
            else:
                response = "I heard what you said, but I'm not sure how to respond. Could you try saying something else?"
                emotion = "neutral"
                
            print(f"Responding with: '{response}' (emotion: {emotion})")
            self.text_to_speech(response, emotion=emotion)
            
    def close(self):
        """Clean up resources"""
        self.audio.terminate()
        print("Resources cleaned up.")


def demo():
    """Run a demonstration of the system's capabilities"""
    system = EmotionalVoiceSystem()
    
    try:
        # Demonstrate TTS with different emotions
        print("\n===== TTS DEMONSTRATION =====")
        
        # Happy emotion
        system.text_to_speech(
            "I'm so excited to meet you! This is what I sound like when I'm happy.",
            emotion="happy",
            output_file="output/happy.wav"
        )
        
        # Sad emotion
        system.text_to_speech(
            "I'm feeling a bit down today. This is what I sound like when I'm sad.",
            emotion="sad",
            output_file="output/sad.wav"
        )
        
        # Angry emotion
        system.text_to_speech(
            "I can't believe this happened again! This is what I sound like when I'm angry.",
            emotion="angry",
            output_file="output/angry.wav"
        )
        
        # STT demonstration
        print("\n===== STT DEMONSTRATION =====")
        print("Let's test speech recognition. Please speak after the prompt.")
        transcribed_text = system.speech_to_text(record_duration=5)
        
        # Echo back what was heard
        system.text_to_speech(
            f"I heard you say: {transcribed_text}",
            emotion="surprised",
            output_file="output/response.wav"
        )
        
        # Interactive mode
        response = input("\nWould you like to try interactive mode? (y/n): ")
        if response.lower() == 'y':
            system.interactive_mode()
            
    finally:
        system.close()
        

if __name__ == "__main__":
    print("Starting Emotional Voice System...")
    demo()