import threading
import queue
import time
import os
import tempfile
from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealtimeTTS:
    def __init__(self, model_name="tts_models/en/ljspeech/tacotron2-DDC", 
                 gpu=True, 
                 sampling_rate=22050,
                 speaker="Jenny"):
        """
        Initialize the real-time TTS system.
        
        Args:
            model_name (str): The TTS model to use
            gpu (bool): Whether to use GPU acceleration
            sampling_rate (int): Audio sampling rate
            speaker (str): For multi-speaker models, specify the speaker
        """
        logger.info(f"Initializing TTS with model: {model_name}, GPU: {gpu}")
        self.tts = TTS(model_name, gpu=gpu)
        self.sampling_rate = sampling_rate
        self.speaker = speaker
        self.temp_dir = tempfile.mkdtemp()
        self.settings = {
            # High-quality settings
            'speed': 1.0,  # Normal speed
            'pitch': 0.0,  # No pitch adjustment
            'volume': 1.0,  # Default volume
            'silence_threshold_db': -40,  # Better silence detection
            'min_pause_ms': 300,  # Minimum pause between sentences (ms)
        }
        
        # Set up the queues
        self.text_queue = queue.Queue()  # For incoming text segments
        self.audio_queue = queue.Queue()  # For processed audio segments
        
        # Set up the processing threads
        self.text_processor_thread = threading.Thread(target=self._process_text, daemon=True)
        self.audio_player_thread = threading.Thread(target=self._play_audio, daemon=True)
        
        # Control flags
        self.running = False
        self.current_buffer = ""
        self.sentence_boundary = re.compile(r'[.!?;]\s+')
        self.file_counter = 0
        
    def start(self):
        """Start the TTS system"""
        if self.running:
            logger.warning("TTS system is already running")
            return
            
        logger.info("Starting TTS system")
        self.running = True
        self.text_processor_thread.start()
        self.audio_player_thread.start()
        
    def stop(self):
        """Stop the TTS system"""
        logger.info("Stopping TTS system")
        self.running = False
        
        # Add a sentinel value to ensure threads exit cleanly
        self.text_queue.put(None)
        self.audio_queue.put(None)
        
        # Wait for threads to finish
        if self.text_processor_thread.is_alive():
            self.text_processor_thread.join(timeout=2.0)
        if self.audio_player_thread.is_alive():
            self.audio_player_thread.join(timeout=2.0)
            
        # Clean up temp directory
        self._cleanup_temp_files()
        
        logger.info("TTS system stopped")
        
    def add_text(self, text):
        """
        Add text to the processing queue.
        
        Args:
            text (str): Text to be synthesized
        """
        if not self.running:
            logger.warning("Cannot add text, TTS system is not running")
            return
            
        logger.info(f"Adding text: {text[:50]}{'...' if len(text) > 50 else ''}")
        self.text_queue.put(text)
        
    def adjust_settings(self, **kwargs):
        """Update TTS settings."""
        for key, value in kwargs.items():
            if key in self.settings:
                self.settings[key] = value
                logger.info(f"Updated setting: {key} = {value}")
            else:
                logger.warning(f"Unknown setting: {key}")
                
    def _process_text(self):
        """Process text segments into audio"""
        logger.info("Text processor thread started")
        
        while self.running:
            try:
                # Get text from the queue
                text = self.text_queue.get(timeout=0.1)
                
                # Check for sentinel value
                if text is None:
                    break
                    
                # Add to current buffer
                self.current_buffer += text
                
                # Split into sentences and process complete sentences
                self._process_sentences()
                
                # Mark task as done
                self.text_queue.task_done()
                
            except queue.Empty:
                # No new text, just process any complete sentences in the buffer
                self._process_sentences()
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in text processor: {str(e)}")
                
        logger.info("Text processor thread stopped")
        
    def _process_sentences(self):
        """Process complete sentences from the buffer"""
        if not self.current_buffer:
            return
            
        # Find sentence boundaries
        matches = list(self.sentence_boundary.finditer(self.current_buffer))
        
        if matches:
            # Get the position of the last complete sentence
            last_match = matches[-1]
            end_pos = last_match.end()
            
            # Extract complete sentences
            sentences = self.current_buffer[:end_pos].strip()
            
            # Update buffer to contain only incomplete sentences
            self.current_buffer = self.current_buffer[end_pos:].strip()
            
            # Generate audio for the complete sentences
            logger.info(f"Generating audio for: {sentences[:50]}{'...' if len(sentences) > 50 else ''}")
            self._generate_audio(sentences)
            
        # Process the remaining buffer if it's been waiting too long or is long enough
        elif len(self.current_buffer) > 80:  # Process chunks of ~80 chars if no sentence boundary
            # Find a good spot to break the text (e.g., at a comma or space)
            break_points = [m.end() for m in re.finditer(r'[,]\s+|\s+', self.current_buffer)]
            if break_points and break_points[-1] > 40:  # Ensure we have enough text
                break_pos = break_points[-1]
                chunk = self.current_buffer[:break_pos]
                self.current_buffer = self.current_buffer[break_pos:].strip()
                logger.info(f"Processing chunk at natural break: {chunk[:50]}{'...' if len(chunk) > 50 else ''}")
                self._generate_audio(chunk)
            
    def _generate_audio(self, text):
        """Generate audio from text using TTS with enhanced quality"""
        try:
            # Clean up text for better synthesis
            text = self._preprocess_text(text)
            
            # Generate a unique temporary file path
            self.file_counter += 1
            temp_path = os.path.join(self.temp_dir, f"tts_segment_{self.file_counter}.wav")
            
            # Generate speech to file
            if hasattr(self.tts, 'speakers') and self.tts.speakers and self.speaker in self.tts.speakers:
                # For multi-speaker models
                self.tts.tts_to_file(text=text, file_path=temp_path, speaker=self.speaker)
            else:
                # For single-speaker models
                self.tts.tts_to_file(text=text, file_path=temp_path)
            
            # Load and enhance the audio
            audio_segment = AudioSegment.from_wav(temp_path)
            enhanced_audio = self._enhance_audio(audio_segment)
            
            # Queue the enhanced audio
            self.audio_queue.put(enhanced_audio)
            
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
        
    def _preprocess_text(self, text):
        """Preprocess text for better TTS quality"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'([.!?;,:])\s*', r'\1 ', text)
        
        # Add commas for natural pauses in long sentences
        if len(text) > 100 and ',' not in text:
            # Try to add commas at natural phrase breaks
            text = re.sub(r'(\s(and|but|or|because|however|therefore)\s)', r', \2 ', text)
            
        return text
    
    def _enhance_audio(self, audio_segment):
        """Enhance audio quality"""
        # Apply settings
        enhanced = audio_segment
        
        # Adjust speed if needed
        if self.settings['speed'] != 1.0:
            enhanced = enhanced.speedup(playback_speed=self.settings['speed'])
            
        # Adjust volume if needed
        if self.settings['volume'] != 1.0:
            enhanced = enhanced.apply_gain(self.settings['volume'] * 10)  # dB scale
            
        # Normalize audio (increase volume to a target level)
        enhanced = enhanced.normalize()
        
        # Remove silence at the beginning and end
        enhanced = enhanced.strip_silence(
            silence_thresh=self.settings['silence_threshold_db'],
            padding=50  # ms of silence to leave
        )
        
        # Ensure there's a small pause at the end
        pause = AudioSegment.silent(duration=self.settings['min_pause_ms'])
        enhanced = enhanced + pause
        
        return enhanced
            
    def _play_audio(self):
        """Play audio segments from the queue"""
        logger.info("Audio player thread started")
        
        while self.running:
            try:
                # Get audio from the queue
                audio = self.audio_queue.get(timeout=0.1)
                
                # Check for sentinel value
                if audio is None:
                    break
                    
                # Play the audio
                logger.info(f"Playing audio segment of length: {len(audio)} ms")
                play(audio)
                
                # Mark task as done
                self.audio_queue.task_done()
                
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in audio player: {str(e)}")
                
        logger.info("Audio player thread stopped")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            os.rmdir(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Initialize and start the TTS system with a high-quality model
    tts_system = RealtimeTTS(
        model_name="tts_models/en/ljspeech/tacotron2-DDC",  # Use a high-quality model
        gpu=True
    )
    tts_system.start()
    
    try:
        # Example of adding text in chunks as if streaming
        print("Starting to stream text...")
        
        # First chunk
        tts_system.add_text("Hello, this is a real-time text-to-speech system with enhanced audio quality. ")
        time.sleep(1)  # Simulate delay between text additions
        
        # Second chunk
        tts_system.add_text("It can handle text being added incrementally while maintaining a pleasant voice. ")
        time.sleep(1.5)
        
        # Third chunk
        tts_system.add_text("The system processes text in chunks and ensures smooth transitions between segments. ")
        time.sleep(1)
        
        # Large paragraph example
        long_text = """
        This system is designed to produce more natural-sounding speech. It breaks the text into proper phrases
        and processes them with attention to rhythm and intonation. The audio is normalized and enhanced to
        provide a pleasant listening experience. You can adjust settings like speed, volume, and pauses to
        customize the speech output. The system runs in separate threads to ensure real-time performance
        while maintaining high audio quality.
        """
        
        tts_system.add_text(long_text)
        
        # Wait to allow all text to be processed and spoken
        print("Waiting for all text to be processed...")
        time.sleep(20)
        
    finally:
        # Stop the TTS system
        print("Stopping TTS system...")
        tts_system.stop()
        print("Done!")