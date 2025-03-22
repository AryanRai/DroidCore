import time
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
from playsound import playsound
import os
import re

# Create a queue for communication between threads
tts_queue = queue.Queue()

# Flag to control worker thread
exit_flag = threading.Event()

# Timer for debouncing input
typing_timer = None
TYPING_TIMEOUT = 0.5  # seconds

def text_to_speech_worker(model_name, use_gpu):
    """Worker thread that processes TTS requests"""
    try:
        # Initialize TTS model - import here to avoid initial loading time
        from TTS.api import TTS
        
        print("Loading TTS model...")
        tts = TTS(model_name, gpu=use_gpu)
        print("TTS model loaded!")
        
        counter = 0
        last_text = ""
        
        while not exit_flag.is_set():
            try:
                # Get text from queue with timeout
                text = tts_queue.get(timeout=1.0)
                
                if text and text != last_text:
                    # Create a unique filename for each speech output
                    filename = f"speech_{counter}.wav"
                    counter += 1
                    
                    # Generate speech
                    tts.tts_to_file(text=text, file_path=filename)
                    
                    # Play the generated audio
                    playsound(filename)
                    
                    # Clean up the file after playing
                    try:
                        os.remove(filename)
                    except:
                        pass  # Ignore errors if file can't be deleted
                    
                    # Update last spoken text
                    last_text = text
                
                # Mark task as done
                tts_queue.task_done()
            except queue.Empty:
                # Queue was empty for the timeout period - just continue the loop
                continue
            except Exception as e:
                print(f"Error in TTS worker: {e}")
                continue
    except Exception as e:
        print(f"Failed to initialize TTS: {e}")

class RealTimeTTS:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Text-to-Speech")
        self.root.geometry("600x400")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text input area
        self.text_input = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=10)
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=5)
        self.text_input.bind('<KeyRelease>', self.on_key_release)
        
        # Create control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # TTS Model selection
        ttk.Label(control_frame, text="TTS Model:").pack(side=tk.LEFT, padx=5)
        
        self.model_var = tk.StringVar(value="tts_models/en/jenny/jenny")
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, width=30)
        self.model_combo['values'] = [
            "tts_models/en/jenny/jenny", 
            "tts_models/en/ljspeech/tacotron2-DCA",
            "tts_models/en/ljspeech/fast_pitch"
        ]
        self.model_combo.pack(side=tk.LEFT, padx=5)
        
        # GPU checkbox
        self.use_gpu = tk.BooleanVar(value=True)
        self.gpu_check = ttk.Checkbutton(control_frame, text="Use GPU", variable=self.use_gpu)
        self.gpu_check.pack(side=tk.LEFT, padx=5)
        
        # Start button
        self.start_button = ttk.Button(control_frame, text="Start TTS", command=self.start_tts)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Stop button (initially disabled)
        self.stop_button = ttk.Button(control_frame, text="Stop TTS", command=self.stop_tts, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.pack(anchor=tk.W, pady=5)
        
        # Initialize worker thread as None
        self.worker_thread = None
        
    def start_tts(self):
        """Start the TTS worker thread"""
        if self.worker_thread and self.worker_thread.is_alive():
            return
            
        # Reset exit flag
        exit_flag.clear()
        
        # Create and start the worker thread
        self.worker_thread = threading.Thread(
            target=text_to_speech_worker, 
            args=(self.model_var.get(), self.use_gpu.get()),
            daemon=True
        )
        self.worker_thread.start()
        
        # Update UI
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.model_combo.config(state=tk.DISABLED)
        self.gpu_check.config(state=tk.DISABLED)
        self.status_var.set("TTS running - type to speak")
        
    def stop_tts(self):
        """Stop the TTS worker thread"""
        if self.worker_thread and self.worker_thread.is_alive():
            # Set exit flag to signal thread to stop
            exit_flag.set()
            
            # Wait for thread to finish (non-blocking)
            threading.Thread(target=self._join_worker_thread, daemon=True).start()
            
            # Update UI
            self.status_var.set("Stopping TTS...")
            
    def _join_worker_thread(self):
        """Join the worker thread and update UI when done"""
        if self.worker_thread:
            self.worker_thread.join()
            
        # Update UI from main thread
        self.root.after(0, self._update_ui_after_stop)
        
    def _update_ui_after_stop(self):
        """Update UI after worker thread has stopped"""
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.model_combo.config(state=tk.NORMAL)
        self.gpu_check.config(state=tk.NORMAL)
        self.status_var.set("Ready")
        
    def on_key_release(self, event):
        """Handler for key release events"""
        global typing_timer
        
        # Don't process if TTS is not running
        if not self.worker_thread or not self.worker_thread.is_alive():
            return
            
        # Cancel previous timer if exists
        if typing_timer:
            typing_timer.cancel()
            
        # Start a new timer
        typing_timer = threading.Timer(TYPING_TIMEOUT, self.process_text)
        typing_timer.start()
        
    def process_text(self):
        """Process the current text input"""
        text = self.text_input.get("1.0", tk.END).strip()
        if text:
            tts_queue.put(text)
            
    def on_closing(self):
        """Clean up when window is closed"""
        self.stop_tts()
        self.root.destroy()
        
def main():
    """Main function to start the application"""
    root = tk.Tk()
    app = RealTimeTTS(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
    
if __name__ == "__main__":
    main()