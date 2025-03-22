from TTS.api import TTS# import required module
from playsound import playsound
 

model_name = "tts_models/en/jenny/jenny"
tts = TTS(model_name, gpu=True)
tts.tts_to_file(text="Hello world!", file_path="output.wav")

# for playing note.mp3 file
playsound('output.wav')
print('playing sound using  playsound')



# List available 🐸TTS models and choose the first one
#for i in TTS.list_models():
#   print(i)

#usable models
#tts_models/en/ljspeech/tacotron2-DCA

#tts_models/en/ljspeech/fast_pitch

#tts_models/en/jenny/jenny