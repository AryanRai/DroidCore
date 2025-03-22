@echo off
echo Running with TTS optimization...

REM Set environment variables for better performance
set OMP_NUM_THREADS=4
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set TOKENIZERS_PARALLELISM=true
set PYTHONUNBUFFERED=1

REM Launch with optimized TTS settings
python Main_Combine.py ^
  --whisper-model=base ^
  --max-tts=50 ^
  --max-sentence=25 ^
  --tts-workers=2 ^
  --perf-log ^
  --skip-reasoning ^
  --parallel ^
  --cache ^
  --timeout=15

echo System terminated.
