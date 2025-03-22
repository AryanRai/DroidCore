@echo off
echo Running with dual input (speech + droid)...

REM Set environment variables for improved performance
set OMP_NUM_THREADS=4
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set TOKENIZERS_PARALLELISM=true
set PYTHONUNBUFFERED=1

REM Launch with dual input settings
python Main_Combine.py ^
  --whisper-model=base ^
  --multi-input ^
  --perf-log ^
  --skip-reasoning ^
  --parallel ^
  --cache ^
  --max-tts=100 ^
  --timeout=15 ^
  --multi-response ^
  --filler-response

echo System terminated.
