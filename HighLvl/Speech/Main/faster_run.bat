@echo off
echo Running with high-performance optimization...

REM Set environment variables for improved performance
set OMP_NUM_THREADS=4
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set TOKENIZERS_PARALLELISM=true
set PYTHONUNBUFFERED=1

REM Launch with optimized parameters
python Main_Combine.py ^
  --whisper-model=base ^
  --perf-log ^
  --skip-reasoning ^
  --cache ^
  --max-tts=150 ^
  --timeout=20

echo System terminated.
