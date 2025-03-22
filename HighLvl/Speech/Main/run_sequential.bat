@echo off
echo Running in sequential processing mode...

REM Set environment variables for controlled processing
set OMP_NUM_THREADS=4
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set TOKENIZERS_PARALLELISM=false

REM Launch with sequential processing settings
python Main_Combine.py ^
  --whisper-model=base ^
  --perf-log ^
  --skip-reasoning ^
  --max-tts=100 ^
  --cache ^
  --sequential-mode

echo System terminated.
