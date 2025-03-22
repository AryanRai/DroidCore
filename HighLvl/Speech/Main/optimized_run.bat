@echo off
echo Running optimized configuration...

REM Set environment variables for better performance
set OMP_NUM_THREADS=4
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

REM Launch with optimized settings
python Main_Combine.py --whisper-model=base --perf-log --skip-reasoning --parallel --cache

echo System terminated.
