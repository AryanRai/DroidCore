@echo off
echo Running in debug mode with memory tracking...

REM Set debug environment variables
set PYTHONDEVMODE=1
set PYTHONASYNCIODEBUG=1
set PYTHONFAULTHANDLER=1
set CUDA_LAUNCH_BLOCKING=1

REM Memory monitoring settings
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8
set CUDA_VISIBLE_DEVICES=0

REM Launch with debug settings
python Main_Combine.py ^
  --whisper-model=base ^
  --perf-log ^
  --perf-interval=5 ^
  --memory-file=debug_memory.json ^
  --max-memories=200 ^
  --skip-reasoning ^
  --cache ^
  --max-tts=100 ^
  --timeout=30 ^
  --debug

REM Start memory monitor in separate window
start "Memory Monitor" cmd /c "python memory_monitor.py --interval 1 --plot"

echo Debug session terminated.
