@echo off
echo Running with performance profiling enabled...
python Main_Combine.py --perf-log --perf-interval 10 %*
echo Performance log saved to performance_metrics.log
