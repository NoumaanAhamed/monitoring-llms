## Requirements: 

torch,transformers,accelerate,memory-profiler,matplotlib,line-profiler


## Monitoring:

Terminal 1 : `watch free -h` 

Terminal 2 : `watch nvidia-smi`

code : memory-profiler, line-profiler

### Commands:


```
python -m memory_profiler main.py
mprof run --python main.py
mprof plot -o output.png

# Run both profilers seperately

LINE_PROFILE=1 python test.py 
```
