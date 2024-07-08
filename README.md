## Requirements: 

torch,transformers,accelerate,memory-profiler,matplotlib,line-profiler,bitsandbytes,pynvml,huggingface-hub,
flash_attn ( optional )

## Monitoring:

Terminal 1 : `watch free -h` 

Terminal 2 : `watch nvidia-smi`

code : memory-profiler, line-profiler, working/utils.py

### Commands:

```
# Run both profilers seperately

## memory profiler
python -m memory_profiler main.py
mprof run --python main.py
mprof plot -o output.png

## line profiler
LINE_PROFILE=1 python test.py 
```
