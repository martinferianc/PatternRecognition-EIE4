import time
import os
import psutil

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_percent()
