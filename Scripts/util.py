# Small but useful functions

import time

def get_timestamp():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())