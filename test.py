"""
Author:  Cax
File:    test
Project: GAN
Time:    2022/7/3
Des:     
"""
import time

from GAN.utils.metric_logger import MetricLogger


mlogger = MetricLogger()
mlogger.update(a=1)
print(str(mlogger))
mlogger.update(a=2)
print(str(mlogger))
mlogger.update(a=3)
print(str(mlogger))
