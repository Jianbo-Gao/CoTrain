#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: gaojianbo@pku.edu.cn

import time

class Timer:
    def __init__(self, info=""):
        self.start_time=time.time()
        self.info=info

    def start(self):
        self.start_time=time.time()
        print "Start %s" % self.info

    def finish(self):
        during_time=time.time()-self.start_time
        print "Finish %s (Time: %.2fs)" % (self.info, during_time)
