#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: gaojianbo@pku.edu.cn

import os, sys

ROOT_DIR=os.path.abspath(os.path.dirname(sys.argv[0]))


DATA_DIR=os.path.join(ROOT_DIR, "data")
TRAIN_DATA_DIR="Train_EN"
UNLABEL_DATA_DIR="Unlabel_CN"
VALIDATION_DATA_DIR="Train_CN"
TEST_DATA_DIR="Test_CN"

TRAIN_DATA_BOOK=os.path.join(DATA_DIR,TRAIN_DATA_DIR,"book","train.data")
TRAIN_DATA_DVD=os.path.join(DATA_DIR,TRAIN_DATA_DIR,"dvd","train.data")
TRAIN_DATA_MUSIC=os.path.join(DATA_DIR,TRAIN_DATA_DIR,"music","train.data")

UNLABEL_DATA_BOOK=os.path.join(DATA_DIR,UNLABEL_DATA_DIR,"book","unlabel.data")
UNLABEL_DATA_DVD=os.path.join(DATA_DIR,UNLABEL_DATA_DIR,"dvd","unlabel.data")
UNLABEL_DATA_MUSIC=os.path.join(DATA_DIR,UNLABEL_DATA_DIR,"music","unlabel.data")

VALIDATION_DATA_BOOK=os.path.join(DATA_DIR,VALIDATION_DATA_DIR,"book","sample.data")
VALIDATION_DATA_DVD=os.path.join(DATA_DIR,VALIDATION_DATA_DIR,"dvd","sample.data")
VALIDATION_DATA_MUSIC=os.path.join(DATA_DIR,VALIDATION_DATA_DIR,"music","sample.data")

TEST_DATA_BOOK=os.path.join(DATA_DIR,TEST_DATA_DIR,"book","testResult.data")
TEST_DATA_DVD=os.path.join(DATA_DIR,TEST_DATA_DIR,"dvd","testResult.data")
TEST_DATA_MUSIC=os.path.join(DATA_DIR,TEST_DATA_DIR,"music","testResult.data")


CACHE_DIR=os.path.join(ROOT_DIR,"cache")
CACHE_EN_TO_CN=os.path.join(CACHE_DIR,"en_to_cn.json")
CACHE_CN_TO_EN=os.path.join(CACHE_DIR,"cn_to_en.json")
CACHE_JIEBA=os.path.join(CACHE_DIR,"jieba.json")
ITERATE_RESULT=os.path.join(CACHE_DIR,"result.json")

