#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: gaojianbo@pku.edu.cn

from setting import *
import json, time, random

import googletrans


class Translator:
    # Keep a translate cache.
    # Search in translate cache first, in roder to reduce google search.
    def __init__(self):
        self.en_to_cn_cache={}
        self.cn_to_en_cache={}
        self.__read_cache()
        self.translator = googletrans.Translator(service_urls=['translate.google.cn'])

        # Save cache every 10 times search
        self.search_cal=1

    def __del__(self):
        self.__write_cache()

    def __read_cache(self):
        # Read cache from file
        with open(CACHE_EN_TO_CN, 'r') as cachefile:
            try:
                self.en_to_cn_cache=json.loads(cachefile.read())
            except Exception, e:
                print "Read Cache ERROR"
                print e

        with open(CACHE_CN_TO_EN, 'r') as cachefile:
            try:
                self.cn_to_en_cache=json.loads(cachefile.read())
            except Exception, e:
                print "Read Cache ERROR"
                print e

    def __write_cache(self):
        # Write cache into file
        with open(CACHE_EN_TO_CN, 'w') as cachefile:
            try:
                data=json.dumps(self.en_to_cn_cache)
                cachefile.write(data)
            except Exception, e:
                print "Write Cache ERROR"
                print e
        with open(CACHE_CN_TO_EN, 'w') as cachefile:
            try:
                data=json.dumps(self.cn_to_en_cache)
                cachefile.write(data)
            except Exception, e:
                print "Write Cache ERROR"
                print e

    def __save_cache(self, cache, src, dst):
        cache[src]=dst
        self.search_cal+=1
        if random.randint(1,50)==5:
            self.__write_cache()
            time.sleep(random.randint(1,3))

    def __google(self, str, type):
        str_list=str.splitlines()
        result_str=""
        for sub_str in str_list:
            try:
                if len(sub_str)>0:
                    if type=="en_to_cn":
                        result_str += self.translator.translate(sub_str,src='en',dest='zh-CN').text
                    elif type=="cn_to_en":
                        result_str += self.translator.translate(sub_str,src='zh-CN',dest='en').text
            except Exception, e:
                period=""
                if type=="en_to_cn":
                    period="."
                elif type=="cn_to_en":
                    period=u"。"

                try:
                    sub_str_list=sub_str.split(period)
                except Exception, e:
                    print sub_str
                    print e
                    continue

                for sub_sub_str in sub_str_list:
                    try:
                        if len(sub_str)>0:
                            if type=="en_to_cn":
                                result_str += self.translator.translate(sub_sub_str,src='en',dest='zh-CN').text
                            elif type=="cn_to_en":
                                result_str += self.translator.translate(sub_sub_str,src='zh-CN',dest='en').text
                    except Exception, e:
                        print sub_sub_str
                        print e


        print "Searched %d text using Google Translate" % self.search_cal
        return result_str


    def en_to_cn(self, en):
        if not self.en_to_cn_cache.has_key(en):
            cn = self.__google(en, "en_to_cn")
            self.__save_cache(self.en_to_cn_cache, en, cn)
        return self.en_to_cn_cache[en]

    def cn_to_en(self, cn):
        if not self.cn_to_en_cache.has_key(cn):
            en = self.__google(cn, "cn_to_en")
            self.__save_cache(self.cn_to_en_cache, cn, en)
        return self.cn_to_en_cache[cn]
