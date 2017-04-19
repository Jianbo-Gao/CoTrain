#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: gaojianbo@pku.edu.cn

import xml.dom.minidom
import copy

class XMLParser:

    def __get_attr(self, item):
        item_data={}
        attr_list=['review_id','summary','text','category','polarity']
        for attr in attr_list:
            nodes=item.getElementsByTagName(attr)
            if len(nodes)>0 and len(nodes[0].childNodes)>0:
                item_data[attr]=nodes[0].childNodes[0].nodeValue
        return item_data

    def parse(self, filename):
        try:
            dom=xml.dom.minidom.parse(filename)
        except Exception, e:
            print e
            print filename
            exit()
        review=dom.documentElement
        items=review.getElementsByTagName("item")
        data=[]
        for item in items:
            data.append(copy.deepcopy(self.__get_attr(item)))
        return data




