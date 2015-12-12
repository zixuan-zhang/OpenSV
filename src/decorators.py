#! /usr/bin/env python
#coding=utf-8

from exception import ParamErrorException

def param_length_matcher(func):
    """
        if parameter length not euqal, raise exception
    """
    def _warpper(*args, **kwargs):
        length = len(args)
        for i in range(len(args)-1):
            if not isinstance(args[i], list):
                continue
            if len(args[i]) != len(args[i+1]):
                raise ParamErrorException("Param length not match")
        ret = func(*args, **kwargs)
        return ret
    return _warpper
