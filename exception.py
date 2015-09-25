#! /usr/bin/env python
#coding=utf-8

class ParamErrorException(Exception):
    """
    Parameter Illegal
    """
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return self.message
