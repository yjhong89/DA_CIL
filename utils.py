import tensorflow as tf
import numpy as np
import os
import configparser
import dataset_utils
import inspect

class MyConfigParser(configparser.ConfigParser):
    def getlist(self, section, option):
        value = self.get(section, option)
        return list(filter(None, value.split(',')))

    def getlistint(self, section, option):
        return [int(x) for x in self.getlist(section, option)]
    
def load_config(config, ini):
    config_file = os.path.expanduser(ini)
    config.read(config_file)


def str2bool(v):
    if v.lower() in ('true', 't', 'y', 'yes'):
        return True
    elif v.lower() in ('false', 'f', 'n', 'no'):
        return False
    else:
        raise ValueError('%s is not supported' % v)
