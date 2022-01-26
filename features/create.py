import pandas as pd
import numpy as np
import re as re

from base import Feature, get_arguments, generate_features

Feature.dir = "features"

class rev_Mean(Feature):
    def create_features(self):
        self.train['rev_Mean'] = train['rev_Mean']
        self.test['rev_Mean'] = test['rev_Mean']

class mou_Mean(Feature):
    def create_features(self):
        self.train['mou_Mean'] = train['mou_Mean']
        self.test['mou_Mean'] = test['mou_Mean']

class totmrc_Mean(Feature):
    def create_features(self):
        self.train['totmrc_Mean'] = train['totmrc_Mean']
        self.test['totmrc_Mean'] = test['totmrc_Mean']

class da_Mean(Feature):
    def create_features(self):
        self.train['da_Mean'] = train['da_Mean']
        self.test['da_Mean'] = test['da_Mean']

if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('./data/input/train.feather')
    test = pd.read_feather('./data/input/test.feather')

    generate_features(globals(), args.force)