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

class ovrmou_Mean(Feature):
    def create_features(self):
        self.train['ovrmou_Mean'] = train['ovrmou_Mean']
        self.test['ovrmou_Mean'] = test['ovrmou_Mean']

class ovrrev_Mean(Feature):
    def create_features(self):
        self.train['ovrrev_Mean'] = train['ovrrev_Mean']
        self.test['ovrrev_Mean'] = test['ovrrev_Mean']

class vceovr_Mean(Feature):
    def create_features(self):
        self.train['vceovr_Mean'] = train['vceovr_Mean']
        self.test['vceovr_Mean'] = test['vceovr_Mean']

class datovr_Mean(Feature):
    def create_features(self):
        self.train['datovr_Mean'] = train['datovr_Mean']
        self.test['datovr_Mean'] = test['datovr_Mean']

class roam_Mean(Feature):
    def create_features(self):
        self.train['roam_Mean'] = train['roam_Mean']
        self.test['roam_Mean'] = test['roam_Mean']

class change_mou(Feature):
    def create_features(self):
        self.train['change_mou'] = train['change_mou']
        self.test['change_mou'] = test['change_mou']

class change_rev(Feature):
    def create_features(self):
        self.train['change_rev'] = train['change_rev']
        self.test['change_rev'] = test['change_rev']

class drop_vce_Mean(Feature):
    def create_features(self):
        self.train['drop_vce_Mean'] = train['drop_vce_Mean']
        self.test['drop_vce_Mean'] = test['drop_vce_Mean']

class drop_dat_Mean(Feature):
    def create_features(self):
        self.train['drop_dat_Mean'] = train['drop_dat_Mean']
        self.test['drop_dat_Mean'] = test['drop_dat_Mean']

class blck_vce_Mean(Feature):
    def create_features(self):
        self.train['blck_vce_Mean'] = train['blck_vce_Mean']
        self.test['blck_vce_Mean'] = test['blck_vce_Mean']

class blck_dat_Mean(Feature):
    def create_features(self):
        self.train['blck_dat_Mean'] = train['blck_dat_Mean']
        self.test['blck_dat_Mean'] = test['blck_dat_Mean']

class unan_vce_Mean(Feature):
    def create_features(self):
        self.train['unan_vce_Mean'] = train['unan_vce_Mean']
        self.test['unan_vce_Mean'] = test['unan_vce_Mean']

class unan_dat_Mean(Feature):
    def create_features(self):
        self.train['unan_dat_Mean'] = train['unan_dat_Mean']
        self.test['unan_dat_Mean'] = test['unan_dat_Mean']

class plcd_vce_Mean(Feature):
    def create_features(self):
        self.train['plcd_vce_Mean'] = train['plcd_vce_Mean']
        self.test['plcd_vce_Mean'] = test['plcd_vce_Mean']

class feature(Feature):
    def create_features(self):
        self.train['feature'] = train['feature']
        self.test['feature'] = test['feature']

class feature(Feature):
    def create_features(self):
        self.train['feature'] = train['feature']
        self.test['feature'] = test['feature']

class feature(Feature):
    def create_features(self):
        self.train['feature'] = train['feature']
        self.test['feature'] = test['feature']

class feature(Feature):
    def create_features(self):
        self.train['feature'] = train['feature']
        self.test['feature'] = test['feature']

class feature(Feature):
    def create_features(self):
        self.train['feature'] = train['feature']
        self.test['feature'] = test['feature']

class feature(Feature):
    def create_features(self):
        self.train['feature'] = train['feature']
        self.test['feature'] = test['feature']

class feature(Feature):
    def create_features(self):
        self.train['feature'] = train['feature']
        self.test['feature'] = test['feature']

class feature(Feature):
    def create_features(self):
        self.train['feature'] = train['feature']
        self.test['feature'] = test['feature']

if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('./data/input/train.feather')
    test = pd.read_feather('./data/input/test.feather')

    generate_features(globals(), args.force)