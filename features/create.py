import pandas as pd
import numpy as np
import re as re

from base import Feature, get_arguments, generate_features

Feature.dir = "features"

# class feature(Feature):
#     def create_features(self):
#         self.train['feature'] = train['feature']
#         self.test['feature'] = test['feature']

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

class plcd_dat_Mean(Feature):
    def create_features(self):
        self.train['plcd_dat_Mean'] = train['plcd_dat_Mean']
        self.test['plcd_dat_Mean'] = test['plcd_dat_Mean']

class recv_vce_Mean(Feature):
    def create_features(self):
        self.train['recv_vce_Mean'] = train['recv_vce_Mean']
        self.test['recv_vce_Mean'] = test['recv_vce_Mean']

class recv_sms_Mean(Feature):
    def create_features(self):
        self.train['recv_sms_Mean'] = train['recv_sms_Mean']
        self.test['recv_sms_Mean'] = test['recv_sms_Mean']

class comp_vce_Mean(Feature):
    def create_features(self):
        self.train['comp_vce_Mean'] = train['comp_vce_Mean']
        self.test['comp_vce_Mean'] = test['comp_vce_Mean']

class comp_dat_Mean(Feature):
    def create_features(self):
        self.train['comp_dat_Mean'] = train['comp_dat_Mean']
        self.test['comp_dat_Mean'] = test['comp_dat_Mean']

class custcare_Mean(Feature):
    def create_features(self):
        self.train['custcare_Mean'] = train['custcare_Mean']
        self.test['custcare_Mean'] = test['custcare_Mean']

class ccrndmou_Mean(Feature):
    def create_features(self):
        self.train['ccrndmou_Mean'] = train['ccrndmou_Mean']
        self.test['ccrndmou_Mean'] = test['ccrndmou_Mean']

class cc_mou_Mean(Feature):
    def create_features(self):
        self.train['cc_mou_Mean'] = train['cc_mou_Mean']
        self.test['cc_mou_Mean'] = test['cc_mou_Mean']

class inonemin_Mean(Feature):
    def create_features(self):
        self.train['inonemin_Mean'] = train['inonemin_Mean']
        self.test['inonemin_Mean'] = test['inonemin_Mean']

class threeway_Mean(Feature):
    def create_features(self):
        self.train['threeway_Mean'] = train['threeway_Mean']
        self.test['threeway_Mean'] = test['threeway_Mean']

class mou_cvce_Mean(Feature):
    def create_features(self):
        self.train['mou_cvce_Mean'] = train['mou_cvce_Mean']
        self.test['mou_cvce_Mean'] = test['mou_cvce_Mean']

class mou_cdat_Mean(Feature):
    def create_features(self):
        self.train['mou_cdat_Mean'] = train['mou_cdat_Mean']
        self.test['mou_cdat_Mean'] = test['mou_cdat_Mean']

class mou_rvce_Mean(Feature):
    def create_features(self):
        self.train['mou_rvce_Mean'] = train['mou_rvce_Mean']
        self.test['mou_rvce_Mean'] = test['mou_rvce_Mean']

if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('./data/input/train.feather')
    test = pd.read_feather('./data/input/test.feather')

    generate_features(globals(), args.force)