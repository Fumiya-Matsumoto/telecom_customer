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

class owylis_vce_Mean(Feature):
    def create_features(self):
        self.train['owylis_vce_Mean'] = train['owylis_vce_Mean']
        self.test['owylis_vce_Mean'] = test['owylis_vce_Mean']

class mouowylisv_Mean(Feature):
    def create_features(self):
        self.train['mouowylisv_Mean'] = train['mouowylisv_Mean']
        self.test['mouowylisv_Mean'] = test['mouowylisv_Mean']

class iwylis_vce_Mean(Feature):
    def create_features(self):
        self.train['iwylis_vce_Mean'] = train['iwylis_vce_Mean']
        self.test['iwylis_vce_Mean'] = test['iwylis_vce_Mean']

class mouiwylisv_Mean(Feature):
    def create_features(self):
        self.train['mouiwylisv_Mean'] = train['mouiwylisv_Mean']
        self.test['mouiwylisv_Mean'] = test['mouiwylisv_Mean']

class peak_vce_Mean(Feature):
    def create_features(self):
        self.train['peak_vce_Mean'] = train['peak_vce_Mean']
        self.test['peak_vce_Mean'] = test['peak_vce_Mean']

class peak_dat_Mean(Feature):
    def create_features(self):
        self.train['peak_dat_Mean'] = train['peak_dat_Mean']
        self.test['peak_dat_Mean'] = test['peak_dat_Mean']

class mou_peav_Mean(Feature):
    def create_features(self):
        self.train['mou_peav_Mean'] = train['mou_peav_Mean']
        self.test['mou_peav_Mean'] = test['mou_peav_Mean']

class mou_pead_Mean(Feature):
    def create_features(self):
        self.train['mou_pead_Mean'] = train['mou_pead_Mean']
        self.test['mou_pead_Mean'] = test['mou_pead_Mean']

class opk_vce_Mean(Feature):
    def create_features(self):
        self.train['opk_vce_Mean'] = train['opk_vce_Mean']
        self.test['opk_vce_Mean'] = test['opk_vce_Mean']

class opk_dat_Mean(Feature):
    def create_features(self):
        self.train['opk_dat_Mean'] = train['opk_dat_Mean']
        self.test['opk_dat_Mean'] = test['opk_dat_Mean']

class mou_opkv_Mean(Feature):
    def create_features(self):
        self.train['mou_opkv_Mean'] = train['mou_opkv_Mean']
        self.test['mou_opkv_Mean'] = test['mou_opkv_Mean']

class mou_opkd_Mean(Feature):
    def create_features(self):
        self.train['mou_opkd_Mean'] = train['mou_opkd_Mean']
        self.test['mou_opkd_Mean'] = test['mou_opkd_Mean']

class drop_blk_Mean(Feature):
    def create_features(self):
        self.train['drop_blk_Mean'] = train['drop_blk_Mean']
        self.test['drop_blk_Mean'] = test['drop_blk_Mean']

class attempt_Mean(Feature):
    def create_features(self):
        self.train['attempt_Mean'] = train['attempt_Mean']
        self.test['attempt_Mean'] = test['attempt_Mean']

class complete_Mean(Feature):
    def create_features(self):
        self.train['complete_Mean'] = train['complete_Mean']
        self.test['complete_Mean'] = test['complete_Mean']

class callfwdv_Mean(Feature):
    def create_features(self):
        self.train['callfwdv_Mean'] = train['callfwdv_Mean']
        self.test['callfwdv_Mean'] = test['callfwdv_Mean']

class callwait_Mean(Feature):
    def create_features(self):
        self.train['callwait_Mean'] = train['callwait_Mean']
        self.test['callwait_Mean'] = test['callwait_Mean']

class months(Feature):
    def create_features(self):
        self.train['months'] = train['months']
        self.test['months'] = test['months']

class uniqsubs(Feature):
    def create_features(self):
        self.train['uniqsubs'] = train['uniqsubs']
        self.test['uniqsubs'] = test['uniqsubs']

class actvsubs(Feature):
    def create_features(self):
        self.train['actvsubs'] = train['actvsubs']
        self.test['actvsubs'] = test['actvsubs']

class new_cell(Feature):
    def create_features(self):
        self.train['new_cell'] = train['new_cell']
        self.test['new_cell'] = test['new_cell']

class crclscod(Feature):
    def create_features(self):
        self.train['crclscod'] = train['crclscod']
        self.test['crclscod'] = test['crclscod']

class asl_flag(Feature):
    def create_features(self):
        self.train['asl_flag'] = train['asl_flag']
        self.test['asl_flag'] = test['asl_flag']

class totcalls(Feature):
    def create_features(self):
        self.train['totcalls'] = train['totcalls']
        self.test['totcalls'] = test['totcalls']

class totmou(Feature):
    def create_features(self):
        self.train['totmou'] = train['totmou']
        self.test['totmou'] = test['totmou']

class totrev(Feature):
    def create_features(self):
        self.train['totrev'] = train['totrev']
        self.test['totrev'] = test['totrev']

class adjrev(Feature):
    def create_features(self):
        self.train['adjrev'] = train['adjrev']
        self.test['adjrev'] = test['adjrev']

class adjmou(Feature):
    def create_features(self):
        self.train['adjmou'] = train['adjmou']
        self.test['adjmou'] = test['adjmou']

class adjqty(Feature):
    def create_features(self):
        self.train['adjqty'] = train['adjqty']
        self.test['adjqty'] = test['adjqty']

class avgrev(Feature):
    def create_features(self):
        self.train['avgrev'] = train['avgrev']
        self.test['avgrev'] = test['avgrev']

class avgmou(Feature):
    def create_features(self):
        self.train['avgmou'] = train['avgmou']
        self.test['avgmou'] = test['avgmou']

class avgqty(Feature):
    def create_features(self):
        self.train['avgqty'] = train['avgqty']
        self.test['avgqty'] = test['avgqty']

class avg3mou(Feature):
    def create_features(self):
        self.train['avg3mou'] = train['avg3mou']
        self.test['avg3mou'] = test['avg3mou']

class avg3qty(Feature):
    def create_features(self):
        self.train['avg3qty'] = train['avg3qty']
        self.test['avg3qty'] = test['avg3qty']

class avg3rev(Feature):
    def create_features(self):
        self.train['avg3rev'] = train['avg3rev']
        self.test['avg3rev'] = test['avg3rev']

class avg6mou(Feature):
    def create_features(self):
        self.train['avg6mou'] = train['avg6mou']
        self.test['avg6mou'] = test['avg6mou']

class avg6qty(Feature):
    def create_features(self):
        self.train['avg6qty'] = train['avg6qty']
        self.test['avg6qty'] = test['avg6qty']

class avg6rev(Feature):
    def create_features(self):
        self.train['avg6rev'] = train['avg6rev']
        self.test['avg6rev'] = test['avg6rev']

class prizm_social_one(Feature):
    def create_features(self):
        self.train['prizm_social_one'] = train['prizm_social_one']
        self.test['prizm_social_one'] = test['prizm_social_one']

class area(Feature):
    def create_features(self):
        self.train['area'] = train['area']
        self.test['area'] = test['area']

class dualband(Feature):
    def create_features(self):
        self.train['dualband'] = train['dualband']
        self.test['dualband'] = test['dualband']

class refurb_new(Feature):
    def create_features(self):
        self.train['refurb_new'] = train['refurb_new']
        self.test['refurb_new'] = test['refurb_new']

class hnd_price(Feature):
    def create_features(self):
        self.train['hnd_price'] = train['hnd_price']
        self.test['hnd_price'] = test['hnd_price']

class phones(Feature):
    def create_features(self):
        self.train['phones'] = train['phones']
        self.test['phones'] = test['phones']

class models(Feature):
    def create_features(self):
        self.train['models'] = train['models']
        self.test['models'] = test['models']

class hnd_webcap(Feature):
    def create_features(self):
        self.train['hnd_webcap'] = train['hnd_webcap']
        self.test['hnd_webcap'] = test['hnd_webcap']

class truck(Feature):
    def create_features(self):
        self.train['truck'] = train['truck']
        self.test['truck'] = test['truck']

class rv(Feature):
    def create_features(self):
        self.train['rv'] = train['rv']
        self.test['rv'] = test['rv']

class ownrent(Feature):
    def create_features(self):
        self.train['ownrent'] = train['ownrent']
        self.test['ownrent'] = test['ownrent']

class lor(Feature):
    def create_features(self):
        self.train['lor'] = train['lor']
        self.test['lor'] = test['lor']

class dwlltype(Feature):
    def create_features(self):
        self.train['dwlltype'] = train['dwlltype']
        self.test['dwlltype'] = test['dwlltype']

class marital(Feature):
    def create_features(self):
        self.train['marital'] = train['marital']
        self.test['marital'] = test['marital']

class adults(Feature):
    def create_features(self):
        self.train['adults'] = train['adults']
        self.test['adults'] = test['adults']

class infobase(Feature):
    def create_features(self):
        self.train['infobase'] = train['infobase']
        self.test['infobase'] = test['infobase']

class income(Feature):
    def create_features(self):
        self.train['income'] = train['income']
        self.test['income'] = test['income']

class numbcars(Feature):
    def create_features(self):
        self.train['numbcars'] = train['numbcars']
        self.test['numbcars'] = test['numbcars']

class HHstatin(Feature):
    def create_features(self):
        self.train['HHstatin'] = train['HHstatin']
        self.test['HHstatin'] = test['HHstatin']

class dwllsize(Feature):
    def create_features(self):
        self.train['dwllsize'] = train['dwllsize']
        self.test['dwllsize'] = test['dwllsize']

class forgntvl(Feature):
    def create_features(self):
        self.train['forgntvl'] = train['forgntvl']
        self.test['forgntvl'] = test['forgntvl']

class ethnic(Feature):
    def create_features(self):
        self.train['ethnic'] = train['ethnic']
        self.test['ethnic'] = test['ethnic']

class kid0_2(Feature):
    def create_features(self):
        self.train['kid0_2'] = train['kid0_2']
        self.test['kid0_2'] = test['kid0_2']

class kid3_5(Feature):
    def create_features(self):
        self.train['kid3_5'] = train['kid3_5']
        self.test['kid3_5'] = test['kid3_5']

class kid6_10(Feature):
    def create_features(self):
        self.train['kid6_10'] = train['kid6_10']
        self.test['kid6_10'] = test['kid6_10']

class kid11_15(Feature):
    def create_features(self):
        self.train['kid11_15'] = train['kid11_15']
        self.test['kid11_15'] = test['kid11_15']

class kid16_17(Feature):
    def create_features(self):
        self.train['kid16_17'] = train['kid16_17']
        self.test['kid16_17'] = test['kid16_17']

class creditcd(Feature):
    def create_features(self):
        self.train['creditcd'] = train['creditcd']
        self.test['creditcd'] = test['creditcd']

class eqpdays(Feature):
    def create_features(self):
        self.train['eqpdays'] = train['eqpdays']
        self.test['eqpdays'] = test['eqpdays']

class Customer_ID(Feature):
    def create_features(self):
        self.train['Customer_ID'] = train['Customer_ID']
        self.test['Customer_ID'] = test['Customer_ID']

if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('./data/input/train.feather')
    test = pd.read_feather('./data/input/test.feather')

    generate_features(globals(), args.force)