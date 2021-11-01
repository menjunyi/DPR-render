import argparse
import os


class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self):
        self.parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
        self.parser.add_argument('--coarseModel', type=str, help='model name for coarse network')
        self.parser.add_argument('--coarseModel_load', type=str, default='', help='saved model for coarse network')
        self.parser.add_argument('--fineModel', type=str, help='model name for fine network')
        self.parser.add_argument('--fineModel_load', type=str, default='', help='saved model for fine network')
        self.parser.add_argument('--detailModel', type=str, help='model name for detail network')
        self.parser.add_argument('--detailModel_load', type=str, default='', help='saved model for detail network')
        self.parser.add_argument('--D_coarseModel', type=str, help='discriminator model name for coarse network')
        self.parser.add_argument('--D_coarseModel_load', type=str, default='', help='saved discriminator model for coarse network')
        self.parser.add_argument('--D_fineModel', type=str, help='discriminator model name for fine network')
        self.parser.add_argument('--D_fineModel_load', type=str, default='', help='discriminator saved model for fine network')
        self.parser.add_argument('--D_detailModel', type=str, help='discriminator model name for detail network')
        self.parser.add_argument('--D_detailModel_load', type=str, default='', help='discriminator saved model for detail network')
        self.parser.add_argument('--print_freq', type=int, default=20, help='frequency of showing training results')
        self.parser.add_argument('--lr', type=float, default= 0.001, help='learnig rate')
        self.parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
        self.parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
        self.parser.add_argument('--coarse_nChannel', type=int, default=27, help='output channel for coarse network')
        self.parser.add_argument('--fine_albedoChannel', type=int, default=6, help='input channel for albedo in fine network')
        self.parser.add_argument('--fine_normalChannel', type=int, default=6, help='input channel for normal in fine network')
        self.parser.add_argument('--fine_lightingChannel', type=int, default=6, help='input channel for lighting in fine network')
        self.parser.add_argument('--detail_albedoChannel', type=int, default=6, help='input channel for albedo in detail network')
        self.parser.add_argument('--detail_normalChannel', type=int, default=6, help='input channel for normal in detail network')
        self.parser.add_argument('--detail_lightingChannel', type=int, default=6, help='input channel for lighting in detail network')
        self.parser.add_argument('--imageSize', type=int, default=64, help='image size')
        self.parser.add_argument('--savePath', type=str, default='.', help='savePath')

