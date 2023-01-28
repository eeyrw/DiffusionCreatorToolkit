from scipy.stats import entropy
from safetensors.torch import save_file
import argparse
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
import random
import torch
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel
import json
import piexif
import copy
torch.backends.cuda.matmul.allow_tf32 = True


class DiffusionModelWeightCompressor:
    def __init__(self, modelWeightRoot='.',
                 modelDictList=[
                     {'name': 'runwayml/stable-diffusion-v1-5'}],
                 defaultDType=torch.float16) -> None:
        self.modelWeightRoot = modelWeightRoot
        self.modelDictList = modelDictList
        self.defaultDType = defaultDType
        self.randGenerator = torch.Generator()
        self.blendMetaInfoDict = {}
        self.loadModel()

    def loadModel(self):
        self.loadMultiModel(self.modelDictList)

    def loadMultiModel(self, blendParamDictList):
        self.modelUNetList = {}
        for blendParamDict in blendParamDictList:
            modelNameRaw = blendParamDict['name']
            if modelNameRaw[0] == '.':
                modelName = os.path.join(
                    self.modelWeightRoot, modelNameRaw[1:], 'unet')
            else:
                modelName = modelNameRaw

            unet = UNet2DConditionModel.from_pretrained(
                modelName,
                subfolder='unet',
                cache_dir=self.modelWeightRoot,
                torch_dtype=self.defaultDType)

            self.modelUNetList[modelNameRaw] = unet

    def diffModelWeight(self):
        firstModelName = self.modelDictList[0]['name']
        secondModelName = self.modelDictList[1]['name']
        diffWeightDict = {}
        print('Total %d layers' %
              len(self.modelUNetList[firstModelName].state_dict().keys()))
        for weightKey in self.modelUNetList[firstModelName].state_dict().keys():
            diffWeightDict[weightKey] = self.modelUNetList[secondModelName].state_dict(
            )[weightKey] - self.modelUNetList[firstModelName].state_dict()[weightKey]
        return diffWeightDict

    def saveNormAndBiasWeightOnly(self, modelWeightDict):
        targetDict = {}
        targetDict2 = {}
        for k, v in modelWeightDict.items():
            if '.bias' in k or '.norm' in k:
                targetDict[k] = v
                print('shape:%s, name: %s' % (v.shape, k))
            else:
                targetDict2[k] = v
        save_file(targetDict, 'NormAndBiasWeightOnly.safetensors')
        save_file(targetDict2, 'ExNormAndBiasWeightOnly.safetensors')

    def estimateQuantRange(self, modelWeight):
        # https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
        # Input: FP32 histogram H with 2048 bins: bin[ 0 ], …, bin[ 2047 ]
        # For i in range( 128 , 2048 ):
        # reference_distribution_P = [ bin[ 0 ] , ..., bin[ i-1 ] ] // take first ‘ i ‘ bins from H
        # outliers_count = sum( bin[ i ] , bin[ i+1 ] , … , bin[ 2047 ] )
        # reference_distribution_P[ i-1 ] += outliers_count
        # P /= sum(P) // normalize distribution P
        # candidate_distribution_Q = quantize [ bin[ 0 ], …, bin[ i-1 ] ] into 128 levels // explained later
        # expand candidate_distribution_Q to ‘ i ’ bins // explained later
        # Q /= sum(Q) // normalize distribution Q
        # divergence[ i ] = KL_divergence( reference_distribution_P, candidate_distribution_Q)
        # End For
        # Find index ‘m’ for which divergence[ m ] is minimal
        # threshold = ( m + 0.5 ) * ( width of a bin )

        quantSteps = 8
        totalBins = 1024
        minimalDv = 10
        minimalDvI = 0
        modelWeight = modelWeight.to(torch.float32)
        hist, histBinEdges = torch.histogram(
            modelWeight, totalBins, range=(0, max(modelWeight)))
        for i in range(quantSteps, totalBins):
            reference_distribution_P = torch.clone(hist[:i])
            reference_distribution_P_RAW = hist[:i]
            outliers_count = sum(hist[i:])
            reference_distribution_P[i-1] += outliers_count
            candi_Q_l = torch.linspace(
                0, quantSteps-1, quantSteps, dtype=torch.int)
            rangeStartIndices = torch.round(candi_Q_l/(quantSteps-1)
                                            * (i-1), decimals=0).int()
            rangeEndIndices = rangeStartIndices + \
                torch.diff(rangeStartIndices, append=torch.tensor(
                    [rangeStartIndices[-1]+1])).int()
            candi_Q_exp = torch.zeros_like(reference_distribution_P)
            for sta, end in zip(rangeStartIndices, rangeEndIndices):
                selected = reference_distribution_P[sta:end]
                selectedNonZero = (selected != 0).int()
                num_nonzero = torch.count_nonzero(selectedNonZero)
                if num_nonzero > 0:
                    candi_Q_exp[sta:end] = sum(
                        selected)/num_nonzero
                    candi_Q_exp[sta:end] = candi_Q_exp[sta:end]*selectedNonZero
                else:
                    candi_Q_exp[sta:end] = 0

            candi_Q_exp_norm = candi_Q_exp/sum(candi_Q_exp)
            reference_distribution_P_RAW_norm = reference_distribution_P_RAW / \
                sum(reference_distribution_P_RAW)

            diver = entropy(reference_distribution_P_RAW_norm.numpy(),
                            candi_Q_exp_norm.numpy())
            # print('%d: %f' % (i, diver))
            if diver < minimalDv:
                minimalDv = diver
                minimalDvI = i

        return histBinEdges[minimalDvI]
        # print('Mdv:%f,MdvI:%d,ClipPoint:%f' % (minimalDv, minimalDvI,histBinEdges[minimalDvI]))

    def visualModelWeight(self, modelWeightDict):
        # ax.set_title('PDF of %s' % outputJson)
        subSize = 2
        keyList = list(modelWeightDict.keys())
        modelWeightDictKeySubList = [keyList[x:x+subSize]
                                     for x in range(0, len(keyList), subSize)]

        for i, modelWeightDictKeys in enumerate(modelWeightDictKeySubList):
            chartNum = len(modelWeightDictKeys)
            fig = plt.figure(figsize=(8, chartNum*4))
            for j, weightKey in enumerate(modelWeightDictKeys):
                if not ('.bias' in weightKey or '.norm' in weightKey):
                    edge = self.estimateQuantRange(
                        torch.flatten(modelWeightDict[weightKey]))
                ax = fig.add_subplot(chartNum, 1, j+1)
                weights = torch.flatten(modelWeightDict[weightKey]).numpy()
                ns, edgeBin, patches = ax.hist(
                    weights, bins=200, label='%s-%d' % (weightKey, len(weights)))
                if not ('.bias' in weightKey or '.norm' in weightKey):
                    ax.vlines(edge, 0, max(ns), colors='red',
                              label='Edge:%f' % edge)
                ax.legend(prop={'size': 10})
            plt.savefig("mygraph_%d.png" % i)
            plt.close(fig)

    def to(self, device):
        self.pipe.to(device)
        self.randGenerator = torch.Generator(device=device)


if __name__ == "__main__":

    modelDictList = (
        {'name': '.sd1.5'},
        {'name': '.TFC_200', 'factor': 0.3},
    )
    creator = DiffusionModelWeightCompressor(modelWeightRoot=r'F:\StableDiffusionWeight',
                                             modelDictList=modelDictList,  # To use local weight you should start with "."
                                             defaultDType=torch.float16,
                                             )
    diffWeight = creator.diffModelWeight()
    creator.visualModelWeight(diffWeight)
    # creator.saveNormAndBiasWeightOnly(diffWeight)
    # creator.estimateQuantRange(torch.normal(0, 3, size=(4096,)))
