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
                print('shape:%s, name: %s'%(v.shape,k))
            else:
                targetDict2[k] = v
        save_file(targetDict, 'NormAndBiasWeightOnly.safetensors')
        save_file(targetDict2, 'ExNormAndBiasWeightOnly.safetensors')

    def visualModelWeight(self, modelWeightDict):
        # ax.set_title('PDF of %s' % outputJson)
        subSize = 100
        keyList = list(modelWeightDict.keys())
        modelWeightDictKeySubList = [keyList[x:x+subSize]
                                     for x in range(0, len(keyList), subSize)]

        for i, modelWeightDictKeys in enumerate(modelWeightDictKeySubList):
            chartNum = len(modelWeightDictKeys)
            fig = plt.figure(figsize=(8, chartNum*4))
            for j, weightKey in enumerate(modelWeightDictKeys):
                ax = fig.add_subplot(chartNum, 1, j+1)
                weights = torch.flatten(modelWeightDict[weightKey]).numpy()
                ns, edgeBin, patches = ax.hist(
                    weights, bins=200, label='%s-%d' % (weightKey, len(weights)))
                ax.legend(prop={'size': 10})
            plt.savefig("mygraph_%d.png" % i)

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
    # creator.visualModelWeight(diffWeight)
    creator.saveNormAndBiasWeightOnly(diffWeight)
