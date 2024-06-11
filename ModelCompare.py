import os
import random
from Creator import DiffusionCreator
from PromptEngineering import PromptGenerator, RandomArtistGenerator, RandomImageSizeGenerator
import torch
import random
from PIL import Image


class ModelMatrixComparator:
    def __init__(self, promptList, negativePrompt, modelList) -> None:
        self.sizeGen = RandomImageSizeGenerator(sizeSet='big')
        self.promptGen = PromptGenerator(
            promptList,
            negativePrompt=negativePrompt,
            artistGenerator=RandomArtistGenerator(
                specifiedStyleList=['fineart', 'digipa-high-impact', 'cartoon', 'scribbles'])
        )

        self.modelList = modelList
        self.creatorCache = {}

    def getGenSeedList(self, num):
        seedList = []
        for _ in range(num):
            seedList.append(random.randint(0, 0xffffffff))
        return seedList

    def loadCreator(self, model):
        modelWeightRoot, modelCfgDict = model
        cacheKey = str(model)
        if cacheKey in self.creatorCache.keys():
            creator = self.creatorCache[cacheKey]
        else:
            creator = DiffusionCreator(modelWeightRoot=modelWeightRoot,
                                       # To use local weight you should start with "."
                                       modelCfgDict=modelCfgDict,
                                       defaultDType=torch.float16,
                                       useXformers=True,
                                       )
            self.creatorCache[cacheKey] = creator
        creator.to('cuda')
        return creator

    def unLoadCreator(self, creator):
        creator.to('cpu')

    def genMatrix(self, seedNum=5, outputDir='matrix_imgs'):
        randomSize = self.sizeGen.getSize()

        genArgDict = {
            'height': randomSize[0],
            'width': randomSize[1],
            'num_inference_steps': 40,
            'guidance_scale': 9
        }
        prompt = self.promptGen.getPrompt()
        print('Prompt: ', prompt['prompt'])
        genArgDict.update(prompt)

        seedList = self.getGenSeedList(seedNum)
        allImages = []

        for model in self.modelList:
            creator = self.loadCreator(model)
            imgs = self.genImages(creator, genArgDict, seedList)
            allImages.extend(imgs)
            self.unLoadCreator(creator)

        if 'originalPrompt' in genArgDict.keys():
            prompt = genArgDict['originalPrompt']
        else:
            prompt = genArgDict['prompt']
        outputDir = os.path.join(
            outputDir, os.path.normpath(prompt))
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        self.genImageJiasaw(allImages, genArgDict['width'], genArgDict['height'],
                            seedNum, len(self.modelList),
                            os.path.join(outputDir, '%d.jpg' % seedList[0]))

    def genImageJiasaw(self, imageList, width, height, col, row, outPath):
        to_image = Image.new('RGB', (col * width, row * height))  # 创建一个新图
        # 循环遍历，把每张图片按顺序粘贴到对应位置上
        for y in range(row):
            for x in range(col):
                from_image = imageList[y*col+x]
                to_image.paste(from_image, (x * width, y * height))
        return to_image.save(outPath, quality=90)  # 保存新图

    def genImages(self, creator, genArgDict, seedList):
        imgs = []
        for seed in seedList:
            img = creator.generate(
                '', returnPILImage=True, seed=seed, extraArgDict=genArgDict)
            imgs.append(img)
        return imgs
