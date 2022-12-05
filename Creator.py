from lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
import csv
import random
import torch
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel
import json
import piexif
import copy
torch.backends.cuda.matmul.allow_tf32 = True


class RandomArtistGenerator:
    def __init__(self, artistDatabase='assets/artists.csv') -> None:
        with open(artistDatabase, newline='', encoding='utf8') as csvfile:
            reader = csv.DictReader(csvfile)
            artistDictList = list(reader)
            self.artistDictList = [artistDict for artistDict in artistDictList if float(
                artistDict['score']) > 0.7]

    def getArtist(self):
        return random.choice(self.artistDictList)['artist']


class RandomImageSizeGenerator:
    def __init__(self, sizeSet='big', customSizeList=None) -> None:
        self.sizeListDict = {
            'big': [(1088, 832), (1088, 768), (1088, 640), (896, 896), (1088, 512)],
            'small': [(768, 768), (512, 512), (768, 640), (768, 512)]
        }
        if customSizeList:
            self.sizeList = customSizeList
        else:
            self.sizeList = self.sizeListDict[sizeSet]

    def getSize(self):
        size = random.choice(self.sizeList)
        if random.random() > 0.5:
            size = (size[1], size[0])
        return size


class PromptGenerator:
    def __init__(self, promptList, negativePrompt=None, artistGenerator=None) -> None:
        self.artistGenerator = artistGenerator
        self.promptList = promptList
        self.negativePrompt = negativePrompt

    def getPrompt(self):
        prompt = random.choice(self.promptList)
        if self.artistGenerator:
            prompt = prompt+',by artist '+self.artistGenerator.getArtist()
        return {
            'prompt': prompt,
            'negative_prompt': self.negativePrompt
        }


class DiffusionCreator:
    def __init__(self, modelWeightRoot='.',
                 modelDictList=[{'name':'runwayml/stable-diffusion-v1-5','factor':1}],
                 defaultDType=torch.float16,
                 useXformers=False) -> None:
        self.modelWeightRoot = modelWeightRoot
        self.modelDictList = modelDictList
        self.defaultDType = defaultDType
        self.useXformers = useXformers
        self.randGenerator = torch.Generator()
        self.loadModel()

    def loadModel(self):
        self.loadMultiModel(self.modelDictList)
        baseModelName = self.modelDictList[0]['name']
        tempUNet = copy.deepcopy(self.modelUNetList[baseModelName])
        if len(self.modelDictList)>1:
            unetWeight = self.blendModel(self.modelDictList)
            tempUNet.load_state_dict(unetWeight)
        
        if baseModelName[0] == '.':
            baseModelName = os.path.join(self.modelWeightRoot, baseModelName[1:])

        self.pipe = StableDiffusionLongPromptWeightingPipeline.from_pretrained(
            baseModelName, cache_dir=self.modelWeightRoot,
            unet = tempUNet,
            requires_safety_checker=False,
            feature_extractor=None,
            safety_checker=None,
            torch_dtype=self.defaultDType,
            text_encoder=CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14-336", cache_dir=self.modelWeightRoot, torch_dtype=self.defaultDType)
        )
        if self.useXformers:
            self.pipe.enable_xformers_memory_efficient_attention()



    def loadMultiModel(self, blendParamDictList):
        self.modelUNetList = {}
        for blendParamDict in blendParamDictList:
            modelNameRaw = blendParamDict['name']
            if modelNameRaw[0] == '.':
                modelName = os.path.join(
                    self.modelWeightRoot, modelNameRaw[1:], 'unet')
                self.modelUNetList[modelNameRaw] = UNet2DConditionModel.from_pretrained(
                        modelName,
                         cache_dir=self.modelWeightRoot,
                         torch_dtype=self.defaultDType)                
            else:
                modelName = modelNameRaw
                pipe = StableDiffusionPipeline.from_pretrained(
                            modelName, cache_dir=self.modelWeightRoot,
                            requires_safety_checker=False,
                            feature_extractor=None,
                            safety_checker=None,
                            torch_dtype=self.defaultDType,
                            text_encoder=None
                        )    

                self.modelUNetList[modelNameRaw] = pipe.unet
                del pipe  
                          


    def blendModel(self, blendParamDictList):
        firstModelParamDict = blendParamDictList[0]
        firstModelName = firstModelParamDict['name']
        firstModelFactor = firstModelParamDict['factor']

        tempStateDict = {}

        for weightKey, weightTensor in self.modelUNetList[firstModelName].state_dict().items():
            tempStateDict[weightKey] = weightTensor*firstModelFactor

        for modelParamDict in blendParamDictList[1:]:
            modelName = modelParamDict['name']
            modelFactor = modelParamDict['factor']
            for weightKey, weightTensor in tempStateDict.items():
                tempStateDict[weightKey] += self.modelUNetList[modelName].state_dict()[
                    weightKey]*modelFactor

        return tempStateDict
    
    def blendInRuntime(self,blendParamDictList):
        self.pipe.unet.load_state_dict(self.blendModel(blendParamDictList))

    def getExif(self, jsonDict):
        # https://stackoverflow.com/questions/52729428/how-to-write-custom-metadata-into-jpeg-with-python/63400376#63400376
        data = json.dumps(jsonDict).encode(encoding='utf8')
        exif_ifd = {piexif.ExifIFD.MakerNote: data}

        exif_dict = {"0th": {}, "Exif": exif_ifd, "1st": {},
                     "thumbnail": None, "GPS": {}}
        exif_dat = piexif.dump(exif_dict)
        return exif_dat

    def generate(self, prompt, outputDir='./imgs', seed=None, extraArgDict={}):
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

        if seed is None:
            seed = self.randGenerator.seed()
            self.randGenerator.manual_seed(seed)
        else:
            self.randGenerator.manual_seed(seed)

        if 'prompt' in extraArgDict.keys():
            prompt = extraArgDict['prompt']
        genMetaInfoDict = {
            'seed': seed,
            'prompt': prompt,
            'model': self.modelDictList
        }

        argDict = {
            'prompt': prompt,
            'height': 512,
            'width': 512,
            'num_inference_steps': 50,
            'guidance_scale': 7.5
        }

        argDict.update(extraArgDict)
        genMetaInfoDict.update(argDict)

        image = self.pipe(generator=self.randGenerator,
                          **argDict
                          ).images[0]

        exif_dat = self.getExif(genMetaInfoDict)
        image.save(os.path.join(outputDir, '%d.jpg' %
                   seed), quality=90, exif=exif_dat)

    def to(self, device):
        self.pipe.to(device)
        self.randGenerator = torch.Generator(device=device)


# example
# from Creator import DiffusionCreator,RandomArtistGenerator,RandomImageSizeGenerator,PromptGenerator
# import torch

# if __name__ == "__main__":
#     modelDictList = ({'name': 'runwayml/stable-diffusion-v1-5', 'factor': 0.4},
#                           {'name': 'Linaqruf/anything-v3.0', 'factor': 0.6})
#     creator = DiffusionCreator(modelWeightRoot=r'../StableDiffusionWeight',
#                                modelDictList=modelDictList,  # To use local weight you should start with "."
#                                defaultDType=torch.float16,
#                                useXformers=True)

#     creator.to('cuda')
#     artistGen = RandomArtistGenerator()
#     sizeGen = RandomImageSizeGenerator(sizeSet='big')
#     promptGen = PromptGenerator(
#         [  
#             'illustration of beautiful huge dahlia garden'
#         ],
#         negativePrompt=None,#'closed eyes,slanted eyes,ugly,Polydactyly,handicapped,extra fingers,fused fingers,poorly drawn hands,extra legs,one leg,woman underwear,low quality,low res,blurry,draft,text,watermark,signature,two heads,mutated hands,mutation,deformed, bad anatomy, bad proportions,too many fingers,morbid, mutilated, extra limbs,disfigured,missing arms,missing legs,extra arms,malformed limbs',
#         artistGenerator=RandomArtistGenerator()
#     )

#     while True:
#         randomSize = sizeGen.getSize()
#         randomArtist = artistGen.getArtist()

#         genArgDict = {
#             'height': randomSize[0],
#             'width': randomSize[1],
#             'num_inference_steps': 60,
#             'guidance_scale': 7.5
#         }
#         genArgDict.update(promptGen.getPrompt())
#         creator.generate(
#             '', extraArgDict=genArgDict)