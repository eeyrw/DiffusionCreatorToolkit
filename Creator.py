from lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
import csv
import random
import torch
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel
import json
import piexif
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
                 defaultModel='runwayml/stable-diffusion-v1-5',
                 defaultDType=torch.bfloat16,
                 useXformers=False) -> None:
        self.modelWeightRoot = modelWeightRoot
        self.defaultModel = defaultModel
        self.defaultDType = defaultDType
        self.useXformers = useXformers
        self.randGenerator = torch.Generator()
        self.loadModel(self.defaultModel)

    def loadModel(self, modelName):
        if modelName[0] == '.':
            modelName = os.path.join(self.modelWeightRoot, modelName[1:])
        self.pipe = StableDiffusionLongPromptWeightingPipeline.from_pretrained(
            modelName, cache_dir=self.modelWeightRoot,
            requires_safety_checker=False,
            feature_extractor=None,
            safety_checker=None,
            torch_dtype=self.defaultDType,
            text_encoder=CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14-336", cache_dir=self.modelWeightRoot, torch_dtype=self.defaultDType)
        )
        if self.useXformers:
            self.pipe.enable_xformers_memory_efficient_attention()

    def loadBlendModel(self, blendParamDictList):
        self.modelUNetList = {}
        for blendParamDict in blendParamDictList:
            modelNameRaw = blendParamDict['name']
            if modelNameRaw[0] == '.':
                modelName = os.path.join(
                    self.modelWeightRoot, modelNameRaw[1:], 'unet')
            else:
                raise RuntimeError('Only support local model weights.')
            self.modelUNetList[modelNameRaw] = UNet2DConditionModel.from_pretrained(
                modelName, cache_dir=self.modelWeightRoot)

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

        self.pipe.unet.load_state_dict(tempStateDict)

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
            'model': self.defaultModel
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
