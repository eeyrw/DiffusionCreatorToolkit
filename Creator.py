import random
import torch
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel
import json
import piexif
import copy
torch.backends.cuda.matmul.allow_tf32 = True
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import diffusers

# {
#     'vae': '.ARTMAN_HUGE_144000',
#     'textEncoder':'.ARTMAN_HUGE_144000',
#     'models':[{'name':'.ARTMAN_HUGE_144000','factor':0.5},]
#     }
class DiffusionCreator:
    def __init__(self, modelWeightRoot='.',
                 modelCfgDict=[
                     {'name': 'runwayml/stable-diffusion-v1-5', 'factor': 1}],
                 defaultDType=torch.float16,
                 useXformers=False,
                 loadMode='blend') -> None:
        self.modelWeightRoot = modelWeightRoot
        self.modelCfgDict = modelCfgDict
        self.defaultDType = defaultDType
        self.useXformers = useXformers
        self.randGenerator = torch.Generator()
        self.blendMetaInfoDict = {}
        self.loadModel()

    def parseModelPath(self,modelPath,modelWeightRoot):
        if modelPath[0] == '.':
            modelPath = os.path.join(
                modelWeightRoot, modelPath[1:])   
        return modelPath
         
    def loadModel(self):
        self.loadMultiModel(self.modelCfgDict['models'])
        baseModelName = self.modelCfgDict['models'][0]['name']
        tempUNet = copy.deepcopy(self.modelUNetList[baseModelName])
        if len(self.modelCfgDict['models']) > 1:
            unetWeight = self.blendModel(self.modelCfgDict['models'])
            tempUNet.load_state_dict(unetWeight)

        baseModelName = self.parseModelPath(baseModelName,self.modelWeightRoot)

        pipeArgDict = {}

        from lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline

        if 'vae' in self.modelCfgDict.keys():
            vaePath = self.parseModelPath(self.modelCfgDict['vae'],self.modelWeightRoot)
            pipeArgDict['vae'] = AutoencoderKL.from_pretrained(vaePath,subfolder = 'vae', cache_dir = self.modelWeightRoot)

        if 'textEncoder' in self.modelCfgDict.keys():
            textEncoderPath = self.parseModelPath(self.modelCfgDict['textEncoder'],self.modelWeightRoot)
            pipeArgDict['text_encoder'] = CLIPTextModel.from_pretrained(textEncoderPath,subfolder = 'text_encoder', cache_dir = self.modelWeightRoot)
            pipeArgDict['tokenizer'] = CLIPTokenizer.from_pretrained(textEncoderPath,subfolder = 'tokenizer', cache_dir = self.modelWeightRoot)


            
        self.pipe = StableDiffusionLongPromptWeightingPipeline.from_pretrained(
            baseModelName, cache_dir=self.modelWeightRoot,
            unet=tempUNet,
            feature_extractor=None,
            safety_checker=None,
            torch_dtype=self.defaultDType,
            **pipeArgDict
        )

        if 'scheduler' in self.modelCfgDict.keys():
            scheduer = getattr(diffusers,self.modelCfgDict['scheduler'])
            self.pipe.scheduler = scheduer.from_config(self.pipe.scheduler.config)
        
        if self.useXformers:
            self.pipe.enable_xformers_memory_efficient_attention()

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

    def blendModel(self, blendParamDictList, blendMode='weightMix'):
        firstModelParamDict = blendParamDictList[0]
        firstModelName = firstModelParamDict['name']
        firstModelFactor = firstModelParamDict['factor']
        firstModelWeightKeys = self.modelUNetList[firstModelName].state_dict(
        ).keys()

        tempStateDict = {}
        blendMetaInfoDict = {
            'mode': blendMode,
            'param': None
        }

        if blendMode == 'randLayer':
            tempStateChoosenDict = {}
            for weightKey in firstModelWeightKeys:
                randomIndex = random.randint(0, len(blendParamDictList)-1)
                choosenModelParamDict = blendParamDictList[randomIndex]
                choosenModelName = choosenModelParamDict['name']
                tempStateDict[weightKey] = self.modelUNetList[choosenModelName].state_dict()[
                    weightKey]
                tempStateChoosenDict[weightKey] = randomIndex
            blendMetaInfoDict['param'] = tempStateChoosenDict
        elif blendMode == 'weightMix':
            modelIdx = 0
            factorDict = {modelIdx: firstModelFactor}
            for weightKey, weightTensor in self.modelUNetList[firstModelName].state_dict().items():
                tempStateDict[weightKey] = weightTensor*firstModelFactor

            for modelParamDict in blendParamDictList[1:]:
                modelName = modelParamDict['name']
                modelFactor = modelParamDict['factor']
                modelIdx += 1
                factorDict[modelIdx] = modelFactor
                for weightKey, weightTensor in tempStateDict.items():
                    tempStateDict[weightKey] += self.modelUNetList[modelName].state_dict()[
                        weightKey]*modelFactor

            blendMetaInfoDict['param'] = factorDict

        self.blendMetaInfoDict = blendMetaInfoDict

        return tempStateDict

    def blendInRuntime(self, blendParamDictList, blendMode='weightMix'):
        self.pipe.unet.load_state_dict(
            self.blendModel(blendParamDictList, blendMode))

    def getExif(self, jsonDict):
        # https://stackoverflow.com/questions/52729428/how-to-write-custom-metadata-into-jpeg-with-python/63400376#63400376
        data = json.dumps(jsonDict).encode(encoding='utf8')
        exif_ifd = {piexif.ExifIFD.MakerNote: data}

        exif_dict = {"0th": {}, "Exif": exif_ifd, "1st": {},
                     "thumbnail": None, "GPS": {}}
        exif_dat = piexif.dump(exif_dict)
        return exif_dat

    def generate(self, prompt,
                 outputDir='./imgs',
                 seed=None, usePromptAsSubDir=False,
                 returnPILImage=False,
                 extraArgDict={}):
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
            'model': self.modelCfgDict,
            'blendMetaInfo': self.blendMetaInfoDict
        }

        argDict = {
            'prompt': prompt,
            'height': 512,
            'width': 512,
            'num_inference_steps': 50,
            'guidance_scale': 7.5,
            'max_embeddings_multiples': 3,
        }

        argDict.update(extraArgDict)
        genMetaInfoDict.update(argDict)

        if not returnPILImage:
            if usePromptAsSubDir:
                if 'originalPrompt' in argDict.keys():
                    prompt = argDict['originalPrompt']
                else:
                    prompt = argDict['prompt']
                outputDir = os.path.join(
                    outputDir, os.path.normpath(prompt))

            if not os.path.exists(outputDir):
                os.makedirs(outputDir)

        if 'originalPrompt' in argDict.keys():
            del argDict['originalPrompt']

        image = self.pipe(generator=self.randGenerator,
                          **argDict
                          ).images[0]

        exifGenMetaInfoDict = genMetaInfoDict
        if 'image' in exifGenMetaInfoDict.keys():
            del exifGenMetaInfoDict['image']

        if returnPILImage:
            return image
        else:
            exif_dat = self.getExif(exifGenMetaInfoDict)
            image.save(os.path.join(outputDir, '%d.jpg' %
                    seed), quality=90, exif=exif_dat)

    def to(self, device):
        self.pipe.to(device)
        self.randGenerator = torch.Generator(device=device)


# example
# from Creator import DiffusionCreator,RandomArtistGenerator,RandomImageSizeGenerator,PromptGenerator
# import torch

# if __name__ == "__main__":
#     modelCfgDict = ({'name': 'runwayml/stable-diffusion-v1-5', 'factor': 0.4},
#                           {'name': 'Linaqruf/anything-v3.0', 'factor': 0.6})
#     creator = DiffusionCreator(modelWeightRoot=r'../StableDiffusionWeight',
#                                modelCfgDict=modelCfgDict,  # To use local weight you should start with "."
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
