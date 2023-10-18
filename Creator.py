from diffusers import UniPCMultistepScheduler, DDIMScheduler
import diffusers
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer
import random
import torch
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler, AutoencoderKL, UNet2DConditionModel,UniPCMultistepScheduler
from transformers import CLIPTextModel
import json
import piexif
import copy
from DF2SD import exportToSD
from PIL import Image
torch.backends.cuda.matmul.allow_tf32 = True

# {
#     'vae': '.ARTMAN_HUGE_144000',
#     'textEncoder':'.ARTMAN_HUGE_144000',
#     'models':[{'name':'.ARTMAN_HUGE_144000','factor':0.5},]
#     }


class DiffusionCreator:
    def __init__(self, modelWeightRoot='.',
                 modelCfgDict=None,
                 defaultDType=torch.float16,
                 useXformers=False,
                 loadMode='weightMix',
                 blendInfoFile=None) -> None:
        self.modelWeightRoot = modelWeightRoot
        self.modelCfgDict = modelCfgDict
        self.defaultDType = defaultDType
        self.useXformers = useXformers
        self.randGenerator = torch.Generator()
        self.blendMetaInfoDict = {}
        self.specifiedBlendInfo = None
        if self.modelCfgDict is not None:
            self.loadModel(blendInfoFile,loadMode)

    def parseModelPath(self, modelPath, modelWeightRoot):
        if modelPath[0] == '.':
            modelPath = os.path.join(
                modelWeightRoot, modelPath[1:])
        return modelPath

    def loadModel(self,blendInfoFile,blendMode):
        if blendInfoFile is not None:
            with open(blendInfoFile,'r') as f:
                self.specifiedBlendInfo = json.load(f)
                self.modelCfgDict['models'] = self.specifiedBlendInfo['models']['models']
                blendMode = self.specifiedBlendInfo['blendInfo']['mode']

        self.loadMultiModel(self.modelCfgDict['models'])
        baseModelName = self.modelCfgDict['models'][0]['name']
        tempUNet = copy.deepcopy(self.modelUNetList[baseModelName])
        if len(self.modelCfgDict['models']) > 1:
            unetWeight = self.blendModel(self.modelCfgDict['models'],blendMode=blendMode)
            tempUNet.load_state_dict(unetWeight)

        baseModelName = self.parseModelPath(
            baseModelName, self.modelWeightRoot)

        pipeArgDict = {}

        from lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
        from lpw_stable_diffusion_multi import StableDiffusionLongPromptWeightingMultiUNetPipeline

        if 'vae' in self.modelCfgDict.keys():
            vaePath = self.parseModelPath(
                self.modelCfgDict['vae'], self.modelWeightRoot)
            pipeArgDict['vae'] = AutoencoderKL.from_pretrained(
                vaePath, subfolder='vae', cache_dir=self.modelWeightRoot,local_files_only=True, torch_dtype=self.defaultDType)

        if 'textEncoder' in self.modelCfgDict.keys():
            textEncoderPath = self.parseModelPath(
                self.modelCfgDict['textEncoder'], self.modelWeightRoot)
            pipeArgDict['text_encoder'] = CLIPTextModel.from_pretrained(
                textEncoderPath, subfolder='text_encoder', cache_dir=self.modelWeightRoot,local_files_only=True, torch_dtype=self.defaultDType)
            pipeArgDict['tokenizer'] = CLIPTokenizer.from_pretrained(
                textEncoderPath, subfolder='tokenizer', cache_dir=self.modelWeightRoot)

        if 'multiUNet' in self.modelCfgDict.keys():
            self.pipe = StableDiffusionLongPromptWeightingMultiUNetPipeline.from_pretrained(
                baseModelName, cache_dir=self.modelWeightRoot,local_files_only=True,
                unet=self.modelUNetList[self.modelCfgDict['models'][0]['name']],
                feature_extractor=None,
                safety_checker=None,
                torch_dtype=self.defaultDType,
                **pipeArgDict
            )
            print('%s -> %s'%(self.modelCfgDict['models'][0]['name'],
                              self.modelUNetList[self.modelCfgDict['models'][0]['name']].device))
            modelCnt=1
            for model in self.modelCfgDict['models'][1:]:
                self.modelUNetList[model['name']].enable_xformers_memory_efficient_attention()
                self.modelUNetList[model['name']].to('cuda:%d'%(modelCnt//2))
                print('%s -> %s'%(model['name'],self.modelUNetList[model['name']].device))                
                modelCnt = modelCnt+1
                self.pipe.appendExtraUNet(self.modelUNetList[model['name']])
        else:
            self.pipe = StableDiffusionLongPromptWeightingPipeline.from_pretrained(
                baseModelName, cache_dir=self.modelWeightRoot,local_files_only=True,
                unet=tempUNet,
                feature_extractor=None,
                safety_checker=None,
                torch_dtype=self.defaultDType,
                **pipeArgDict
            )

        if 'scheduler' in self.modelCfgDict.keys():
            if self.modelCfgDict['scheduler'] == 'DDIMScheduler':
                self.pipe.scheduler = DDIMScheduler(
                    **{
                        "beta_end": 0.012,
                        "beta_schedule": "scaled_linear",
                        "beta_start": 0.00085,
                        "clip_sample": False,
                        "num_train_timesteps": 1000,
                        "prediction_type": "epsilon",
                        "set_alpha_to_one": False,
                        "steps_offset": 1,
                    }
                )
            else:
                scheduer = getattr(diffusers, self.modelCfgDict['scheduler'])
                self.pipe.scheduler = scheduer.from_config(
                    self.pipe.scheduler.config)
        if 'loras' in self.modelCfgDict.keys():  
            if len(self.modelCfgDict['loras']) > 1:
                self.loadMultiLora(self.modelCfgDict['loras'])
                lora = self.blendLora(self.modelCfgDict['loras'],blendMode=blendMode)
            else:
                lora = self.pipe.lora_state_dict(self.modelCfgDict['loras'][0]['name'])
            self.pipe.load_lora_weights(lora)           
            self.applyLora = True
        else:
            self.applyLora = False
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
                cache_dir=self.modelWeightRoot,local_files_only=True,
                torch_dtype=self.defaultDType)

            self.modelUNetList[modelNameRaw] = unet

    def loadMultiLora(self, blendParamDictList):
        self.loraList = {}
        for blendParamDict in blendParamDictList:
            modelNameRaw = blendParamDict['name']
            modelName = modelNameRaw

            lora = self.pipe.lora_state_dict(modelName)

            self.loraList[modelNameRaw] = lora

    def blendLora(self, blendParamDictList, blendMode='weightMix'):
        firstModelParamDict = blendParamDictList[0]
        firstModelName = firstModelParamDict['name']
        firstModelFactor = firstModelParamDict['factor']
        firstModelWeightKeys = self.loraList[firstModelName][0].keys()

        tempStateDict = {}
        blendMetaInfoDict = {
            'mode': blendMode,
            'param': None
        }

        if blendMode == 'randLayer':
            tempStateChoosenDict = {}
            for weightKey in firstModelWeightKeys[0]:
                if self.specifiedBlendInfo is not None:
                    randomIndex = self.specifiedBlendInfo['blendInfo']['param'][weightKey]
                else:          
                    randomIndex = random.randint(0, len(blendParamDictList)-1)
                choosenModelParamDict = blendParamDictList[randomIndex]
                choosenModelName = choosenModelParamDict['name']
                tempStateDict[weightKey] = self.loraList[choosenModelName][0][
                    weightKey]
                tempStateChoosenDict[weightKey] = randomIndex
            blendMetaInfoDict['param'] = tempStateChoosenDict
        elif blendMode == 'weightMix':
            modelIdx = 0
            factorDict = {modelIdx: firstModelFactor}
            for weightKey, weightTensor in self.loraList[firstModelName][0].items():
                tempStateDict[weightKey] = weightTensor*firstModelFactor

            for modelParamDict in blendParamDictList[1:]:
                modelName = modelParamDict['name']
                modelFactor = modelParamDict['factor']
                modelIdx += 1
                factorDict[modelIdx] = modelFactor
                for weightKey, weightTensor in tempStateDict.items():
                    tempStateDict[weightKey] += self.loraList[modelName][0][
                        weightKey]*modelFactor

        self.blendMetaInfoDict = blendMetaInfoDict

        return tempStateDict

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
                if self.specifiedBlendInfo is not None:
                    randomIndex = self.specifiedBlendInfo['blendInfo']['param'][weightKey]
                else:          
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
        elif blendMode == 'bavMix':
            factorDict = {0: firstModelFactor}
            for key in self.modelUNetList[firstModelName].state_dict().keys():
                values = [model.state_dict()[key] for model in self.modelUNetList.values()]
                mean_value = sum(values) / len(values)

                alpha = 0.4
                beta = 1.6
                tempStateDict[key] = mean_value * (abs(self.modelUNetList[firstModelName].state_dict()[key] - alpha*mean_value) > abs(self.modelUNetList[firstModelName].state_dict()[key] - beta*values[0])) \
                    + values[0] * (abs(self.modelUNetList[firstModelName].state_dict()[key] - alpha*mean_value) <= abs(self.modelUNetList[firstModelName].state_dict()[key] - beta*values[0]))


            blendMetaInfoDict['param'] = factorDict

        self.blendMetaInfoDict = blendMetaInfoDict

        return tempStateDict
    
    def recordBlendMetaInfo(self,outputPath):
        with open(outputPath,'w') as f:
            json.dump({'blendInfo':self.blendMetaInfoDict,'models':self.modelCfgDict},f)

    def outputUNetWeightInfo(self,outputPath):
        with open(outputPath,'w') as f:
            for weightKey in self.pipe.unet.state_dict():
                f.write(weightKey+'\n')

    def exportSDModel(self,outputPath):
        exportToSD(self.pipe.text_encoder.state_dict(),
                   self.pipe.unet.state_dict(),
                   self.pipe.vae.state_dict(),
                   outputPath)

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
    
    def reproduce(self,refImagePath,device='cuda'):
        img = Image.open(refImagePath)
        exif_dict = piexif.load(img.info['exif'])
        exifGenMetaInfoDict = json.loads(exif_dict['Exif'][piexif.ExifIFD.MakerNote])
        self.modelCfgDict = exifGenMetaInfoDict.pop('model')
        self.loadModel(None,'weightMix')
        self.to(device)
        exifGenMetaInfoDict.pop('blendMetaInfo')
        self.generate(exifGenMetaInfoDict['prompt'],outputDir='reproduce_imgs',
                      seed=exifGenMetaInfoDict.pop('seed'), extraArgDict=exifGenMetaInfoDict)


        

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


        if (not self.applyLora) and 'cross_attention_kwargs' in extraArgDict.keys():
            del extraArgDict['cross_attention_kwargs']

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
