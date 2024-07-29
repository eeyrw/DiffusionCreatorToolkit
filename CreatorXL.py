from diffusers import UniPCMultistepScheduler, DDIMScheduler,StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline
import diffusers
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from safetensors import safe_open
from transformers import CLIPTextModel, CLIPTokenizer
import random
import torch
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler, AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
from transformers import CLIPTextModel
import json
import piexif
import copy
from DF2SD import exportToSD
from PIL import Image
from diffusers.models.model_loading_utils import load_state_dict
torch.backends.cuda.matmul.allow_tf32 = True

# {
#     'vae': '.ARTMAN_HUGE_144000',
#     'textEncoder':'.ARTMAN_HUGE_144000',
#     'models':[{'name':'.ARTMAN_HUGE_144000','factor':0.5},]
#     }


class DiffusionCreator:

    def __init__(self,
                 modelWeightRoot='.',
                 modelCfgDict=None,
                 defaultDType=torch.float16,
                 useXformers=False,
                 loadMode='weightMix',
                 blendInfoFile=None) -> None:
        self.schedulerMapping = {
            'DPM++ 2M': ('DPMSolverMultistepScheduler', {}),
            'DPM++ 2M Karras': ('DPMSolverMultistepScheduler', {
                'use_karras_sigmas': True
            }),
            'DPM++ 2M SDE': ('DPMSolverMultistepScheduler', {
                'algorithm_type': 'sde-dpmsolver++'
            }),
            'DPM++ 2M SDE Karras': ('DPMSolverMultistepScheduler', {
                'algorithm_type': 'sde-dpmsolver++',
                'use_karras_sigmas': True
            }),
            'DPM++ SDE': ('DPMSolverSinglestepScheduler', {}),
            'DPM++ SDE Karras': ('DPMSolverSinglestepScheduler', {
                'use_karras_sigmas': True
            }),
            'DPM2': ('KDPM2DiscreteScheduler', {}),
            'DPM2 Karras': ('KDPM2DiscreteScheduler', {
                'use_karras_sigmas': True
            }),
            'DPM2 a': ('KDPM2AncestralDiscreteScheduler', {}),
            'DPM2 a Karras': ('KDPM2AncestralDiscreteScheduler', {
                'use_karras_sigmas': True
            }),
            'Euler': ('EulerDiscreteScheduler', {}),
            'Euler a': ('EulerAncestralDiscreteScheduler', {}),
            'Heun': ('HeunDiscreteScheduler', {}),
            'LMS': ('LMSDiscreteScheduler', {}),
            'LMS Karras': ('LMSDiscreteScheduler', {
                'use_karras_sigmas': True
            })
        }
        self.modelWeightRoot = modelWeightRoot
        self.modelCfgDict = modelCfgDict
        self.defaultDType = defaultDType
        self.useXformers = useXformers
        self.randGenerator = torch.Generator()
        self.blendMetaInfoDict = {}
        self.specifiedBlendInfo = None
        self.useRefiner = False
        if self.modelCfgDict is not None:
            self.loadModel(blendInfoFile, loadMode)

    def parseModelPath(self, modelPath, modelWeightRoot):
        if modelPath[0] == '.':
            modelPath = os.path.join(modelWeightRoot, modelPath[1:])
        return modelPath

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
        blendMetaInfoDict = {'mode': blendMode, 'param': None}

        if blendMode == 'randLayer':
            tempStateChoosenDict = {}
            for weightKey in firstModelWeightKeys[0]:
                if self.specifiedBlendInfo is not None:
                    randomIndex = self.specifiedBlendInfo['blendInfo'][
                        'param'][weightKey]
                else:
                    randomIndex = random.randint(0,
                                                 len(blendParamDictList) - 1)
                choosenModelParamDict = blendParamDictList[randomIndex]
                choosenModelName = choosenModelParamDict['name']
                tempStateDict[weightKey] = self.loraList[choosenModelName][0][
                    weightKey]
                tempStateChoosenDict[weightKey] = randomIndex
            blendMetaInfoDict['param'] = tempStateChoosenDict
        elif blendMode == 'weightMix':
            modelIdx = 0
            factorDict = {modelIdx: firstModelFactor}
            for weightKey, weightTensor in self.loraList[firstModelName][
                    0].items():
                tempStateDict[weightKey] = weightTensor * firstModelFactor

            for modelParamDict in blendParamDictList[1:]:
                modelName = modelParamDict['name']
                modelFactor = modelParamDict['factor']
                modelIdx += 1
                factorDict[modelIdx] = modelFactor
                for weightKey, weightTensor in tempStateDict.items():
                    tempStateDict[weightKey] += self.loraList[modelName][0][
                        weightKey] * modelFactor

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
        elif blendMode == 'EMA':
            modelIdx = 0
            modelNum = len(blendParamDictList)
            beta = 0.9
            print(list(range(modelNum-1, -1, -1)))
            EMAFactorList = [
                (1-beta)*pow(beta, t)/(1-pow(beta, modelNum)) for t in range(modelNum-1, -1, -1)
            ]
            print(EMAFactorList)
            print(sum(EMAFactorList))

            for weightKey, weightTensor in self.modelUNetList[firstModelName].state_dict().items():
                tempStateDict[weightKey] = weightTensor*EMAFactorList[0]

            for modelParamDict, factor in zip(blendParamDictList[1:], EMAFactorList[1:]):
                modelName = modelParamDict['name']
                for weightKey, weightTensor in tempStateDict.items():
                    tempStateDict[weightKey] += self.modelUNetList[modelName].state_dict()[
                        weightKey]*factor

        elif blendMode == 'bavMix':
            factorDict = {0: firstModelFactor}
            for key in self.modelUNetList[firstModelName].state_dict().keys():
                values = [model.state_dict()[key]
                          for model in self.modelUNetList.values()]
                mean_value = sum(values) / len(values)

                alpha = 0.4
                beta = 1.6
                tempStateDict[key] = mean_value * (abs(self.modelUNetList[firstModelName].state_dict()[key] - alpha*mean_value) > abs(self.modelUNetList[firstModelName].state_dict()[key] - beta*values[0])) \
                    + values[0] * (abs(self.modelUNetList[firstModelName].state_dict()[key] - alpha*mean_value)
                                   <= abs(self.modelUNetList[firstModelName].state_dict()[key] - beta*values[0]))

            blendMetaInfoDict['param'] = factorDict

        self.blendMetaInfoDict = blendMetaInfoDict

        return tempStateDict
    
    def loadMultiModel(self, blendParamDictList):
        self.modelUNetList = {}
        for blendParamDict in blendParamDictList:
            modelNameRaw = blendParamDict['name']
            if 'variant' in blendParamDict.keys():
                variant = blendParamDict['variant']
            else:
                variant = None
            if modelNameRaw[0] == '.':
                modelName = os.path.join(
                    self.modelWeightRoot, modelNameRaw[1:], 'unet')
            else:
                modelName = modelNameRaw

            unet = UNet2DConditionModel.from_pretrained(
                modelName,
                subfolder='unet',
                cache_dir=self.modelWeightRoot, local_files_only=True,
                variant=variant,
                torch_dtype=self.defaultDType)

            self.modelUNetList[modelNameRaw] = unet

    def loadModel(self, blendInfoFile, blendMode):

        if blendInfoFile is not None:
            with open(blendInfoFile, 'r') as f:
                self.specifiedBlendInfo = json.load(f)
                self.modelCfgDict['models'] = self.specifiedBlendInfo['models']['models']
                blendMode = self.specifiedBlendInfo['blendInfo']['mode']

        self.loadMultiModel(self.modelCfgDict['models'])
        baseModelName = self.modelCfgDict['models'][0]['name']

        if 'variant' in self.modelCfgDict['models'][0].keys():
            variant = self.modelCfgDict['models'][0]['variant']
        else:
            variant = None

        tempUNet = copy.deepcopy(self.modelUNetList[baseModelName])
        if len(self.modelCfgDict['models']) > 1:
            unetWeight = self.blendModel(
                self.modelCfgDict['models'], blendMode=blendMode)
            tempUNet.load_state_dict(unetWeight)

        baseModelName = self.parseModelPath(
            baseModelName, self.modelWeightRoot)

        pipeArgDict = {}

        if 'vae' in self.modelCfgDict.keys():
            vaePath = self.parseModelPath(self.modelCfgDict['vae'],
                                          self.modelWeightRoot)
            if os.path.isdir(os.path.join(vaePath, 'vae')):
                pipeArgDict['vae'] = AutoencoderKL.from_pretrained(
                    vaePath,
                    subfolder='vae',
                    cache_dir=self.modelWeightRoot,
                    local_files_only=True,
                    torch_dtype=self.defaultDType)
            elif os.path.isfile(vaePath):
                with open('assets/sdxl_vae_default_cfg.json') as f:
                    vae_cfg = json.load(f)
                pipeArgDict['vae'] = AutoencoderKL(**vae_cfg)
                vaeStateDict = load_state_dict(vaePath)
                pipeArgDict['vae'].load_state_dict(vaeStateDict)
                pipeArgDict['vae'].to(self.defaultDType)
                pipeArgDict['vae'].eval()
            else:
                pipeArgDict['vae'] = AutoencoderKL.from_pretrained(
                    vaePath,
                    cache_dir=self.modelWeightRoot,
                    local_files_only=True,
                    torch_dtype=self.defaultDType)
                

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            baseModelName,
            unet=tempUNet,
            cache_dir=self.modelWeightRoot,
            variant=variant,
            use_safetensors=True,
            torch_dtype=self.defaultDType,
            local_files_only=True,
            add_watermarker=False,
            **pipeArgDict)

        if 'refiner' in self.modelCfgDict['models'][0].keys():
            self.useRefiner = True
            refinerModelName = self.modelCfgDict['models'][0]['refiner']
            refinerModelName = self.parseModelPath(refinerModelName,
                                                   self.modelWeightRoot)

            self.pipeRefiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                refinerModelName,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                add_watermarker=False,
                cache_dir=self.modelWeightRoot,
                local_files_only=True,
                **pipeArgDict)

        if 'scheduler' in self.modelCfgDict.keys():
            prediction_type = self.pipe.scheduler.config.prediction_type
            if self.modelCfgDict['scheduler'] in self.schedulerMapping.keys():
                schedulerName, schedulerExtraConfig = self.schedulerMapping[
                    self.modelCfgDict['scheduler']]
                scheduer = getattr(diffusers, schedulerName)
                self.pipe.scheduler = scheduer.from_config(
                    # predictor_order=3, corrector_order=4
                    self.pipe.scheduler.config, **schedulerExtraConfig)
            else:
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
                        self.pipe.scheduler.config, prediction_type=prediction_type)
        if 'loras' in self.modelCfgDict.keys():
            if len(self.modelCfgDict['loras']) > 1:
                self.loadMultiLora(self.modelCfgDict['loras'])
                lora = self.blendLora(self.modelCfgDict['loras'],
                                      blendMode=blendMode)
            else:
                lora, _ = self.pipe.lora_state_dict(
                    self.modelCfgDict['loras'][0]['name'])
            # loraDict = {}
            # for k in lora.keys():
            #     nk = k.replace('module.', '')
            #     loraDict[nk]=lora[k]
            self.pipe.load_lora_weights(lora)
            self.applyLora = True
        else:
            self.applyLora = False
        if self.useXformers:
            self.pipe.enable_xformers_memory_efficient_attention()
            if self.useRefiner:
                self.pipeRefiner.enable_xformers_memory_efficient_attention()

    def recordBlendMetaInfo(self, outputPath):
        with open(outputPath, 'w') as f:
            json.dump(
                {
                    'blendInfo': self.blendMetaInfoDict,
                    'models': self.modelCfgDict
                }, f)

    def outputUNetWeightInfo(self, outputPath):
        with open(outputPath, 'w') as f:
            for weightKey in self.pipe.unet.state_dict():
                f.write(weightKey + '\n')

    def exportSDModel(self, outputPath):
        exportToSD(self.pipe.text_encoder.state_dict(),
                   self.pipe.unet.state_dict(), self.pipe.vae.state_dict(),
                   outputPath)

    def getExif(self, jsonDict):
        # https://stackoverflow.com/questions/52729428/how-to-write-custom-metadata-into-jpeg-with-python/63400376#63400376
        data = json.dumps(jsonDict).encode(encoding='utf8')
        exif_ifd = {piexif.ExifIFD.MakerNote: data}

        exif_dict = {
            "0th": {},
            "Exif": exif_ifd,
            "1st": {},
            "thumbnail": None,
            "GPS": {}
        }
        exif_dat = piexif.dump(exif_dict)
        return exif_dat

    def reproduce(self, refImagePath, device='cuda'):
        img = Image.open(refImagePath)
        exif_dict = piexif.load(img.info['exif'])
        exifGenMetaInfoDict = json.loads(
            exif_dict['Exif'][piexif.ExifIFD.MakerNote])
        self.modelCfgDict = exifGenMetaInfoDict.pop('model')
        self.loadModel(None, 'weightMix')
        self.to(device)
        exifGenMetaInfoDict.pop('blendMetaInfo')
        self.generate(exifGenMetaInfoDict['prompt'],
                      outputDir='reproduce_imgs',
                      seed=exifGenMetaInfoDict.pop('seed'),
                      extraArgDict=exifGenMetaInfoDict)

    def genImageJiasaw(self, imageList, width, height, col, row, outPath):
        to_image = Image.new('RGB', (col * width, row * height))  # 创建一个新图
        # 循环遍历，把每张图片按顺序粘贴到对应位置上
        for y in range(row):
            for x in range(col):
                from_image = imageList[y * col + x]
                to_image.paste(from_image, (x * width, y * height))
        return to_image.save(outPath, quality=90)  # 保存新图

    def generate(self,
                 prompt,
                 outputDir='./imgs',
                 seed=None,
                 usePromptAsSubDir=False,
                 returnPILImage=False,
                 extraArgDict={}):
        if seed is None:
            seed = self.randGenerator.seed()
            self.randGenerator.manual_seed(seed)
        else:
            self.randGenerator.manual_seed(seed)

        if (not self.applyLora
            ) and 'cross_attention_kwargs' in extraArgDict.keys():
            del extraArgDict['cross_attention_kwargs']

        if 'prompt' in extraArgDict.keys():
            prompt = extraArgDict['prompt']

        genMetaInfoDict = {
            'seed': seed,
            'prompt': prompt,
            'model': self.modelCfgDict,
            'blendMetaInfo': self.blendMetaInfoDict
        }
        if 'prompt_2' in extraArgDict.keys():
            genMetaInfoDict['prompt_2'] = extraArgDict['prompt_2']
        argDict = {
            'prompt': prompt,
            'height': 512,
            'width': 512,
            'num_inference_steps': 50,
            'guidance_scale': 7.5,
        }

        argDict.update(extraArgDict)
        genMetaInfoDict.update(argDict)

        if not returnPILImage:
            if usePromptAsSubDir:
                if 'originalPrompt' in argDict.keys():
                    prompt = argDict['originalPrompt']
                else:
                    prompt = argDict['prompt']
                if len(prompt) > 133:
                    prompt = prompt[0:133]
                outputDir = os.path.join(outputDir, os.path.normpath(prompt))

            if not os.path.exists(outputDir):
                os.makedirs(outputDir)

        if 'originalPrompt' in argDict.keys():
            del argDict['originalPrompt']
        if 'negativePrompt' in argDict.keys(
        ) and argDict['negativePrompt'] == '':
            argDict['negativePrompt'] = None
        # image = self.pipe(generator=self.randGenerator,
        #                   **argDict
        #                   ).images[0]

        high_noise_frac = 0.8

        if self.useRefiner:
            image = self.pipe(generator=self.randGenerator,
                              denoising_end=high_noise_frac,
                              output_type="latent",
                              **argDict).images

            del argDict['height']
            del argDict['width']
            image = self.pipeRefiner(denoising_start=high_noise_frac,
                                     aesthetic_score=4,
                                     negative_aesthetic_score=1,
                                     image=image,
                                     **argDict).images[0]

        else:
            image = self.pipe(generator=self.randGenerator,
                              **argDict).images[0]

        exifGenMetaInfoDict = genMetaInfoDict
        if 'image' in exifGenMetaInfoDict.keys():
            del exifGenMetaInfoDict['image']

        if returnPILImage:
            return image
        else:
            exif_dat = self.getExif(exifGenMetaInfoDict)
            image.save(os.path.join(outputDir, '%d.jpg' % seed),
                       quality=90,
                       exif=exif_dat)

    def to(self, device):
        self.pipe.to(device)
        if self.useRefiner:
            self.pipeRefiner.to('cuda:1')
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
