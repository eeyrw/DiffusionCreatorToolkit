from diffusers import UniPCMultistepScheduler, DDIMScheduler,StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline
import diffusers
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
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
            self.loadModel(blendInfoFile, loadMode)

    def parseModelPath(self, modelPath, modelWeightRoot):
        if modelPath[0] == '.':
            modelPath = os.path.join(
                modelWeightRoot, modelPath[1:])
        return modelPath

    def loadModel(self, blendInfoFile, blendMode):

        baseModelName = self.modelCfgDict['models'][0]['name']
        baseModelName = self.parseModelPath(
            baseModelName, self.modelWeightRoot)

        pipeArgDict = {}

        from lpw_stable_diffusion_xl import SDXLLongPromptWeightingPipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            baseModelName, cache_dir=self.modelWeightRoot,
            variant='fp16',
            use_safetensors=True,
            torch_dtype=self.defaultDType,
            local_files_only=True,
            add_watermarker=False,
            **pipeArgDict
        )

        refinerModelName = self.modelCfgDict['models'][0]['refiner']
        refinerModelName = self.parseModelPath(
            refinerModelName, self.modelWeightRoot)


        self.pipeRefiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            refinerModelName,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            add_watermarker=False,
             cache_dir=self.modelWeightRoot,local_files_only=True,**pipeArgDict)

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
                self.pipeRefiner.scheduler = scheduer.from_config(
                    self.pipe.scheduler.config)
        if self.useXformers:
            self.pipe.enable_xformers_memory_efficient_attention()
            self.pipeRefiner.enable_xformers_memory_efficient_attention()

    def recordBlendMetaInfo(self, outputPath):
        with open(outputPath, 'w') as f:
            json.dump({'blendInfo': self.blendMetaInfoDict,
                      'models': self.modelCfgDict}, f)

    def outputUNetWeightInfo(self, outputPath):
        with open(outputPath, 'w') as f:
            for weightKey in self.pipe.unet.state_dict():
                f.write(weightKey+'\n')

    def exportSDModel(self, outputPath):
        exportToSD(self.pipe.text_encoder.state_dict(),
                   self.pipe.unet.state_dict(),
                   self.pipe.vae.state_dict(),
                   outputPath)

    def getExif(self, jsonDict):
        # https://stackoverflow.com/questions/52729428/how-to-write-custom-metadata-into-jpeg-with-python/63400376#63400376
        data = json.dumps(jsonDict).encode(encoding='utf8')
        exif_ifd = {piexif.ExifIFD.MakerNote: data}

        exif_dict = {"0th": {}, "Exif": exif_ifd, "1st": {},
                     "thumbnail": None, "GPS": {}}
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
        self.generate(exifGenMetaInfoDict['prompt'], outputDir='reproduce_imgs',
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
                outputDir = os.path.join(
                    outputDir, os.path.normpath(prompt))

            if not os.path.exists(outputDir):
                os.makedirs(outputDir)

        if 'originalPrompt' in argDict.keys():
            del argDict['originalPrompt']

        # image = self.pipe(generator=self.randGenerator,
        #                   **argDict
        #                   ).images[0]

        high_noise_frac = 0.8

        image = self.pipe(
            generator=self.randGenerator,
            denoising_end=high_noise_frac,
            output_type="latent",
            **argDict
        ).images
        del argDict['height']
        del argDict['width']
        image = self.pipeRefiner(
            denoising_start=high_noise_frac,
            aesthetic_score = 4,
            negative_aesthetic_score = 1,
            image=image,
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
