import csv
import random
import torch
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler, AutoencoderKL
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
    def __init__(self, sizeList=[(1088, 832), (1088, 768), (1088, 640), (896, 896), (1088, 512)]) -> None:
        self.sizeList = sizeList

    def getSize(self):
        size = random.choice(self.sizeList)
        if random.random() > 0.5:
            size = (size[1], size[0])
        return size


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
        self.pipe = StableDiffusionPipeline.from_pretrained(
            modelName, cache_dir=self.modelWeightRoot,
            requires_safety_checker=False,
            feature_extractor=None,
            safety_checker=None,
            torch_dtype=self.defaultDType,
        )
        if self.useXformers:
            self.pipe.enable_xformers_memory_efficient_attention()

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

        genMetaInfoDict = {
            'seed': seed,
            'prompt': prompt,
            'model': self.defaultModel
        }

        argDict = {
            'height': 512,
            'width': 512,
            'num_inference_steps': 50,
            'guidance_scale': 7.5
        }

        argDict.update(extraArgDict)
        genMetaInfoDict.update(argDict)

        image = self.pipe(prompt,
                          generator=self.randGenerator,
                          **argDict
                          ).images[0]

        exif_dat = self.getExif(genMetaInfoDict)
        image.save(os.path.join(outputDir, '%d.jpg' %
                   seed), quality=90, exif=exif_dat)

    def to(self, device):
        self.pipe.to(device)
        self.randGenerator = torch.Generator(device=device)


if __name__ == "__main__":
    creator = DiffusionCreator(modelWeightRoot=r'F:\StableDiffusionWeight',
                               defaultModel='.Anything',  # To use local weight you should start with "."
                               defaultDType=torch.bfloat16,
                               useXformers=True)
    creator.to('cuda')
    artistGen = RandomArtistGenerator()
    sizeGen = RandomImageSizeGenerator()

    while True:
        randomSize = sizeGen.getSize()
        randomArtist = artistGen.getArtist()

        genArgDict = {
            'negative_prompt':
            'closed eyes,ugly,Polydactyly,extra fingers,fused fingers,extra legs,draft,text,watermark,signature,two heads,mutated hands,mutation,deformed, bad anatomy, bad proportions,too many fingers,morbid, mutilated, extra limbs,disfigured,missing arms,missing legs,extra arms,malformed limbs',
            'height': randomSize[0],
            'width': randomSize[1],
            'num_inference_steps': 60
        }
        creator.generate(
            'A beautiful girl in huge Dahlia garden,by artist '+randomArtist, extraArgDict=genArgDict)
