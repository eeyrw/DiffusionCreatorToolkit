import csv
import random
from PIL import Image

class RandomArtistGenerator:
    def __init__(self, artistDatabase='assets/artists.csv',
                 specifiedArtist=None,
                 specifiedArtistList=None,
                 specifiedStyle=None,
                 specifiedStyleList=None) -> None:
        useSpecifiedStyle = False
        if specifiedStyle is not None:
            specifiedStyleList = (specifiedStyle,)
            useSpecifiedStyle = True
        if specifiedStyleList is not None:
            useSpecifiedStyle = True

        with open(artistDatabase, newline='', encoding='utf8') as csvfile:
            reader = csv.DictReader(csvfile)
            artistDictList = list(reader)
            if useSpecifiedStyle:
                self.artistDictList = [artistDict for artistDict in artistDictList if float(
                    artistDict['score']) > 0.6 and artistDict['category'] in specifiedStyleList]
                print('Total Artist Num: %d' % len(self.artistDictList))
            else:
                self.artistDictList = [artistDict for artistDict in artistDictList if float(
                    artistDict['score']) > 0.6]

        self.specifiedArtist = specifiedArtist
        self.specifiedArtistList = specifiedArtistList

    def getArtist(self):
        if self.specifiedArtist:
            return self.specifiedArtist
        elif self.specifiedArtistList:
            return random.choice(self.specifiedArtistList)
        else:
            return random.choice(self.artistDictList)['artist']


class RandomImageSizeGenerator:
    def __init__(self, sizeSet='big', customSizeList=None) -> None:
        self.sizeListDict = {
            'giant': [(1280, 1280), (1536, 768), (1536, 832), (1536, 1024)],
            #'huge': [(1024, 1024), (1216, 768), (1152, 768),(1088, 704)],
            'huge': [(1088, 1088), (1152, 768), (1152, 832)],
            'big': [(1088, 832), (1088, 768), (1088, 640), (896, 896)],
            'littleMedium' : [(768, 768), (1024, 768), (1024, 640)],
            'medium' : [(1024, 1024),(768, 768), (1024, 768), (1024, 640)],
            'small': [(768, 768), (512, 512), (768, 640), (768, 512)],
            'square': [(768, 768), (832, 832), (1024, 1024)],
            'big_square': [(1024, 1024),(1152,1152)],
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
    def __init__(self, promptList, 
                 negativePrompt=None, 
                 artistGenerator=None,
                 postivePrompt=None) -> None:
        self.artistGenerator = artistGenerator
        self.promptList = promptList
        self.negativePrompt = negativePrompt
        self.postivePrompt = postivePrompt

    def getPrompt(self):
        returnDict = {}
        additionalNegPrompt = ''
        rawPrompt = random.choice(self.promptList)
        if isinstance(rawPrompt, (tuple, list)):
            prompt = rawPrompt[0]
            additionalNegPrompt = rawPrompt[1]+','
        elif isinstance(rawPrompt, str):
            prompt = rawPrompt
            additionalNegPrompt = ''
        elif isinstance(rawPrompt, dict):
            print(rawPrompt)
            prompt = rawPrompt['prompt']
            if 'additionalNegPrompt' in rawPrompt.keys():
                additionalNegPrompt = rawPrompt['additionalNegPrompt']
            if 'refImage' in rawPrompt.keys():
                rawPrompt['init_image'] = Image.open(rawPrompt['refImage'])
            returnDict.update(rawPrompt)
        else:
            raise RuntimeError('Wrong Prompt type')
        rawPrompt = prompt
        if self.postivePrompt:
            prompt = self.postivePrompt + ',' + prompt
        if self.artistGenerator:
            prompt = prompt+',by artist '+self.artistGenerator.getArtist()
        returnDict.update({
            'originalPrompt': rawPrompt,
            'prompt': prompt,
            'negative_prompt': additionalNegPrompt + self.negativePrompt
        })
        return returnDict