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
            'huge': [(1024, 1024), (1280, 768), (1280, 640), (1280, 832)],
            'big': [(1088, 832), (1088, 768), (1088, 640), (896, 896), (1088, 512)],
            'small': [(768, 768), (512, 512), (768, 640), (768, 512)],
            'square': [(768, 768), (832, 832), (1024, 1024)],
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
        if self.artistGenerator:
            prompt = prompt+',by artist '+self.artistGenerator.getArtist()
        returnDict.update({
            'originalPrompt': rawPrompt,
            'prompt': prompt,
            'negative_prompt': additionalNegPrompt + self.negativePrompt
        })
        return returnDict