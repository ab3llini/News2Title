from bs4 import BeautifulSoup
import requests
from enum import Enum
import re
from tqdm import tqdm
import pickle
import pymongo

def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', str(data))


WIRED = {
        'title':
            {
                'object': 'span',
                'attributes': {
                    'aria-label': 'Article Title'
                }
            },
        'date' : {
            'object' : 'time',
            'attributes' : {
                'class' : 'date-mdy'
            }
        },
        'news':
            {
                'object': 'article',
                'attributes': {}
            }
    }


VERGE = {
    'title':
        {
            'object':'title',
            'attributes':{}
        },
    'news':
        {
            'object':'div',
            'attributes':{
                "class": "c-entry-content"
            }
        }
}

GIZMODO = {
    'title':
        {
            'object': 'title',
            'attributes': {}
        },
    'news':
        {
            'object': 'div',
            'attributes':{
                'class':'post-content entry-content js_entry-content '
            }
        }
}

class Parser:

    def __init__(self, format):
        self.format = format

    def parse(self, href):

        req = requests.get(href)
        html = req.text

        soup = BeautifulSoup(html)

        out = {}

        lookfor = self.format.keys()

        for comp in lookfor:

            parsed = soup.find(self.format[comp]['object'], self.format[comp]['attributes'])

            if parsed is None:
                raise Exception("Can't find specified component '%s' in link %s" % (comp, href))

            out[comp] = striphtml(parsed)

        return out

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
db = myclient["News2Titledb"]

p = Parser(format=WIRED)
with open("wired.pkl","rb") as f:
    links = pickle.load(f)

for link in tqdm(links):
    print(link)
    result = p.parse(link)
    #print(result)
    #db.news.insert_one(result)
