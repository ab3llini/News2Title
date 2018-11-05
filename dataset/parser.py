from bs4 import BeautifulSoup
import requests
import re
from tqdm import tqdm
import pickle
import pymongo
import tldextract

def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', str(data))


HEADERS = {
    'method': 'GET',
    'scheme': 'https',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'accept-language': 'it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7',
    'cache-control': 'max-age=0',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'
}


WIRED = {
        'title':
            {
                'object': 'h1',
                'attributes': {
                    'class': 'title'
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
            'object': 'h1',
            'attributes': {
                'class': 'c-page-title'
            }
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

        req = requests.get(href, headers=HEADERS)
        html = req.text

        soup = BeautifulSoup(html)

        out = {
            'href' : href,
            'domain': tldextract.extract(href).domain
        }

        lookfor = self.format.keys()

        for comp in lookfor:

            parsed = soup.find(self.format[comp]['object'], self.format[comp]['attributes'])

            if parsed is None:
                raise Exception("Can't find specified component '%s' in link %s" % (comp, href))

            out[comp] = striphtml(parsed)

        return out


myclient = pymongo.MongoClient("mongodb://localhost:27017/")
db = myclient["News2Titledb"]

for cfg, file in zip((VERGE, WIRED, GIZMODO), ('theverge.pkl', 'wired.pkl', 'gizomodo.pkl')):

    p = Parser(format=cfg)
    with open(file, "rb") as f:
        links = pickle.load(f)

    for link in tqdm(links):
        try:
            result = p.parse(link)
            db.news.insert_one(result)
        except Exception as e:
            print("Something went wrong with one article.. (%s)" % str(e))





