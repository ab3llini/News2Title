from bs4 import BeautifulSoup
import requests
from enum import Enum
import re


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


p = Parser(format=WIRED)
result = p.parse('https://www.wired.com/review/review-coral-one-vacuum/')

print(result)
