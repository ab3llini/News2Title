import pandas as pd
import re
import requests
import pickle


class LinkBuilder:

    def __init__(self, tweets, href):
        self.tweets = tweets
        self.href = href
        self.unprocessed = []
        self.next = []

        self.data = list(pd.read_pickle(tweets)[2])

        for e in self.data:
            urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-&(-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                              str(e))
            self.unprocessed.append(urls)

    def nlinks(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):

        while len(self.next) == 0:
            self.next = self.unprocessed.pop()

        curr = self.next.pop()

        try :
            final = requests.get(curr).url

            if self.href in final:
                return final
            else:
                return self.__next__()

        except Exception as e:
            print('Unable to fetch link %s' % curr)
            return self.__next__()


pkls = [
    '../twitter_scraper/verge_tweets_2018-10-03.pkl',
    '../twitter_scraper/Gizmodo_tweets_2018-10-03.pkl',
    '../twitter_scraper/techcrunch_tweets_2018-10-03.pkl',
    '../twitter_scraper/WIRED_tweets_2018-10-03.pkl'
]

hrefs = [
    'https://www.theverge.com/',
    'https://gizmodo.com/',
    'https://www.wired.com/'
]

dsnames = [
    'theverge',
    'gizomodo',
    'wired'
]

backup_interval = 10

for tweets, href, fname in zip(pkls, hrefs, dsnames):
    i = 1

    print('Working on %s, href = %s' % (tweets, href))
    links = LinkBuilder(tweets=tweets, href=href)
    n = links.nlinks()
    fetched = []
    file = fname + '.pkl'

    for link in links:
        print("%s/%s - New link = %s" % (i, n, link))
        fetched.append(link)

        if i % backup_interval == 0:
            print('Saving links so far into file %s..' % file)
            with open(fname, 'wb') as f:
                pickle.dump(fetched, f)

        i += 1

