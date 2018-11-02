#!/usr/bin/env python
# encoding: utf-8

import tweepy  # https://github.com/tweepy/tweepy
import logging

# Twitter API credentials
consumer_key = "lQrEBmxVYSraCOFGsp81vIbIP"
consumer_secret = "JnK0WtrfxKT6AwMC6JTMOw4bQRlC4JaQp0wqV8VCC19v4AF85v"
access_key = "520288508-LyAE9FV7slne7GYrygJphhjghWshAkbhucokJ64W"
access_secret = "q19Wbx0Z2X2bUZF6nbwZQMiuHkyKVE0Wi1bHirEmbhtZ7"


def get_all_tweets(screen_name):
    import os
    path = os.path.dirname(os.path.abspath(__file__))
    logging.basicConfig(filename=path+'/../logs/twitter_scraper.log', level=logging.DEBUG)

    # Twitter only allows access to a users most recent 3240 tweets with this method

    # authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # initialize a list to hold all the tweepy Tweets
    alltweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name=screen_name, count=200)

    # save most recent tweets
    alltweets.extend(new_tweets)

    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:

        logging.info("getting tweets before %s" % (oldest))

        # all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest)

        # save most recent tweets
        alltweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        logging.info("...%s tweets downloaded so far" % (len(alltweets)))

    # transform the tweepy tweets into a 2D array that will populate the csv
    outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]
    return outtweets
#profile_list=['VitalikButerin','aantonop','SatoshiLite','fluffypony','lopp','TraceMayer','maxkeiser','diaryofamademan']
#def update_tweet_database(profile_list):
    #for profile in profile_list:
    #    get_all_tweets(profile)

import datetime
import pandas as pd

def get_all_news_in_dataframe(twitter_id:str):
    tweet_list = get_all_tweets(twitter_id)
    print(tweet_list[-1])
    print(len(tweet_list))
    today = datetime.date.today()
    bloomberg_tweet_dataframe = pd.DataFrame(tweet_list)
    bloomberg_tweet_dataframe.to_pickle(twitter_id+'_tweets_'+today.__str__()+'.pkl')

list= ['WIRED','Gizmodo','verge']
for element in list:
    get_all_news_in_dataframe(element)
