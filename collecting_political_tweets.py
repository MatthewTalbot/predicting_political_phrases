import tweepy
import pandas as pd
CONSUMER_KEY = "dNFCPVZyPMJd0JoVcVip0ZDQg"
CONSUMER_SECRET = "NeW7sPAL9DNsM2wR1fZrlMQtC6dSsaRD8ktfwrvDFTxDejf1tW"
ACCESS_TOKEN = "1318646990150971392-R56C7ZdiMkmKrX6LGbs1VUVCRVXwan"
ACCESS_SECRET = "6JIfWa6SHwbeqceHYIrOx9chFc9WBzXhILTO5mJW82nMg"
AUTH = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
AUTH.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
API = tweepy.API(AUTH, wait_on_rate_limit_notify=True, wait_on_rate_limit=True)

user_tweets = []
politicians = pd.read_csv("pollitician_info.csv")
politicians_tweets = pd.DataFrame(columns=["Name", "Twitter ID","Time Of Tweet", "Tweet", "Political Party"])

for _, row in politicians.iterrows():
  new_tweets = API.user_timeline(screen_name = row["Name"], count = 10, tweet_mode = "extended")
  user_tweets.extend(new_tweets)
  
  for tweet in user_tweets:
    new_row = {"Name": row["Name"], "Twitter ID": row["Twitter ID"], "Time Of Tweet": tweet.created_at, "Tweet": tweet.full_text, "Political Party": row["Party"]}
    politicians_tweets = politicians_tweets.append(new_row, ignore_index=True)
    politicians_tweets.to_csv("shortned_twitter_logs.csv")

  print("User: " + row["Name"] + " completed")

