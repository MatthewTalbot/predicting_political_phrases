import tweepy
import pandas as pd
CONSUMER_KEY = "MaXotFe9IKBjaAg3UpEChYxZi"
CONSUMER_SECRET = "dgVHFG3CfH0rStZSfM4smHSHRQW00dmaRPjKOGum5opyuJJsmv"
ACCESS_TOKEN = "1318646990150971392-zDn6CZeEGYKfpnTMQ1Z9ZPv7TL4HnS"
ACCESS_SECRET = "CiFAgH3KdFXDS8EXoinn7UIZjK6PzG0qBn4QoJL77uBFC"
AUTH = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
AUTH.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
API = tweepy.API(AUTH, wait_on_rate_limit_notify=True, wait_on_rate_limit=True)

class DataMining:
  def __init__(self) -> None:
    self.house_democrat_list_id = 110250128
    self.senate_democrat_list_id = 7465836
    self.house_republican_list_id = 817470159027929089
    self.senate_republican_list_id = 7466072

  def get_list_members(self, id, party):
    members = []
    for page in tweepy.Cursor(API.list_members, list_id = id).items():
      members.append(page)
    
    return {(m.screen_name, API.get_user(m.screen_name).id_str, party) for m in members}
  
  def get_all_democrats(self):
    house_democrats = self.get_list_members(self.house_democrat_list_id, party="Democrat")
    senate_democrats = self.get_list_members(self.senate_democrat_list_id, party="Democrat")

    return house_democrats.union(senate_democrats)
  
  def get_all_republicans(self):
    house_republicans = self.get_list_members(self.house_republican_list_id, party="Republican")
    senate_republicans = self.get_list_members(self.senate_republican_list_id, party="Republican")

    return house_republicans.union(senate_republicans)

  def save_politicians_info(self):
    democrats = self.get_all_democrats()
    republicans = self.get_all_republicans()
    politicians = list(democrats.union(republicans))
    user_names = []
    user_ids = []
    user_party = []

    for pol in politicians:
      user_names.append(pol[0])
      user_ids.append(pol[1])
      user_party.append(pol[2])
    
    save_politicians = {"Name": user_names, "Twitter ID": user_ids, "Party": user_party}
    df = pd.DataFrame(save_politicians, columns=["Name", "Twitter ID", "Party"])
    df.to_csv("pollitician_info.csv", index=False)

  def collect_tweets(self):
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
def main():
  data_mining = DataMining()
  data_mining.save_politicians_info()
  data_mining.collect_tweets()

main()
