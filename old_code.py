import json
import pandas as pd
import os
import glob

ev_big = "/content/gdrive/MyDrive/Colab Notebooks/HAIM_Ukraine/RUWA Dataset/ev-big"

def article_to_dataframe(article_id, article_title, article_body, article_source, eventUri, eventDate):
    article_dict = {
        "id": article_id,
        "title": article_title,
        "body": article_body,
        "source": article_source,
        "eventUri": eventUri,
        "eventDate":eventDate

    }
    return pd.DataFrame([article_dict])

def events_to_dataframe(event_filenames):
  dataframe_list = []
   # For each article, do:
  for i in range(len(event_filenames)):

    with open(event_filenames[i], 'r') as f:
            articles = json.load(f)
 
            for article in articles:
              dataframe_list.append(article_to_dataframe(article["uri"], article["title"], article["body"], article['source']['uri'], article['eventUri'], article['date']))

  return pd.concat(dataframe_list, ignore_index=True)

article_filenames = sorted(glob.glob(os.path.join(ev_big, "*.json")))
print(len(article_filenames))



df = events_to_dataframe(article_filenames)
sources = ['ukrinform.net', 'censor.net', 'rt.com', 'news-front.info', 'bbc.com', 'euronews.com', 'nbcnews.com', 'edition.cnn.com', 'aljazeera.com', 'reuters.com', 'bloomberg.com']

select_df = df[df['source'].isin(sources)].reset_index(drop=True)

df.to_excel('/content/gdrive/MyDrive/Colab Notebooks/HAIM_Ukraine/RUWA Dataset/df_ruwa_big.xlsx', index=False)



"-------------------------------"

class Article:
    def __init__(self, article_id, title, body, source, event_uri, event_date):
        self.id = article_id
        self.title = title
        self.body = body
        self.source = source
        self.event_uri = event_uri
        self.event_date = event_date

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "source": self.source,
            "eventUri": self.event_uri,
            "eventDate": self.event_date
        }

class Event:
    def __init__(self, articles):
        self.articles = articles

    @classmethod
    def from_json(cls, event_file):
        with open(event_file, 'r') as f:
            articles_json = json.load(f)
        
        articles = []
        for article_json in articles_json:
            articles.append(Article(
                article_json["uri"],
                article_json["title"],
                article_json["body"],
                article_json['source']['uri'],
                article_json['eventUri'],
                article_json['date']
            ))
        
        return cls(articles)

    def to_dataframe(self):
        article_dicts = [article.to_dict() for article in self.articles]
        return pd.DataFrame(article_dicts)

class EventDataset:
    def __init__(self, event_folder):
        self.event_folder = event_folder
        self.event_files = sorted(glob.glob(os.path.join(self.event_folder, "*.json")))

    def get_events(self, sources=None):
        events = []
        for event_file in self.event_files:
            event = Event.from_json(event_file)
            if sources is None or any(article.source in sources for article in event.articles):
                events.append(event)
        return events

    def to_dataframe(self, sources=None):
        events = self.get_events(sources)
        dataframes = [event.to_dataframe() for event in events]
        return pd.concat(dataframes, ignore_index=True)

ev_big = "/content/gdrive/MyDrive/Colab Notebooks/HAIM_Ukraine/RUWA Dataset/ev-big"

sources = ['ukrinform.net', 'censor.net', 'rt.com', 'news-front.info', 'bbc.com', 'euronews.com', 'nbcnews.com', 'edition.cnn.com', 'aljazeera.com', 'reuters.com', 'bloomberg.com']

dataset = EventDataset(ev_big)
df = dataset.to_dataframe(sources=sources)

df.to_excel('/content/gdrive/MyDrive/Colab Notebooks/HAIM_Ukraine/RUWA Dataset/df_ruwa_big.xlsx', index=False)