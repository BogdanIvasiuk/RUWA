import os
import json
import glob
import pandas as pd
from info import event_dict, sources

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

    
class JsonParser:
    def __init__(self, path, json_fields):
        self.path = path
        self.json_fields = json_fields

    def parse(self):
        with open(self.path, encoding='utf-8') as f:
            data = json.load(f)

        articles = []
        for item in data:
            for key, value in self.json_fields.items():
                setattr(self, key, item.get(value))
            id, title, body = getattr(self, 'id', ''), getattr(self, 'title', ''), getattr(self, 'body', '')
            source, event_uri, event_date = self.parse_web(item["ref"]) if 'ref' in item else self.parse_ev(item)
            article = Article(id, title, body, source, event_uri, event_date)
            articles.append(article)

        return articles
        
    def parse_web(self, ref):
        source, event_uri, event_date = None, None, None
        if ref:
            source = ref[0].get('site')
            event_uri = ref[0].get('event')
            event_date = ref[0].get('dt')
        return source, event_uri, event_date
    
    def parse_ev(self, item):
        source_fields = self.json_fields["source"]
        source = item.get(source_fields.split('.')[0], {}).get(source_fields.split('.')[1])
        event_uri = item.get(self.json_fields["eventUri"])
        event_date = item.get(self.json_fields["date"])
        return source, event_uri, event_date



class WebJsonParser(JsonParser):
    def __init__(self, path):
        json_fields = {'id': 'src', 'title': 'title', 'body': 'content', 'source': 'src'}
        super().__init__(path, json_fields)



class EvJsonParser(JsonParser):
    def __init__(self, path):
        json_fields = {'id': 'uri', 'title': 'title', 'body': 'body', 
                       'source': 'source.uri', 'eventUri': 'eventUri', 'date': 'date'}
        super().__init__(path, json_fields)

class EventDataset:
    def __init__(self, path, dataset_name):
        self.path = path
        self.dataset_name = dataset_name
        self.articles = []
        self.parse_articles()

    def parse_articles(self):
        for file in self.path:
            if self.dataset_name == "ev_big":
                json_parser = EvJsonParser(file)
            elif self.dataset_name == "nina":
                json_parser = WebJsonParser(file)
            else:
                print("Il dataset selezionato non è previsto.")
                return
            self.articles += json_parser.parse()

    def to_dataframe(self):
        return pd.DataFrame(
            {
                "id": [a.id for a in self.articles],
                "title": [a.title for a in self.articles],
                "body": [a.body for a in self.articles],
                "source": [a.source for a in self.articles],
                "event_uri": [a.event_uri for a in self.articles],
                "event_date": [a.event_date for a in self.articles],
            }
        )

def get_dataset(name):
    if name == "ev_big":
        json_files = glob.glob("ev-big_dataset/*.json")
        df = EventDataset(json_files, "ev_big").to_dataframe()
        print(df)
        select_df = df[df['source'].isin(sources)].reset_index(drop=True)
        select_df['Event'] = select_df['event_uri'].map(event_dict)
        return select_df
    elif name == "nina":
        json_files = glob.glob("nina_dataset/*.json")
        df = EventDataset(json_files, "nina").to_dataframe()
        df["source"] = df["source"].str.replace("https://www.ukrinform.net", "ukrinform.net")
        df['Event'] = df['event_uri'].map(event_dict)
        return df
    else:
        raise ValueError("Dataset non previsto")

name="ev_big"
#name="nina"
df = get_dataset(name)
print(df['source'].unique())