import glob
import json
from pathlib import Path
import pandas as pd

class Article:
    def __init__(self, article_id, title, body, source, event_uri, event_date):
        self.id = article_id
        self.title = title
        self.body = body
        self.source = source
        self.event_uri = event_uri
        self.event_date = event_date

class JsonParser:
    def __init__(self, path):
        self.path = path

    def parse(self):
        with open(self.path, encoding='utf-8') as f:
            data = json.load(f)

        articles = []
        
        for item in data:
            ref = item.get('ref')
            event_uri = None
            event_date = None
            source = None
            if ref:
                source = ref[0].get('site')
                event_uri = ref[0].get('event')
                event_date = ref[0].get('dt')
            article = Article(item['src'], item['title'], item['content'], source, event_uri, event_date)
            articles.append(article)
        return articles


json_files = glob.glob("nina_dataset/*.json")
articles = []
for file in json_files:
    json_parser = JsonParser(file)
    articles += json_parser.parse()

df = pd.DataFrame(
    {
        "id": [a.id for a in articles],
        "title": [a.title for a in articles],
        "body": [a.body for a in articles],
        "source": [a.source for a in articles],
        "event_uri": [a.event_uri for a in articles],
        "event_date": [a.event_date for a in articles],
    }
)

print(df)
print(df['event_uri'].unique())
"""

path = Path("nina_dataset/bbc.json")
articles = []
with open(path, encoding='utf-8') as f:
    data = json.load(f)
    i = 0
    for item in data:
        print(i)
        if item.get('ref'):
            event_uri = item['ref'][0]['event']
            event_date = item['ref'][0]['dt']
        else:
            event_uri = None
            event_date = None

        article = Article(i, item['title'], item['content'], item['src'], event_uri, event_date)

        articles.append(article)
        i+=1

df = pd.DataFrame(
    {
        "id": [a.id for a in articles],
        "title": [a.title for a in articles],
        "body": [a.body for a in articles],
        "source": [a.source for a in articles],
        "event_uri": [a.event_uri for a in articles],
        "event_date": [a.event_date for a in articles],
    }
)
print(df)
"""