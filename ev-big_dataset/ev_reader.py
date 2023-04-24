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

class EvJsonParser:
    def __init__(self, path):
        self.path = path

    def parse(self):
        articles = []
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            article = Article(item['uri'], item['title'], item['body'], item['source']['uri'], item['eventUri'], item['date'])
            articles.append(article)
        return articles

json_files = glob.glob("ev-big_dataset/*.json")
articles = []
for file in json_files:
    json_parser = EvJsonParser(file)
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