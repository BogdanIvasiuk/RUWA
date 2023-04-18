import os
import json
import glob
import pandas as pd
from info import event_dict, sources
import tqdm

class Article:
    def __init__(self, article_id, site, title, body, source, event_uri, event_date):
        self.id = article_id
        self.site = site
        self.title = title
        self.body = body
        self.source = source
        self.event_uri = event_uri
        self.event_date = event_date

    def to_dict(self):
        return {
            "id": self.id,
            "site": self.site,
            "title": self.title,
            "body": self.body,
            "source": self.source,
            "eventUri": self.event_uri,
            "eventDate": self.event_date
        }

    
class JsonParser:
    def __init__(self, path, json_fields, name, id_num):
        self.path = path
        self.json_fields = json_fields
        self.name = name
        self.id_num = id_num

    def parse(self):
        with open(self.path, encoding='utf-8') as f:
            data = json.load(f)

        articles = []
        for i,item in enumerate(data):
            for key, value in self.json_fields.items():
                setattr(self, key, item.get(value))
            id, site, title, body = getattr(self, 'id', ''), getattr(self, 'site', ''), getattr(self, 'title', ''), getattr(self, 'body', '')
            if self.name =="nina":
                id = f"{self.id_num}-{i}"
                site = item["src"]
            source, event_uri, event_date = self.parse_web(item["ref"]) if 'ref' in item else self.parse_ev(item)
            article = Article(id, site, title, body, source, event_uri, event_date)
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
    def __init__(self, path, id_num):
        json_fields = {'id': 0, 'title': 'title', 'body': 'content', 'source': 'src'}
        super().__init__(path, json_fields,"nina",id_num)




class EvJsonParser(JsonParser):
    def __init__(self, path):
        json_fields = {'id': 'uri', 'title': 'title', 'body': 'body', 
                       'source': 'source.uri', 'eventUri': 'eventUri', 'date': 'date'}
        super().__init__(path, json_fields,"ev-big",0)

class EventDataset:
    def __init__(self, path, dataset_name):
        self.path = path
        self.dataset_name = dataset_name
        self.articles = []
        self.parse_articles()

    def parse_articles(self):
        for i, file in enumerate(tqdm.tqdm(self.path)):
            if self.dataset_name == "ev_big":
                json_parser = EvJsonParser(file)
            elif self.dataset_name == "nina":
                json_parser = WebJsonParser(file, i+1)
            else:
                print("Il dataset selezionato non è previsto.")
                return
            self.articles += json_parser.parse()

    def to_dataframe(self):
        return pd.DataFrame(
            {
                "id": [a.id for a in self.articles],
                "site": [a.site for a in self.articles],
                "title": [a.title for a in self.articles],
                "body": [a.body for a in self.articles],
                "source": [a.source for a in self.articles],
                "event_uri": [a.event_uri for a in self.articles],
                "event_date": [a.event_date for a in self.articles],
            }
        )



def get_dataset(name, path_ruwa_dataset):
    excel_file = os.path.join(path_ruwa_dataset, f"{name}.xlsx")
    
    if os.path.exists(excel_file):
        print('Dataset already exists in Excel format')
        # Dataset already exists in Excel format, load it directly
        return pd.read_excel(excel_file)
    
    if name == "ev_big":
        json_files = glob.glob(os.path.join(path_ruwa_dataset, "ev-big_dataset/*.json"))
        df = EventDataset(json_files, "ev_big").to_dataframe()
        select_df = df[df['source'].isin(sources)].reset_index(drop=True)
        select_df['Event'] = select_df['event_uri'].map(event_dict)
    elif name == "nina":
        json_files = glob.glob(os.path.join(path_ruwa_dataset, "nina_dataset/*.json"))
        df = EventDataset(json_files, "nina").to_dataframe()
        df["source"] = df["source"].str.replace("https://www.ukrinform.net", "ukrinform.net")
        select_df = df[df['source'].isin(sources)].reset_index(drop=True)
        df['Event'] = df['event_uri'].map(event_dict)
        select_df = df
    else:
        raise ValueError("Dataset non previsto")
    
    # Save the dataset to Excel for future use
    select_df.to_excel(excel_file, index=False)
    print("The dataset has been created in the correct format and saved as an xlsx file.")
    
    return select_df


