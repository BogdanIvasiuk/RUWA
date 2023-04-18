from transformers import BertConfig, BertTokenizer, BertForSequenceClassification

"""
path_base = "E:/Bogs Life/Datasets/HAIM_Ukraine/datasets"
result_path =path_base+"/"+"results"
path_base_dataset = "E:/Bogs Life/Datasets/HAIM_Ukraine/datasets"
"""
path_base_propaganda_dataset = "/content/drive/MyDrive/Colab Notebooks/HAIM_Ukraine/PTC_CORPUS/datasets"
path_base= "/content/drive/MyDrive/Colab Notebooks/HAIM_Ukraine"
result_path =path_base+"/"+"results"
path_ruwa_dataset ="/content/drive/MyDrive/Colab Notebooks/HAIM_Ukraine/RUWA Dataset"



sources = ['ukrinform.net', 'censor.net', 'rt.com', 'news-front.info', 'bbc.com', 'euronews.com', 'nbcnews.com', 'edition.cnn.com', 'aljazeera.com', 'reuters.com', 'bloomberg.com']

events = ['Beginning of the invasion', 'Bucha', 'Kramatorsk railway station attack', 'Sinking of the Moskva', 'Siege of Azovstal', 'Kremenchuk', 'Vinnytsia', 'Olenivka', 'Nuclear terrorism', 'Ukrainian eastern counteroffensive', 'Nord Stream pipeline sabotage', 'Crimean Bridge explosion', 'Liberation of Kherson']


event_dict = {
    "eng-7623621": "Kramatorsk railway station attack",
    "eng-7637285": "Kramatorsk railway station attack",
    "eng-7639490": "Kramatorsk railway station attack",
    "eng-7637057": "Sinking of the Moskva",
    "eng-7641614": "Sinking of the Moskva",
    "eng-7644527": "Sinking of the Moskva",
    "eng-7641352": "Sinking of the Moskva",
    "eng-7716241": "Siege of Azovstal",
    "eng-7724228": "Siege of Azovstal",
    "eng-7715582": "Siege of Azovstal",
    "eng-7724776": "Siege of Azovstal",
    "eng-7704374": "Siege of Azovstal",
    "eng-7697627": "Siege of Azovstal",
    "eng-7817444": "Kremenchuk",
    "eng-7821909": "Kremenchuk",
    "eng-7815600": "Kremenchuk",
    "eng-7820773": "Kremenchuk",
    "eng-7860811": "Vinnytsia",
    "eng-7867005": "Vinnytsia",
    "eng-7868051": "Vinnytsia",
    "eng-7864141": "Vinnytsia",
    "eng-7865602": "Vinnytsia",
    "eng-7862224": "Vinnytsia",
    "eng-7903716": "Olenivka",
    "eng-7930013": "Nuclear terrorism",
    "eng-7916201": "Nuclear terrorism",
    "eng-7913137": "Nuclear terrorism",
    "eng-8014739": "Ukrainian eastern counteroffensive",
    "eng-8015186": "Ukrainian eastern counteroffensive",
    "eng-8009085": "Ukrainian eastern counteroffensive",
    "eng-8008252": "Ukrainian eastern counteroffensive",
    "eng-8053392": "Nord Stream pipeline sabotage",
    "eng-8059177": "Nord Stream pipeline sabotage",
    "eng-8052865": "Nord Stream pipeline sabotage",
    "eng-8094546": "Crimean Bridge explosion",
    "eng-8085213": "Crimean Bridge explosion",
    "eng-8166202": "Liberation of Kherson",
    "eng-8166083": "Liberation of Kherson",
    "eng-8169941": "Liberation of Kherson",
    "eng-8166048": "Liberation of Kherson",
    "sinking": "Sinking of the Moskva",
    "nuclear": "Nuclear terrorism",
    "azovstal": "Siege of Azovstal",
    "railway": "Kramatorsk railway station attack",
    "prisoners": "Olenivka",
    "supermarket": "Kremenchuk",
    "bucha": "Bucha massacre",
    "beginning": "Beginning of full-scale invasion in Ukraine",
    "theatre": "Mariupol theatre airstrike"
}


MODEL_CLASSES = {"bert": (BertConfig, BertForSequenceClassification, BertTokenizer)}


args = {"data_dir": "datasets/",
        "model_type": "bert",
        "model_name": "bert-base-uncased",
        "output_dir": result_path+"/"+"Task_SI/",
        "max_seq_length": 128,
        "train_batch_size": 8,
        "eval_batch_size": 8,
        "num_train_epochs": 1,
        "weight_decay": 0,
        "learning_rate": 4e-5,
        "adam_epsilon": 1e-8,
        "warmup_ratio": 0.06,
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "logging_steps": 50,
        "save_steps": 2000,
        "overwrite_output_dir": False}