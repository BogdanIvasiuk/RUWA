from info import *
import torch
from datasetGenerator import DatasetsGenerator
from utils import save_model
from newsArticlesStandardizer import get_dataset
from calculation import Computation
from logging import getLogger
from tqdm import tqdm
from models import ModelPropaganda


logger = getLogger(__name__)
device = torch.device("cuda")




# Set the number of steps for the progress bar
num_steps = 7

# Create the progress bar with the total number of steps
with tqdm(total=num_steps) as pbar:
    # Step 1
    print("Configuring model...")
    propaganda = ModelPropaganda(args=args,logger=logger, device=device)
    pbar.update(1)
    # Step 2
    print("Loading tokenizer and model...")
    model, tokenizer, config = propaganda.load_model()
    pbar.update(1)
    # Step 3
    print("Preparing data...")
    datasetGenerator = DatasetsGenerator(path_base_propaganda_dataset, tokenizer)
    train_dataset, df_train = datasetGenerator.generate_dataset(logger,"train")
    pbar.update(1)
    # Step 4
    print("Training model...")
    global_step, tr_loss, model_propaganda = propaganda.load_or_train(train_dataset)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    logger.info("Saving model checkpoint to %s", args['output_dir'])
    pbar.update(1)
    # Step 5
    print("Saving model...")
    save_model(args, model_propaganda, tokenizer, config, global_step,tr_loss, logger)
    pbar.update(1)
    # Step 6
    print("Generating dataset for prediction...")
    #nameDataset="nina"
    nameDataset="ev_big"
    df_articles_of_events = get_dataset(nameDataset, path_ruwa_dataset)
    pbar.update(1)
    # Step 7
    print("Doing and saving predictions...")
    Computation(events, df_articles_of_events, model_propaganda, tokenizer, device, result_path,nameDataset)
    #save_predictions(args, predictions)
    pbar.update(1)