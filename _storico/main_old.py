from info import *
import torch
from datasetGenerator import DatasetsGenerator
from training import TrainingSpace
from utils import save_model
from newsArticlesStandardizer import get_dataset
from calculation import Computation
from logging import getLogger
from tqdm import tqdm


logger = getLogger(__name__)
device = torch.device("cuda")



# Set the number of steps for the progress bar
num_steps = 10

# Create the progress bar with the total number of steps
with tqdm(total=num_steps) as pbar:
    # Step 1
    print("Configuring model...")
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args["model_type"]]
    config = config_class.from_pretrained(args["model_name"], num_labels=2, finetuning_task="binary")


    pbar.update(1)

    # Step 2
    print("Loading tokenizer...")
    tokenizer = tokenizer_class.from_pretrained(args["model_name"])
    pbar.update(1)

    # Step 3
    print("Loading model...")
    model_propaganda = model_class(config)
    model_propaganda.to(device)
    pbar.update(1)

    # Step 4
    print("Preparing data...")
    datasetGenerator = DatasetsGenerator(path_base_propaganda_dataset, tokenizer)
    train_dataset, df_train = datasetGenerator.generate_dataset(logger,"train")
    pbar.update(1)

    # Step 5
    print("Training model...")
    global_step, tr_loss, model_propaganda = TrainingSpace(args).train(train_dataset, model_propaganda, tokenizer,logger)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    logger.info("Saving model checkpoint to %s", args['output_dir'])
    pbar.update(1)

    # Step 6
    print("Saving model...")
    save_model(args, model_propaganda, tokenizer, config, global_step,tr_loss, logger)
    pbar.update(1)

    # Step 7
    print("Evaluating model...")
    print('Da fare ancora')
    #eval_results = evaluate(args, model, tokenizer)
    pbar.update(1)

    # Step 8
    print("Generating dataset for prediction...")
    #nameDataset="nina"
    nameDataset="ev_big"
    df_articles_of_events = get_dataset(nameDataset, path_ruwa_dataset)

    pbar.update(1)

    # Step 9
    print("Doing and saving predictions...")
    Computation(events, df_articles_of_events, model_propaganda, tokenizer, device, result_path,nameDataset)
    #save_predictions(args, predictions)
    pbar.update(1)

    # Step 10
    print("Done!")
    pbar.update(1)

