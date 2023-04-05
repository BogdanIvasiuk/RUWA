import pandas as pd
import torch
from config import args,logger
from torch.utils.data import TensorDataset
import codecs
import os
import glob
import csv


import pandas as pd
import glob
import os
import csv
import codecs


class ArticleProcessor:
    def __init__(self, article_folder, label_folder):
        self.article_folder = article_folder
        self.label_folder = label_folder

    def process_articles(self):
        """
        Preprocesses the articles into dataframes with sequences with binary tags
        """
        # First sort the filenames and make sure we have label file for each articles
        article_filenames = sorted(glob.glob(os.path.join(self.article_folder, "*.txt")))
        label_filenames = sorted(glob.glob(os.path.join(self.label_folder, "*.labels")))
        assert len(article_filenames) == len(label_filenames)

        # Initialize sequences
        sequences = []

        # For each article, do:
        for i in range(len(article_filenames)):
            # Get the id name
            article_id = os.path.basename(article_filenames[i]).split(".")[0][7:]

            # Read in the article
            with codecs.open(article_filenames[i], "r", encoding="utf8") as f:
                article = f.read()

            # Read in the label file and store indices for SI task
            with open(label_filenames[i], "r") as f:
                reader = csv.reader(f, delimiter="\t")
                indices_list = [[int(row[1]), int(row[2])] for row in reader]

            # Merge the indices if overlapping
            indices_list = self.merge_overlapping(indices_list)

            # Add to the sequences
            sequences.append(self.article_labels_to_sequences(article, indices_list))

        # Concatenate all dataframes
        dataframe = pd.concat(sequences, ignore_index=True)

        return dataframe

    @staticmethod
    def article_labels_to_sequences(article, indices_list):
        """
        Divides article into sequences, where each are tagged to be propaganda or not
        """
        # Start at 0 indices, and split the article into lines
        curr = 0
        lines = article.split("\n")
        sequences = {}

        # For each line, do:
        for line in lines:
            # If an empty line, just continue after adding \n character
            if line == "":
                curr += 1
                continue

            # If nothing in indices_list or current line is not part of propaganda,
            # just mark it to be none
            elif not indices_list or curr + len(line) <= indices_list[0][0]:
                sequences[line] = 0

            # If current line is part of propaganda, do:
            else:
                # If the propaganda is contained within the line, add it accordingly
                # and pop that indices range
                if curr + len(line) >= indices_list[0][1]:
                    sequences[line] = 1
                    indices_list.pop(0)

            # Add the current line length plus \n character
            curr += len(line) + 1

        dataframe = pd.DataFrame({"label": list(sequences.values()), "text": list(sequences.keys())})
        return dataframe

    @staticmethod
    def merge_overlapping(indices_list):
        """
        Merges overlapping indices and sorts indices from list of tuples
        """
        # If no propaganda, return empty list
        if not indices_list:
            return []

        # Sort the list
        indices_list = sorted(indices_list)
        merged_indices = [indices_list[0]]

        # Merge overlapping ranges
        for start, end in indices_list[1:]:
            last_start, last_end = merged_indices[-1]
            if start <= last_end:
                merged_indices[-1] = (last_start, max(end, last_end))
        else:
            merged_indices.append((start, end))
        
        return merged_indices



def convert_dataframe_to_features(dataframe, max_seq_length, tokenizer):
  """
  Converts dataframe into features dataframe, where each feature will
  take form of [CLS] + A + [SEP]
  """
  # Create features
  features = pd.DataFrame(None, range(dataframe.shape[0]), 
                              ["input_ids", "input_mask", "segment_ids", "label_ids"])

  # For each sequence, do:
  for i in range(len(dataframe)):
    # Set first and second part of the sequences
    tokens = tokenizer.tokenize(dataframe["text"][i])

    # If length of the sequence is greater than max sequence length, truncate it
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]

    # Concatenate the tokens
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]

    # Compute the ids
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    input_ids = input_ids + [pad_token] * padding_length
    input_mask = input_mask + [0] * padding_length
    segment_ids = segment_ids + [0] * padding_length
    label_id = dataframe["label"][i]

    # Assert to make sure we have same length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # Put the data into features dataframe
    features["input_ids"][i] = input_ids
    features["input_mask"][i] = input_mask
    features["segment_ids"][i] = segment_ids
    features["label_ids"][i] = label_id

  return features

def generate_training_dataset_from_articles(articles_folders, labels_folders, tokenizer, train="train"):
  """
  Generates dataset to go into BERT from articles and labels
  """
  # If generating dataset for evaluation, do:
  logger.info("Generating training dataset...")
    
  # For each articles and labels folder set, turn them into dataframes
  dataframe_list = []
  for i in range(len(articles_folders)):
    logger.info("Generating dataframe for folder %s", articles_folders[i])
    dataframe_list.append(articles_to_dataframe(articles_folders[i], labels_folders[i]))

  # Concatenate the dataframes to make a total dataframe
  if train == "train":
    df_temp = pd.concat(dataframe_list, ignore_index=True)
    propaganda_df = df_temp[df_temp["label"] == 1]
    #duplicate the propaganda sentences
    dataframe  = pd.concat([df_temp, propaganda_df], ignore_index=True)
  else:
    dataframe = pd.concat(dataframe_list, ignore_index=True)


  # Process into features dataframe
  logger.info("Creating features from dataframe")
  features = convert_dataframe_to_features(dataframe, args['max_seq_length'], tokenizer)
      
  # Creating TensorDataset from features
  logger.info("Creating TensorDataset from features dataframe")
  all_input_ids = torch.tensor(features["input_ids"], dtype=torch.long)
  all_input_mask = torch.tensor(features["input_mask"], dtype=torch.long)
  all_segment_ids = torch.tensor(features["segment_ids"], dtype=torch.long)
  all_label_ids = torch.tensor(features["label_ids"], dtype=torch.long)

  dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
  return dataset, dataframe