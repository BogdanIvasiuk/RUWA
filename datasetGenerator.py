import os
import glob
import csv
import codecs
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer


class Features:
    def __init__(self, max_seq_length, tokenizer):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def _concatenate_tokens(self, tokens):
        return [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]

    def _compute_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def _zero_pad_up(self, input_vector, padding_value):
        num_pads = self.max_seq_length - len(input_vector)
        return input_vector + [padding_value] * num_pads

    def _assert_same_length(self, input_ids, input_mask, segment_ids):
        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

    def convert_dataframe_to_features(self, dataframe):
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
            tokens = self.tokenizer.tokenize(dataframe["text"][i])

            # If length of the sequence is greater than max sequence length, truncate it
            if len(tokens) > self.max_seq_length - 2:
                tokens = tokens[:(self.max_seq_length - 2)]

            # Concatenate the tokens
            tokens = self._concatenate_tokens(tokens)

            # Compute the ids
            input_ids = self._compute_ids(tokens)
            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            input_ids = self._zero_pad_up(input_ids, self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0])
            input_mask = self._zero_pad_up(input_mask, 0)
            segment_ids = self._zero_pad_up(segment_ids, 0)
            label_id = dataframe["label"][i]

            # Assert to make sure we have same length
            self._assert_same_length(input_ids, input_mask, segment_ids)

            # Put the data into features dataframe
            features["input_ids"][i] = input_ids
            features["input_mask"][i] = input_mask
            features["segment_ids"][i] = segment_ids
            features["label_ids"][i] = label_id

        return features



class DatasetsGenerator:
    def __init__(self, path_base, tokenizer, task_type="binary", max_seq_length=512):
        """
        Initializes the GeneratedDataset class with articles and labels folders,
        a tokenizer, task type (default is binary) and maximum sequence length (default is 512)
        """
        self.path_base = path_base

        self.tokenizer = tokenizer
        self.task_type = task_type
        self.max_seq_length = max_seq_length
        self.features_converter = Features(max_seq_length, tokenizer)

    def generate_dataset(self, logger, dataset_type="train"):
        """
        Generates dataset from articles and labels
        """
        dataset_path = os.path.join(self.path_base, f"{dataset_type}_dataset.pt")
        dataframe_path = os.path.join(self.path_base, f"{dataset_type}_dataframe.pkl")
        
        if os.path.exists(dataset_path) and os.path.exists(dataframe_path):
            logger.info(f"{dataset_type} dataset and dataframe already exist in {self.path_base}")
            dataset = torch.load(dataset_path)
            dataframe = pd.read_pickle(dataframe_path)
            return dataset, dataframe
        
        if dataset_type == "train":
            logger.info("Generating training dataset...")
            self.articles_folder = self.path_base + "/"  + "train-articles"
            self.labels_folder = self.path_base + "/"   + "train-labels-task-si"
            
        elif dataset_type == "val":
            logger.info("Generating validation dataset...")
            self.articles_folder = self.path_base + "/"  + "dev-articles"
            self.labels_folder = self.path_base + "/"   + "dev-labels-task-si"
        else:
            raise ValueError("dataset_type can only be 'train' or 'val'")

        # For each articles and labels folder set, turn them into dataframes

        df_temp = self.articles_to_dataframe(self.articles_folder, self.labels_folder)

        if self.task_type == "binary":
            propaganda_df = df_temp[df_temp["label"] == 1]
            # Duplicate the propaganda sentences
            dataframe = pd.concat([df_temp, propaganda_df], ignore_index=True)
        else:
            dataframe = df_temp

        # Process into features dataframe
        logger.info("Creating features from dataframe")
        features = self.features_converter.convert_dataframe_to_features(dataframe)

        # Creating TensorDataset from features
        logger.info("Creating TensorDataset from features dataframe")
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids = self._create_tensors(features)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Saving dataset and dataframe to path_base directory
        if not os.path.exists(self.path_base):
            os.makedirs(self.path_base)
        
        torch.save(dataset, dataset_path)
        dataframe.to_pickle(dataframe_path)
        print(f"The torch dataset has been created and saved in {dataset_path} ")

        return dataset, dataframe

    def _create_tensors(self, features):
        """
        Creates tensors from features dataframe
        """
        all_input_ids = torch.tensor(features["input_ids"], dtype=torch.long)
        all_input_mask = torch.tensor(features["input_mask"], dtype=torch.long)
        all_segment_ids = torch.tensor(features["segment_ids"], dtype=torch.long)
        all_label_ids = torch.tensor(features["label_ids"], dtype=torch.long)

        return all_input_ids, all_input_mask, all_segment_ids, all_label_ids


    def articles_to_dataframe(self, article_folder, label_folder):
        """
        Preprocesses the articles into dataframes with sequences with binary tags
        """
        # First sort the filenames and make sure we have label file for each articles
        article_filenames = sorted(glob.glob(os.path.join(article_folder, "*.txt")))
        label_filenames = sorted(glob.glob(os.path.join(label_folder, "*.labels")))
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
                indices_list = []
                for row in reader:
                    indices_list.append([int(row[1]), int(row[2])])

                # Merge the indices if overlapping
                indices_list = DatasetsGenerator.merge_overlapping(indices_list)

            # Add to the sequences
            
            sequences.append(DatasetsGenerator.article_labels_to_sequences(article, indices_list))
            if not sequences:
                return None

        # Concatenate all dataframes
        dataframe = pd.concat(sequences, ignore_index=True)
        return dataframe


    def merge_overlapping(indices):
        """
        Merges overlapping indices and sorts indices from list of tuples
        """
        # If no propaganda, return empty list
        if not indices:
            return []

        # Sort the list by the start index of each interval
        sorted_indices = sorted(indices, key=lambda x: x[0])
        result = [sorted_indices[0]]

        # Iterate over each interval in the list and merge overlapping intervals
        for start, end in sorted_indices[1:]:
            last_end = result[-1][1]
            # If the current interval overlaps with the last one, merge them
            if start <= last_end:
                result[-1][1] = max(last_end, end)
            # Otherwise, add the current interval to the result list
            else:
                result.append([start, end])

        return result


    def article_labels_to_sequences(article, indices_list):
        """
        Divides article into sequences, where each are tagged to be propaganda or not
        """
        # Start at 0 indices, and split the article into lines
        curr = 0
        lines = article.split("\n")
        sequences = {}

        # For each lines, do:
        for line in lines:
            # If an empty line, just continue after adding \n character
            if line == "":
                curr += 1
                continue

            # If nothing in indices_list or current line is not part of propaganda, 
            # just mark it to be none 
            elif indices_list == [] or curr + len(line) <= indices_list[0][0]:
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

        dataframe = pd.DataFrame(None, range(len(sequences)), ["label", "text"])
        dataframe["label"] = sequences.values()
        dataframe["text"] = sequences.keys()
        return dataframe
    



