from info import events
from metrics import *
import os
import numpy as np
import pandas as pd
from itertools import combinations
import logging

def add_columns(temp_df):
    temp_df['numbered_body'] = ''
    temp_df['number of sentences'] = ''
    temp_df['number of Prop sentences'] = ''
    temp_df['Propaganda indices'] = ''
    temp_df['freq of propaganda'] = ''
    
    return temp_df


def check_similarity_embedings(event, article_id, sentences, folder_path, similarity_calculator):
    ## Method Description
    """
     If the embeddings file exists, the embeddings are loaded from the file using `np.load()`. 
     Otherwise, the embeddings are computed using the SentenceTransformer model's `encode()` method,
       and then saved to the embeddings file using `np.save()`.
    """

    """ Parameters:
    - `article_id` (str): An article id
    - `sentences` (List[str]): A list of strings representing sentences.
    - `folder_path` (str): A string representing the path to the directory where embeddings will be saved.

    """
    
    embeddings_folder = os.path.join(folder_path, event)
    os.makedirs(embeddings_folder, exist_ok=True)
    embeddings_filename = os.path.join(embeddings_folder, f"{article_id}_embeddings.npy")
    if os.path.exists(embeddings_filename):
        # Load embeddings from file
        sentence_embs = np.load(embeddings_filename)
    else:
        # Encode each sentence and save embeddings to file
        sentence_embs = similarity_calculator.get_embeddings(sentences)
        # Save embeddings to file
        np.save(embeddings_filename, sentence_embs)

    return sentence_embs


def check_propaganda(article_id, idx, temp_df, sentences, propaganda_predictor):
    # Calculate propaganda indices
    try:
        propaganda_indices_text, numbered_body, propaganda_indices = propaganda_predictor.propaganda_detection(sentences)
    except Exception as e:  # cattura qualsiasi eccezione
        print(f"Error in propaganda computation with article {article_id}: {sentences}\n{str(e)}")
        return
    # Trova l'indice numerico della colonna "Numbered_body"
    temp_df.at[idx, 'numbered_body'] = numbered_body
    temp_df.at[idx, 'number of sentences'] = len(sentences)
    temp_df.at[idx, 'number of Prop sentences'] = len(propaganda_indices)
    temp_df.at[idx, 'Propaganda indices'] = propaganda_indices_text
    temp_df.at[idx, 'freq of propaganda'] = len(propaganda_indices) / len(sentences)

def put_similarity_score(temp_df, idx, row, mean_score):
    id = row['id']
    source = row['source']
    temp_df.at[idx, f'similarity with article {source}_{id}'] = mean_score





def calculate_similarity_matrix(event,idx1,row1, temp_df, sentences1, sentence_embs1, folder_path, similarity_calculator):
    try:
        # Calculate the cosine similarity with the other articles in the temporary dataframe
        for idx2, row2 in temp_df.iterrows():
            if row1['id'] == row2['id'] or row1['source'] == row2['source']:
                mean_similarity_score=0
                put_similarity_score(temp_df, idx1, row2, mean_similarity_score)
                continue

            sentences2 = [s.strip() for p in str(row2['body']).split('\n') for s in p.split('.') if s.strip()]
            min_num_sentences = min(len(sentences1), len(sentences2))
            article_id = row2['id']
            # Check if embeddings file already exists for second article
            
            sentence_embs2 = check_similarity_embedings(event, article_id, sentences2, folder_path, similarity_calculator)
            mean_similarity_score = similarity_calculator.get_mean_similarity(sentence_embs1,sentence_embs2,min_num_sentences)
            put_similarity_score(temp_df, idx1, row2, mean_similarity_score)
    except Exception as e:
        logging.error(f"Error in calculate_similarity_matrix: {e}. Parameters used: event={event}, idx1={idx1}, row1={row1}, temp_df={temp_df}, sentences1={sentences1}, sentence_embs1={sentence_embs1}, folder_path={folder_path}, similarity_calculator={similarity_calculator}")



def calculate_similarity_by_source(df_input):
  # Melt the DataFrame to convert it to long format
  df = df_input.copy()

  df = df.melt(id_vars=['source', 'id'], var_name='similarity_with_article', value_name='score')

  # Extract the source from the column name
  df['source_2_temp'] = df['similarity_with_article'].str.split('article ').str[-1]
  
  df['source_2'] = df['source_2_temp'].str.split('_').str[0]
  df['id_2'] = df['source_2_temp'].str.split('_').str[1]

  all_sources = df['source'].unique()
  source_combinations = [(source1, source2) for source1 in all_sources for source2 in all_sources]

  # Create an empty list to store the DataFrames
  dfs = []

  # Loop through the source combinations
  for source1, source2 in source_combinations:
      # Create a new DataFrame for the source combination
      df_temp = df.loc[(df['source'] == source1) & (df['source_2'] == source2)]
      # Append the new DataFrame to the list
      n = min(df_temp['id'].nunique(), df_temp['id_2'].nunique())
      # Sort the dataframe by score in descending order
      df_sorted = df_temp.sort_values('score', ascending=False)
      # Calculate the mean of the first n values
      mean_score = df_sorted.head(n)['score'].mean()
      df_temp = df_temp.assign(mean_score=mean_score)
  


      dfs.append(df_temp)

  # Concatenate the list of DataFrames into a single DataFrame
  try:
    result_intermediate = pd.concat(dfs, ignore_index=True)
    result_final = result_intermediate.groupby(['source', 'source_2']).first().reset_index()[['source', 'source_2', 'mean_score']]
  except ValueError:
    result_intermediate = pd.DataFrame()
    result_final = pd.DataFrame(columns=['source', 'source_2', 'mean_score'])
  
  return result_intermediate, result_final




def export_data(temp_df, event, excel_path, name):
    # Calculate similarity between sources and extract final scores
    try:
        df_source_final = calculate_similarity_by_source(temp_df)[1]
    except Exception as e:
        df_source_final = pd.DataFrame()
        logging.error(f"Error calculating similarity: {e}")
        
    # Create a pivot table from df_source_final
    try:
        pivot_table = df_source_final.pivot_table(index='source', columns='source_2', values='mean_score')
    except Exception as e:
        pivot_table = pd.DataFrame()
        logging.error(f"Error creating pivot table: {e}")


    # Compute mean value of "freq of propaganda" for each source
    try:
        temp_df['freq of propaganda'] = temp_df['freq of propaganda'].replace('', 0).astype(float)
        source_mean = temp_df.groupby('source')['freq of propaganda'].mean()
        propaganda_df = pd.DataFrame({'%propaganda': source_mean.values}, index=source_mean.index)
    except Exception as e:
        propaganda_df = pd.DataFrame()
        logging.error(f"Error computing frequency of propaganda: {e}")
        logging.error(f"The error occurred at index {temp_df['freq of propaganda'].index[temp_df['freq of propaganda'].apply(lambda x: isinstance(x, str))]}")
    

    # Compute number of articles for each source and merge into propaganda_df
    try:
        article_counts = temp_df.groupby('source').size().reset_index(name='num_articles')
        propaganda_df = propaganda_df.merge(article_counts, on='source', how='left')
    except Exception as e:
        propaganda_df = pd.DataFrame()
        logging.error(f"Error computing article counts: {e}")
        logging.error(f"The error occurred at index {temp_df.index[temp_df['source'].isnull()].tolist()}")
    

    # Compute mean value of "number of sentences" for each source and merge into propaganda_df
    try:
        temp_df['number of sentences'] = temp_df['number of sentences'].replace('', 0).astype(int)
        mean_sentences = temp_df.groupby('source')['number of sentences'].mean()
        propaganda_df = propaganda_df.merge(mean_sentences, on='source', how='left')
    except Exception as e:
        logging.error(f"Error computing mean number of sentences: {e}")
        logging.error(f"The error occurred at index {temp_df['number of sentences'].index[temp_df['number of sentences'].apply(lambda x: isinstance(x, str))]}")
    

    # Construct file path by concatenating excel_path and Excel file name
    file_name = f"{event.replace(' ', '_')}_results.xlsx"
    file_path = os.path.join(excel_path, name, file_name)

    if not os.path.exists(os.path.join(excel_path, name)):
            os.makedirs(os.path.join(excel_path, name))
    

    # Write DataFrames to Excel file
    try:
        with pd.ExcelWriter(file_path) as writer:
            temp_df.to_excel(writer, sheet_name='df', index=False)
            df_source_final.to_excel(writer, sheet_name='Source_final', index=False)
            pivot_table.to_excel(writer, sheet_name='Source_final_pivot', index=True)
            propaganda_df.to_excel(writer, sheet_name='Propaganda_results', index=True)
    except Exception as e:
        logging.error(f"Error writing DataFrames to Excel file: {e}")

    # Return None
    return None

def count_empty_sentences(text):
    text = str(text)  # convert to string
    sentences = []
    for p in text.split('\n'):
        for s in p.split('.'):
            if s.strip():
                sentences.append(s.strip())
            else:
                sentences.append("EMPTY SENTENCE")

    empty_sentences = [s for s in sentences if s == "EMPTY SENTENCE"]
    num_empty_sentences = len(empty_sentences)
    print(f"Numero di frasi vuote: {num_empty_sentences}")
    print("Frasi vuote:")
    for s in empty_sentences:
        print("- " + s)

    return sentences




def Computation(events, df_articles_of_events, model_propaganda, tokenizer, device, result_path, name):
    # Create instances of propaganda predictor and similarity calculator
    propaganda_predictor = PredictPropaganda(model_propaganda, tokenizer, device)
    similarity_calculator = CalculateSimilarity()
    similarity_path = result_path + '/' + 'sim_embeddings'
    # Saving embeddings to similarity_path directory
    if not os.path.exists(similarity_path):
            os.makedirs(similarity_path)


    for event in events:
        print(f'Analysis of event:{event}')
        temp_df = df_articles_of_events[df_articles_of_events['Event'] == event].reset_index(drop=True)
        temp_df = temp_df.sort_values('source', ascending=True).reset_index(drop=True)
        add_columns(temp_df)
        

        # Loop through each article in the temporary dataframe
        for idx, row in temp_df.iterrows():
            article_id = row['id']
            # Split the body of the article into sentences
            #[s.strip() for p in row['body'].split('\n') for s in p.split('.') if s.strip()]
            sentences = count_empty_sentences(row['body']) 

            sentence_embs1 = check_similarity_embedings(event, article_id, sentences, similarity_path, similarity_calculator)

            check_propaganda(article_id, idx, temp_df, sentences, propaganda_predictor)

            calculate_similarity_matrix(event, idx, row, temp_df, sentences, sentence_embs1, similarity_path, similarity_calculator)

            export_data(temp_df, event, result_path,name)
