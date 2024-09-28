import pandas as pd
import torch
from torch.utils.data import Dataset
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('m.model')

def encode(
    df: pd.DataFrame,
    bos_token: int,
    eos_token: int,
    mean: float = None,
    std:float = None
    ):
  """ Tokenize dataframe in the form

      | headline |	category |	short_description |	authors |	date |
            ||          ||              ||             ||       ||
      sentencepiece  index-based   sentencepiece  index-based  Normalized

  Args:
      df: The dataframe to Tokenize.
      mean: Mean for normalization of date.
      std: Standard deviation for normalization of date.

  Returns:
      The tokenized dataframe.
  """

  categories         = df.category.unique().tolist()
  category_tokens_ix = dict(enumerate(categories))
  category_tokens_xi = {x:i for i, x in category_tokens_ix.items()}

  authors           = df.authors.unique().tolist()
  authors_tokens_ix = dict(enumerate(authors))
  authors_tokens_xi = {x:i for i, x in authors_tokens_ix.items()}

  print("Total number of categories:", len(category_tokens_ix))
  print("Total number of Authors:",    len(authors_tokens_ix))

  df['date']    = (df['date'] - df['date'].min()).dt.total_seconds()

  mean = mean if mean else df['date'].mean()
  std  = std if std else df['date'].std()

  print('Using mean as {} for date'.format(mean))
  print('Using std as {} for date'.format(std))

  df['date']              = (df['date'] - mean)/std
  df['category']          = df['category'].map(category_tokens_xi)
  df['authors']           = df['authors'].map(authors_tokens_xi)
  df['headline']          = df['headline'].map(lambda x: [bos_token] + sp.encode_as_ids(x) + [eos_token])
  df['short_description'] = df['short_description'].map(lambda x: [bos_token] + sp.encode_as_ids(x) + [eos_token])

  return df, category_tokens_ix, authors_tokens_ix, mean, std, (len(category_tokens_ix),
                                                                len(authors_tokens_ix))




def pad_dataset(df: pd.DataFrame, PAD_TOKEN: int):
  """Pads the headline and short_description columns of the dataframe with the given pad token.

  Args:
      df: The dataframe to pad.
      pad_token: The token to use for padding.

  Returns:
      The padded dataframe.
  """
  MAX_HEADLINE_LENGHT       = df['headline'].str.len().max()
  MAX_SHORT_DESCRIPTION_LEN = df['short_description'].str.len().max()

  print("Maxlen for headline:", MAX_HEADLINE_LENGHT)
  print("Maxlen for short_description:", MAX_SHORT_DESCRIPTION_LEN)

  df['headline'] = df['headline'].map(lambda x: x + [PAD_TOKEN] * (MAX_HEADLINE_LENGHT - len(x)))
  df['short_description'] = df['short_description'].map(lambda x: x + [PAD_TOKEN] * (MAX_SHORT_DESCRIPTION_LEN - len(x)))

  return df, MAX_HEADLINE_LENGHT, MAX_SHORT_DESCRIPTION_LEN





class FN_Dataset(Dataset):
    """  News articles data """
    def __init__(self,
                 df: pd.DataFrame,
                 bos_token: int,
                 eos_token: int,
                 pad_token: int,
                 mean: float = None,
                 std:float = None
                 ):
      """

      Args:
        df              : Dataframe containing the data.
        bos_token       : Beginning of sentence token.
        eos_token       : End of sentence token.
        pad_token       : Padding token.
        mean(Optional)  : Mean for normalization of date.
        std(Optional)   : Standard deviation for normalization of date.

      """

      super().__init__()

      self.df, self.Category_decoder, self.Author_decoder, self.mean, self.std, self.vocabsizes = encode(df, bos_token, eos_token, mean, std)
      self.df, self.maxlen_H, self.maxlen_S = pad_dataset(self.df, pad_token)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
      sample = self.df.iloc[idx]

      return (torch.tensor(sample['headline']),
              torch.tensor([sample['authors'], sample['category'], sample['date']]).to(torch.float32),
              torch.tensor(sample['short_description']))
