import torch
import math
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self,
                 embed_d: int,
                 dropout: float,
                 max_len: int
                 ):
        """
        Args:
          d_model: Dimension of the embedding.
          dropout: Dropout rate.
          max_len: Maximum length of the sequence.

        """

        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pos_encoding = torch.zeros(max_len, embed_d)
        positions_list = torch.arange(0,
                                      max_len,
                                      dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5

        division_term = torch.exp(torch.arange(0, embed_d, 2).float() * (-math.log(10000.0)) / embed_d) # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])





class FN_Generator(nn.Module):
    """ Generate short_description """
    def __init__(self,
                 output_dim: int,
                 headline_vocabsize: int,
                 inputsize_H: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_decoder_layers: int = 6,
                 dropout: float = 0.1
                 ) -> None:

        """
        Args:
          output_dim          : Dimension of the output.
          embed_dim           : Dimension of the embedding.
          headline_vocabsize  : Vocabulary size of the headline.
          inputsize_H         : Input size of the headline.
          inputsize_S         : Input size of the short_description.
          cat_vocabsize       : Vocabulary size of the category.
          auth_vocabsize      : Vocabulary size of the authors.
          d_model             : Dimension of the Transformer.
          nhead               : Number of heads.
          num_encoder_layers  : Number of encoder layers.
          num_decoder_layers  : Number of decoder layers.
          dropout             : Dropout rate.

        """
        super(FN_Generator, self).__init__()

        self.positional_encoder = PositionalEncoding(embed_d=d_model,
                                                     dropout=dropout,
                                                     max_len=inputsize_H)
        self.data_embed = nn.Linear(3, d_model)
        self.relu       = nn.ReLU()
        self.flatten    = nn.Flatten()

        self.headline_embed = nn.Embedding(headline_vocabsize, d_model)

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model,
                                       nhead,
                                       batch_first=True),
            num_decoder_layers)

        self.output = nn.Linear(d_model * (inputsize_H+1), output_dim)

    def forward(self, src_h, src_o, tgt):
        lin = self.relu(self.data_embed(src_o)).unsqueeze(1)

        headline = self.positional_encoder(self.headline_embed(src_h))
        short_description = self.positional_encoder(self.headline_embed(tgt))

        trans_in = torch.cat((headline, lin), dim=1)
        trans_out = self.transformer(trans_in, short_description)

        return nn.functional.softmax(self.output(self.flatten(trans_out)), dim=1)