import torch
import torch.nn as nn

class CentralityEncoding(nn.Module):
    def __init__(self,
                 hidden_size,
                 max_in_degree,
                 max_out_degree,
                 in_degree_list,
                 out_degree_list):
        super(CentralityEncoding, self).__init__()
        
        self.in_degree_encoder = nn.Embedding(max_in_degree, hidden_size, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(max_out_degree, hidden_size, padding_idx=0)
        self.in_degree_list = in_degree_list.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.out_degree_list = out_degree_list.to('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self):
        return self.in_degree_encoder(self.in_degree_list) + self.out_degree_encoder(self.out_degree_list)
