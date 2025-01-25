import torch.nn as nn
import torch

from transformers import ViTModel
from transformers import ViTImageProcessor
import math
from ..utils.misc import get_device_available
from ..utils.criterion import SCL

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, temperature = 10000):
        '''
        d_model: feature dimension (default = 768)
        seq_len: sequence length
        '''
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.temperature = temperature


    def forward(self):
        pos = torch.arange(self.seq_len, dtype=torch.float32).unsqueeze(1)              # pos = [[0], [1], ..., [seq_len-1]]
        i = torch.arange(self.d_model // 2, dtype=torch.float32).unsqueeze(0)           # i = [[0, 1, ..., d_model/2 - 1]]

        # Compute the positional encodings
        angle_rates = 1 / (self.temperature ** (2 * i / self.d_model))
        pos_encoding = torch.zeros(self.seq_len, self.d_model, dtype=torch.float32)
        pos_encoding[:, 0::2] = torch.sin(pos * angle_rates)
        pos_encoding[:, 1::2] = torch.cos(pos * angle_rates)

        # Add a dimension for batch size
        pos_encoding = pos_encoding.unsqueeze(0)

        # Disable gradient because PE are not learnable parameters
        pos_encoding.requires_grad_(False)

        return pos_encoding.to(get_device_available())     # pos_encoding = [1, seq_len, 768]

class InputEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        #Init embedding model when create instance of dataset -> Avoid model retrieval loops
        model_name = 'google/vit-base-patch16-224'
        self.emb_model = ViTModel.from_pretrained(model_name)
        self.emb_processor = ViTImageProcessor.from_pretrained(model_name)
        self.device = get_device_available()
        self.emb_model.to(self.device)


    def forward(self, frames):
        '''
        frames: sequence of PIL Image (batch, seq_len, channel, width, height)
        input_embed: (batch, seq_len, d_model)
        '''
        #input_embeds holds embedded frames of the batch (input_embeds = [batch, seq_len, d_model])
        input_embeds = []
        for i in range(frames.size(dim=0)):
          #input_embed holds embedded frames of one sample
          input_embed = []
          single_sample = frames[i]
          for frame in single_sample:
              #Embed frames
              inputs = self.emb_processor(images=frame, return_tensors='pt')
              pixel_values = inputs.pixel_values.to(self.device)                  # pixel_values = [1, 3, 224, 224]
              with torch.no_grad():
                  output = self.emb_model(pixel_values)
                  # Get the representation of the entire frame
                  output = output.last_hidden_state.mean(dim=1)       # shape = [d_model]
                  input_embed.append(output)
          input_embed = torch.cat(input_embed, dim=0)                 # input_embed = [seq_len, d_model]
          input_embeds.append(input_embed)
        #Concatenate individual sample embeddings to form input_embeds
        input_embeds = torch.stack(input_embeds)
        d_model = input_embeds.shape[-1]
        # Scale the embeddings
        input_embeds = input_embeds * math.sqrt(d_model)
        
        return input_embeds        # input_embed = [batch, seq_len, d_model]

class LayerNormalization(nn.Module):
    def __init__(self, d_model, epsilon=10**-6):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(d_model))      # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(d_model))      # bias is a learnable parameter

    def forward(self, x):
        '''
        Args:
            x: (batch, seq_len, d_model)
            return: normalized x (batch, seq_len, d_model)
        '''
        mean = x.mean(dim=-1, keepdim=True)         # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)           # (batch, seq_len, 1)
        return self.alpha * (x-mean) / (std + self.epsilon) + self.bias     # (batch, seq_len, d_model)

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, dtype=torch.float32)
        self.linear_2 = nn.Linear(d_ff, d_model, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        x: (batch, seq_len, d_model)
        '''
        output = self.linear_1(x)       # output: (batch, seq_len, d_ff)
        output = torch.relu(output)
        output = self.dropout(output)
        output = self.linear_2(output)  # output: (batch, seq_len, d_model)

        return output

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        w_q = [nn.Linear(d_model, d_model, dtype=torch.float32) for _ in range(num_heads)]      # w_q = (num_heads, d_model, d_model)
        w_k = [nn.Linear(d_model, d_model, dtype=torch.float32) for _ in range(num_heads)]      # w_k = (num_heads, d_model, d_model)
        w_v = [nn.Linear(d_model, d_model, dtype=torch.float32) for _ in range(num_heads)]      # w_v = (num_heads, d_model, d_model)

        self.w_q = nn.ModuleList(w_q)
        self.w_k = nn.ModuleList(w_k)
        self.w_v = nn.ModuleList(w_v)
        self.w_o = nn.Linear(num_heads * d_model, d_model, bias=False, dtype=torch.float32)      # w_o = (num_heads * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        x: (batch, seq_len, d_model)
        '''
        q = [layer(x) for layer in self.w_q]        # q[i] = (batch, seq_len, d_model)
        k = [layer(x) for layer in self.w_k]        # k[i] = (batch, seq_len, d_model)
        v = [layer(x) for layer in self.w_v]        # v[i] = (batch, seq_len, d_model)
        q, k, v = torch.stack(q), torch.stack(k), torch.stack(v)        # q, k, v = (num_heads, batch, seq_len, d_model)
        q, k, v = q.permute(1, 0, 2, 3), k.permute(1, 0, 2, 3), v.permute(1, 0, 2, 3)                              # q, k, v = (batch, num_heads, seq_len, d_model)

        k_transpose = k.transpose(-2, -1)           # k_transpose = (batch, num_heads, d_model, seq_len)
        attention_scores = q @ k_transpose          # attention_score = (batch, num_heads, seq_len, seq_len)

        # Normalise the attention scores
        attention_scores = attention_scores / math.sqrt(self.d_model)      # attention_scores = (batch, num_heads, seq_len, seq_len)
        # Apply softmax to attention scores
        attention_scores = attention_scores.softmax(dim=-1)     # attention_scores = (batch, num_heads, seq_len, seq_len)

        # Dropout
        if self.dropout is not None:
            attention_scores = self.dropout(attention_scores)   # attention_scores = (batch, num_heads, seq_len, seq_len)

        # Calculate all heads
        heads = attention_scores @ v                            # heads = (batch, num_heads, seq_len, d_model)
        #heads_values is a copy of heads (Use clone to avoid affecting heads when calculating SCL loss)
        scl_module = SCL()
        scl_value = scl_module(heads)
        # Concatenate heads along the seq_len dimension
        heads = heads.transpose(1, 2)                                           # heads = (batch, seq_len, num_heads, d_model)
        heads = heads.contiguous().view(heads.shape[0], heads.shape[1], -1)     # heads = (batch, seq_len, num_heads * d_model)

        # Linear transform with output weights
        output = self.w_o(heads)                                # output = (batch, seq_len, d_model)

        # Compute SCL loss

        return output, scl_value

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, dropout):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_model]
        _, _, _, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose [batch_size, head, d_model, length]
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product [batch_size, head, length, length]

        # 2. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 3. dropout
        if self.dropout is not None:
            score = self.dropout(score)   # attention_scores = (batch, num_heads, seq_len, seq_len)

        # 3. multiply with Value
        heads = score @ v

        return heads

class OriginalMultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout):
        super(OriginalMultiHeadAttention, self).__init__()
        self.n_head = num_heads
        self.attention = ScaleDotProductAttention(dropout=dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        self.scl = SCL()

    def forward(self, x, mask=None):
        # x: Input tensor [batch, seq_len, d_model]
        # 1. dot product with weight matrices
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        heads = self.attention(q, k, v, mask=mask)

        # 4. calculate SCL loss
        loss = self.scl(heads)

        # 5. concat and pass to linear layer
        heads = self.concat(heads)
        out = self.w_concat(heads)

        return out, loss

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class AddNormBlock(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer, *args, **kwargs):
        """
        Apply residual connection to any sublayer with the same size.
        x: Input tensor
        sublayer: A function representing the sublayer (e.g., multi-head attention, feed-forward)
        args: Additional positional arguments for the sublayer
        kwargs: Additional keyword arguments for the sublayer
        """
        output = sublayer(x)
        if isinstance(output, tuple):
            #If output not a single tensor -> MHSA (SCL loss is calculated along MHSA output)
            output, _ = output
        return self.norm(x + self.dropout(output))

class EncoderBlock(nn.Module):
    def __init__(self, mhsa_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 d_model: int,
                 dropout: float):
        super().__init__()
        self.mhsa_block = mhsa_block
        self.feed_forward_block = feed_forward_block
        self.add_norm_block = nn.ModuleList([AddNormBlock(d_model, dropout) for _ in range(2)])

    def forward(self, x):
        '''
        Args:
            x: input [batch, seq_len, d_model]
        '''
        #Get SCL loss for input x
        _, loss = self.mhsa_block(x)
        x = self.add_norm_block[0](x, lambda x: self.mhsa_block(x))
        x = self.add_norm_block[1](x, self.feed_forward_block)
        return x, loss

class OriginalEncoderBlock(nn.Module):
    def __init__(self, mha_block: OriginalMultiHeadAttention,
                feed_forward_block: FeedForwardBlock,
                d_model: int,
                dropout: float):
        super().__init__()
        self.mha_block = mha_block
        self.feed_forward_block = feed_forward_block
        self.add_norm_block = nn.ModuleList([AddNormBlock(d_model, dropout) for _ in range(2)])

    def forward(self, x):
        '''
        Args:
            x: input [batch, seq_len, d_model]
        '''
        #Get SCL loss for input x
        _, loss = self.mha_block(x)
        x = self.add_norm_block[0](x, lambda x: self.mha_block(x))
        x = self.add_norm_block[1](x, self.feed_forward_block)
        return x, loss
    
class Encoder(nn.Module):
    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x):
        loss = None               #Sum up all losses as final loss => Update whole model
        for layer in self.layers:
          if loss == None:
            #Init loss
            x, loss = layer(x)
          else:
            #Add up tmp_loss to loss
            x, tmp_loss = layer(x)
            loss += tmp_loss

        return self.norm(x), loss

class OriginalEncoder(nn.Module):
    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x):
        loss = None               #Sum up all losses as final loss => Update whole model
        for layer in self.layers:
          if loss == None:
            #Init loss
            x, loss = layer(x)
          else:
            #Add up tmp_loss to loss
            x, tmp_loss = layer(x)
            loss += tmp_loss

        return self.norm(x), loss
class PredictionLayer(nn.Module):
    def __init__(self, d_model):
        '''
        Args:
            d_model: feature dimension of an input embedding
        '''
        super().__init__()
        self.fc = nn.Linear(d_model, d_model, dtype=torch.float32)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        Take the Transformer Encoder's output of the last frame in the sequence to predict the embedding of the next frame
        Args:
            x: (batch, seq_len, d_model)
            return: (batch, d_model)
        '''
        # Get the last row, which is the Attention encoded representation of the last frame
        x = x[:, -1, :]                 # x = (batch, d_model)
        x = self.relu(self.fc(x))       # x = (batch, d_model)

        return x

class SCPModel(nn.Module):
    def __init__(self, encoder: Encoder, pred_layer: PredictionLayer, src_embed: InputEmbeddings, src_pos: PositionalEncoding):
        super().__init__()
        self.encoder = encoder
        self.pred_layer = pred_layer
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.device = get_device_available()

    def forward(self, src):
        '''
        Args:
            src: 'n' frames
        '''

        # Get input embedding
        src = self.src_embed(src)        # src = (batch, seq_len, d_model)

        # Get positional encoding
        pos_encoding = self.src_pos()       # pos_encoding = (1, seq_len, d_model)

        # Add input embedding + positional encoding to generate the complete input
        input = src + pos_encoding          # input = (batch, seq_len, d_model)

        # Get output and loss from the encoder module
        output, loss = self.encoder(input)        # output = (batch, seq_len, d_model), SCL_value

        # Get output from the prediction layer module
        output = self.pred_layer(output)    # output = (batch, embed_num_features * d_model)

        return output, loss
    
class TransformerEncoderModel(nn.Module):
    def __init__(self, encoder: OriginalEncoder, pred_layer: PredictionLayer, src_embed: InputEmbeddings, src_pos: PositionalEncoding):
        super().__init__()
        self.encoder = encoder
        self.pred_layer = pred_layer
        self.src_embed = src_embed
        self.src_pos = src_pos

    def forward(self, src):
        '''
        Args:
            src: 'n' frames
        '''

        # Get input embedding
        src = self.src_embed(src)        # src = (batch, seq_len, d_model)

        # Get positional encoding
        pos_encoding = self.src_pos()       # pos_encoding = (1, seq_len, d_model)

        # Add input embedding + positional encoding to generate the complete input
        input = src + pos_encoding          # input = (batch, seq_len, d_model)

        # Get output and loss from the encoder module
        output, loss = self.encoder(input)        # output = (batch, seq_len, d_model), SCL_value

        # Get output from the prediction layer module
        output = self.pred_layer(output)    # output = (batch, embed_num_features * d_model)

        return output, loss

def build_model(d_model, seq_len, N = 12, h = 16, dropout = 0.1, d_ff = 2048, device='cpu'):
    '''
    d_model: feature dimension of an input embedding
    seq_len: length of the input sequence
    N: number of encoder blocks in the model
    h: number of heads for multi-head self-attention
    d_ff: the dimension of the hidden layer of Feed Forward Block
    '''
    # Input embedding layer
    src_embed = InputEmbeddings()

    # Positional encoding layer
    pos_enc = PositionalEncoding(d_model, seq_len)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        mhsa_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(mhsa_block, feed_forward_block, d_model, dropout)
        encoder_blocks.append(encoder_block)

    # Create the encoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    # Create the prediction layer
    pred_layer = PredictionLayer(d_model)

    # Create the Semantic Concentration Encoder
    model = SCPModel(encoder, pred_layer, src_embed, pos_enc)

    # Initialise the parameters of the model
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

def build_original_model(d_model, seq_len, N = 12, h = 16, dropout = 0.1, d_ff = 2048, device='cpu'):
    '''
    d_model: feature dimension of an input embedding
    seq_len: length of the input sequence
    N: number of encoder blocks in the model
    h: number of heads for multi-head self-attention
    d_ff: the dimension of the hidden layer of Feed Forward Block
    '''
    # Input embedding layer
    src_embed = InputEmbeddings()

    # Positional encoding layer
    pos_enc = PositionalEncoding(d_model, seq_len)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        mhsa_block = OriginalMultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(mhsa_block, feed_forward_block, d_model, dropout)
        encoder_blocks.append(encoder_block)

    # Create the encoder
    encoder = OriginalEncoder(d_model, nn.ModuleList(encoder_blocks))

    # Create the prediction layer
    pred_layer = PredictionLayer(d_model)

    # Create the Transformer Encoder Model
    model = TransformerEncoderModel(encoder, pred_layer, src_embed, pos_enc)

    # Initialise the parameters of the model
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model