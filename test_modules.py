import torch
from src.models.model import PositionalEncoding, MultiHeadAttentionBlock, FeedForwardBlock, EncoderBlock

#First test: PositionalEncoding
#Init a 5 * 5 * 768 tensor with all zeros (batch, seq_len, 768)
x = torch.zeros(5, 5, 768)
positional_encoding = PositionalEncoding(d_model=768, seq_len=5)
output = positional_encoding()
#print((output + x))
'''
Row 1:
    [[ 0.0000e+00,  1.0000e+00,  0.0000e+00,  ...,  1.0000e+00,
    0.0000e+00,  1.0000e+00],
    [ 8.4147e-01,  5.4030e-01,  8.2843e-01,  ...,  1.0000e+00,
    1.0243e-04,  1.0000e+00],
    [ 9.0930e-01, -4.1615e-01,  9.2799e-01,  ...,  1.0000e+00,
    2.0486e-04,  1.0000e+00],
    [ 1.4112e-01, -9.8999e-01,  2.1109e-01,  ...,  1.0000e+00,
    3.0728e-04,  1.0000e+00],
    [-7.5680e-01, -6.5364e-01, -6.9153e-01,  ...,  1.0000e+00,
    4.0971e-04,  1.0000e+00]]

Row 2: 
    [[ 0.0000e+00,  1.0000e+00,  0.0000e+00,  ...,  1.0000e+00,
    0.0000e+00,  1.0000e+00],
    [ 8.4147e-01,  5.4030e-01,  8.2843e-01,  ...,  1.0000e+00,
    1.0243e-04,  1.0000e+00],
    [ 9.0930e-01, -4.1615e-01,  9.2799e-01,  ...,  1.0000e+00,
    2.0486e-04,  1.0000e+00],
    [ 1.4112e-01, -9.8999e-01,  2.1109e-01,  ...,  1.0000e+00,
    3.0728e-04,  1.0000e+00],
    [-7.5680e-01, -6.5364e-01, -6.9153e-01,  ...,  1.0000e+00,
    4.0971e-04,  1.0000e+00]]  

    Each row is a positional encoding of a single sample
    => Successful broadcasted positional encoding to the input tensor
'''

#Test 2: Test encoder block
dummy_input = torch.ones(2, 5, 768, dtype=torch.float64)
mhsa = MultiHeadAttentionBlock(d_model=768, num_heads=12, dropout=0.1)
ff = FeedForwardBlock(d_model=768, d_ff=2048, dropout=0.1)
encoder = EncoderBlock(mhsa_block=mhsa, feed_forward_block=ff, d_model=768, dropout=0.1)
print(encoder(dummy_input))
print(mhsa(dummy_input))
'''
Loss from encoder: tensor([1.0002], grad_fn=<DivBackward0>)
Loss from mhsa:  tensor([1.0003], grad_fn=<DivBackward0>)
Problem: Floating point issue
Solution: Change all modules to float64
'''
