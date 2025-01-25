#Test model architecture

from src.models.model import build_model

d_model = 768               # feature dimension of an input embedding
N = 6                       # Number of encoder blocks in the model
h = 8                       # Number of heads
seq_len = 5                 # Number of input frames

model = build_model(seq_len=seq_len, d_model=d_model, N=N, h=h)

print(model.modules)
#Verified

