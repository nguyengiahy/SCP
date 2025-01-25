import torch
import torch.nn.functional as F
import torch.nn as nn

# ---- Mean Squared Error Loss ----
class MSE(nn.Module):
    def __init__(self, norm_dim = None):
        """
        Args:
            norm_dim: the dimension that we want to normalize the ground truth and the prediction
        """
        super().__init__()
        self.norm_dim = norm_dim
    
    def forward(self, pred, gt):
        """
        Args:
            pred: prediction made by model, tensor with shape [batch, d_model]
            gt: ground truth value, tensor with shape [batch, d_model]
        """
        if self.norm_dim is not None:
            pred = F.normalize(pred, p=2, dim=self.norm_dim)
            gt = F.normalize(gt, p=2, dim=self.norm_dim)

        squared_error = torch.square(pred - gt)     # squared_error = [batch, d_model]
        mse = torch.mean(squared_error)             # mse = [] (scalar tensor)
        return mse

# ---- Semantic Concentration Loss ----    
class SCL(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, heads):
        '''
        Compute the  dissimilarity between each head in each batch and then calculate the average across all batches.
        Args:
            heads = (batch, num_heads, seq_len, d_model)
        Return: 
            Average cosine similarity between batches
        '''
        num_batches, num_heads, seq_len, _ = heads.shape
        if num_heads <= 1:
            raise ValueError("There must be at least 2 heads to compute SCL.")
        
        loss = 0
        for batch in range(num_batches):
            batch_scl = count = 0
            head_anchor = heads[batch, 0]     # head_anchor = (seq_len, d_model)
            for head1 in range(1, num_heads - 1):
                # Flatten the head tensors
                head_tensor1 = heads[batch, head1]        # head_tensor1 = (seq_len, d_model)
                scl = 0
                for seq in range(0, seq_len):
                    # Compute the cosine similarity between the current pair of heads
                    similarity = F.cosine_similarity(head_tensor1[seq].unsqueeze(0), head_anchor[seq].unsqueeze(0))
                    # Compute SCL loss for the current pair of heads
                    temp_scl = torch.sub(1, similarity)
                    scl += temp_scl
                scl /= seq_len
                batch_scl += scl
                count += 1
            batch_scl /= count
            loss += batch_scl

        loss /= num_batches
        return loss