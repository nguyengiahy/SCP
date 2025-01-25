import torch
from src.utils.criterion import SCL

#Init two tensor, one with all zeros and one with all ones
#Two test tensor is 2D matrix => Able to manually check the cosine of the vector
scl = SCL()
x = torch.ones(5, 768)
y = torch.ones(5, 768)
z = torch.stack([x, y]).unsqueeze(0)
#Case 1: Same vector (Cos = 1)
print(torch.nn.functional.cosine_similarity(x, y))
print(scl(z))
'''
tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
tensor([5.9605e-08])
=> Incorrect (SCL should equal to 0)
=> Problem: Floating point issue (https://docs.python.org/3/tutorial/floatingpoint.html)
=> Solution 1: Assign all initialized number to double precision (float64) => Notably increase memory used
=> Solution 2: Add round function for the the similarity to be rounded at 6 decimal places
'''

#Case 2: Opposite vector (Cos = -1)
y = torch.mul(y, -1)
z = torch.stack([x, y]).unsqueeze(0)
#print(torch.nn.functional.cosine_similarity(x, y))
#print(scl(z))
'''
tensor([-1.0000, -1.0000, -1.0000, -1.0000, -1.0000])
tensor([2.])
=> Correct
'''

#Case 3: Random case
y = torch.zeros(5, 768)
y[0, 0] = 1
y[1, 1] = 1
y[2, 2] = 1
y[3, 3] = 1
y[4, 4] = 1
z = torch.stack([x, y]).unsqueeze(0)
#print(torch.nn.functional.cosine_similarity(x, y))
#print(scl(z))
'''
tensor([0.0361, 0.0361, 0.0361, 0.0361, 0.0361])
tensor([0.9639])
=> Correct
'''

