import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

names = open('names.txt', 'r').read().splitlines()

chars = list('abcdefghijklmnopqrstuvwxyz')
stoi = {s:i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {}
for k, v in stoi.items():
    itos[v] = k

block_size = 3 # content size (how many characters do we take into account to make a prediction?)
X, Y = [], []
for name in names:
    context = [0] * block_size
    for ch in name + '.':
        idx = stoi[ch]
        X.append(context)
        Y.append(idx)
        # print(''.join(itos[i] for i in context), '--->', itos[idx])
        context = context[1:] + [idx] # crop and append

X = torch.tensor(X)
Y = torch.tensor(Y)
print(X.shape, Y.shape)

'''

C = torch.randn((27,2))
emb = C[X]

W1 = torch.randn((6, 100))
b1 = torch.randn(100)

# emb @ W1 + b1 # won't work because emb is shape (x, 3, 2) and W1 is shape (6, 100). Want to make emb shape (x, 6)
# print(torch.cat(torch.unbind(emb, 1), 1).shape) # unbind at dim 1 of emb and concat across dim 1 of resultant vector of unbind
h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # view manipulates emb the same as the above (-1 lets python imply what the dimension should be)
print(h.shape)

W2 = torch.randn((100, 27)) # Use 27 since there are 27 characters including '.'
b2 = torch.randn(27)

logits = h @ W2 + b2
print(logits.shape)

counts = logits.exp()
prob = counts / counts.sum(1, keepdims=True)
print(prob.shape)

# Probabilities for each block_size context for each output character in Y. Log, mean and negative for quantifying loss
loss = -prob[torch.arange(prob.shape[0]), Y].log().mean()
print(loss)

'''

### Cleaner Version ###

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

for _ in range(10000):

    # minibatch construct (pick 36 random indexes among the inputs to test on)
    idx = torch.randint(0, X.shape[0], (36,))

    # forward pass
    emb = C[X[idx]] # (36, 3, 2)
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (36, 100)
    logits = h @ W2 + b2 # (36, 100) @ (100, 27) --> (36, 27) + (, 27) --> (36, 27)
    # counts = logits.exp()
    # prob = counts / counts.sum(1, keepdims=True)
    # loss = -prob[torch.arange(prob.shape[0]), Y].log().mean()
    loss = F.cross_entropy(logits, Y[idx]) # Calculates the exact number commented out lines directly above but more efficient

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    # update parameters
    for p in parameters:
        p.data += -0.1 * p.grad

# print(loss.item())


emb = C[X]
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y)
print(loss.item())