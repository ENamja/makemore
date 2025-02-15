import torch
import torch.nn.functional as F
import random

names = open('names.txt', 'r').read().splitlines()

chars = list('abcdefghijklmnopqrstuvwxyz')
stoi = {s:i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {}
for k, v in stoi.items():
    itos[v] = k

'''

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

def build_dataset(names):
    block_size = 3
    X, Y = [], []
    for name in names:

        context = [0] * block_size
        for ch in name + '.':
            idx = stoi[ch]
            X.append(context)
            Y.append(idx)
            context = context[1:] + [idx]
        
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y


random.seed(42)
random.shuffle(names)
n1 = int(0.8 * len(names))
n2 = int(0.9 * len(names))

# training, validation, and testing sets
Xtr, Ytr = build_dataset(names[:n1])
Xdev, Ydev = build_dataset(names[n1:n2])
Xte, Yte = build_dataset(names[n2:])

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 10), generator=g)
W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

lr = 0.1
for i in range(60000):

    if i == 20000: lr = 0.05
    if i == 40000: lr = 0.01
    # minibatch construct (pick 36 random indexes among the inputs to test on)
    idx = torch.randint(0, Xtr.shape[0], (36,))

    # forward pass
    emb = C[Xtr[idx]] # (36, 3, 2)
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (36, 100)
    logits = h @ W2 + b2 # (36, 100) @ (100, 27) --> (36, 27) + (, 27) --> (36, 27)
    # counts = logits.exp()
    # prob = counts / counts.sum(1, keepdims=True)
    # loss = -prob[torch.arange(prob.shape[0]), Y].log().mean()
    loss = F.cross_entropy(logits, Ytr[idx]) # Calculates the exact number commented out lines directly above but more efficient

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    # update parameters
    for p in parameters:
        p.data += -lr * p.grad


emb = C[Xdev]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print(loss.item())

emb = C[Xtr]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)
print(loss.item())

'''
## Testing Set Loss
emb = C[Xte]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Yte)
print(loss.item())
'''

# Sampling:
block_size = 3
for _ in range(20):
    
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        idx = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [idx]
        out.append(idx)
        if idx == 0:
            break

    print(''.join(itos[i] for i in out))