from torch.utils.data import DataLoader
import pickle

def train_model(model,data, lr):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criteria = nn.CrossEntropyLoss(ignore_index=0)
    print('Starting Traning...')
    for epoch in range(20):

        model.train()
        for batch_idx,(src, trgt) in enumerate(data):
            src_input = src.contiguous().to(device)
            trgt_input = trgt[:,:-1].contiguous().to(device)
            trgt_expected = trgt[:,1:].contiguous().to(device)
            optimizer.zero_grad()
            output = model(src_input, trgt_input) 
            loss = criteria(output.contiguous().view(-1, trgt_vocabs_size), trgt_expected.view(-1))
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_load)} | Loss: {loss.item():.4f}")

