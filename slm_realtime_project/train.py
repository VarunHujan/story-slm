import torch
import torch.optim as optim
from model import RealtimeSLM

def train_dummy():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training Realtime SLM on device: {device}')
    
    model = RealtimeSLM(vocab_size=32000, dim=256, n_layers=4, n_heads=4, n_kv_heads=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    dummy_input = torch.randint(0, 32000, (2, 64), device=device)
    target = torch.randint(0, 32000, (2, 64), device=device)

    model.train()
    for step in range(1, 11):
        optimizer.zero_grad()
        logits, _ = model(dummy_input)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 32000), target.view(-1))
        loss.backward()
        optimizer.step()
        print(f'Step [{step}/10] - Loss: {loss.item():.4f}')

    print('Training dummy loop completed successfully!')

if __name__ == '__main__':
    train_dummy()
