import torch

def evaluate(model, test_loader):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    criterion = torch.nn.MSELoss()  # Assuming regression, adjust for your use case

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    average_loss = total_loss / len(test_loader)
    return {"average_loss": average_loss}