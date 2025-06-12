import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, dataloader, optimizer, num_epochs=5, freeze_encoder=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Freeze or unfreeze encoder parameters based on flag
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False  # Freeze encoder parameters (no gradient updates):contentReference[oaicite:0]{index=0}
    else:
        for param in model.encoder.parameters():
            param.requires_grad = True   # Fine-tune encoder parameters

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss (includes LogSoftmax internally):contentReference[oaicite:1]{index=1}
    # Only optimize parameters that require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for (x_categ, x_cont, labels) in dataloader:
            x_categ, x_cont, labels = x_categ.to(device), x_cont.to(device), labels.to(device)

            # Forward pass
            outputs = model(x_categ, x_cont)
            # Convert logits to probabilities for interpretation
            probs = torch.softmax(outputs, dim=1)  # get class probabilities:contentReference[oaicite:2]{index=2}
            # Compute loss (CrossEntropyLoss expects raw logits but computes LogSoftmax internally):contentReference[oaicite:3]{index=3}
            loss = criterion(outputs, labels)

            # Backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()  # Backpropagate loss:contentReference[oaicite:4]{index=4}
            optimizer.step() # Update model parameters:contentReference[oaicite:5]{index=5}

            running_loss += loss.item()

            # Compute accuracy: compare predicted class to true labels
            preds = torch.argmax(probs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {epoch_loss:.4f}  Accuracy: {epoch_acc:.2f}%")
    print("Training complete.")