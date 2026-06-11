import os
import torch
import copy

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one epoch of training and return (avg_loss, accuracy)."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)

        if model.num_classes == 2:
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True) 
        else:
            labels = labels.long().to(device, non_blocking=True)              

        optimizer.zero_grad()
        outputs = model(images) # raw logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        if model.num_classes == 2:
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
        else:
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Evaluate on a dataloader and return (avg_loss, accuracy)."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if model.num_classes == 2:
                labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
            else:
                labels = labels.long().to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            if model.num_classes == 2:
                preds = (torch.sigmoid(outputs) > 0.5).long()
                correct += (preds == labels.long()).sum().item()
            else:
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
            total += images.size(0)

    return running_loss / total, correct / total


def train_model(
    model, train_loader, val_loader,
    optimizer, scheduler, criterion,
    device,
    epochs, ckpt_path,
    early_stop_patience=10,
    start_epoch=0
):
    """
    """
    # Ensure checkpoint directory exists
    ckpt_dir = os.path.dirname(ckpt_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
                
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_weights  = None
    patience_ctr  = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)

        scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        print(f'Epoch {epoch+1:>4}/{start_epoch+epochs} '
              f'| train_loss {tr_loss:.4f}  train_acc {tr_acc:.4f} '
              f'| val_loss {vl_loss:.4f}  val_acc {vl_acc:.4f}')

        # ── Checkpoint on best val_loss ──────────────────────────────────────
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_weights  = copy.deepcopy(model.state_dict())
            
            torch.save(best_weights, ckpt_path)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience:
                print(f'  Early stopping at epoch {epoch+1}.')
                break

    # Restore best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f'  Best weights restored from {ckpt_path}')
    return history

