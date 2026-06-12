import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def get_predictions(model, loader, device, task_type):
    """Return (y_true, y_pred) numpy arrays from the test set."""
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)

            if task_type == 'binary':
                preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
            else:
                preds = outputs.argmax(dim=1)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

    return np.array(all_true), np.array(all_pred)

def evaluate_model(model, loader, device, task_type, model_name, class_names=None):
    """Print classification report and return (y_true, y_pred, confusion_matrix)."""
    print(f'\n{"="*60}')
    print(f'EVALUATING  {model_name}  –  {task_type.upper()}')
    print(f'{"="*60}')

    y_true, y_pred = get_predictions(model, loader, device, task_type)

    target_names = [str(c) for c in class_names] if class_names else None
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    return y_true, y_pred, cm

