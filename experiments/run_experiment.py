import argparse
import sys
import os
from src.Data import DRDataset
from torchvision import transforms
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.Data import dataset_download, dataset_prepare, data_load, data_split, create_dataloader
from src.Model import model_setup
from src.Train import train_model
from src.Test import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="DR Classification Experiment")
    parser.add_argument('--dataset',     required=True, choices=["Aptos", "IDRiD", "DDR", "Messidor-2"])
    parser.add_argument('--model',       required=True)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs',      type=int, default=30)
    parser.add_argument('--batch_size',  type=int, default=16)
    parser.add_argument('--img_size',    type=int, default=224)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--data_dir',    default='./Data/')
    parser.add_argument('--ckpt_dir',    default='./checkpoints/')
    args = parser.parse_args()

    # 1. Data
    download_path  = dataset_download(args.dataset, args.data_dir)
    dataset_path   = dataset_prepare(args.dataset, download_path, args.data_dir)
    
    labels_type = "Grading" if args.num_classes == 5 else "Binary"
    dataset = data_load(dataset_path, args.img_size, labels_type)

    splits = data_split(dataset, test_split_ratio=0.2, val_split=True, val_split_ratio=0.1)
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    images_path = os.path.join(dataset_path, "Images")
    transform   = dataset.transform
    train_ds = DRDataset(images_path, X_train, y_train, transform)
    val_ds   = DRDataset(images_path, X_val,   y_val,   transform)
    test_ds  = DRDataset(images_path, X_test,  y_test,  transform)

    train_loader = create_dataloader(train_ds, args.batch_size, shuffle=True)
    val_loader   = create_dataloader(val_ds,   args.batch_size, shuffle=False)
    test_loader  = create_dataloader(test_ds,  args.batch_size, shuffle=False)

    # 2. Model
    conf = {'LR': args.lr, 'LR_SCHED_FACTOR': 0.5, 'LR_SCHED_PAT': 5, 'LR_MIN': 1e-6}
    setup = model_setup(args.model, args.num_classes, conf)

    ckpt_path = os.path.join(args.ckpt_dir, f"{args.model}_{args.dataset}.pt")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # 3. Train
    train_model(
        model=setup['model'], train_loader=train_loader, val_loader=val_loader,
        optimizer=setup['optimizer'], scheduler=setup['scheduler'],
        criterion=setup['criterion'], device=setup['device'],
        epochs=args.epochs, ckpt_path=ckpt_path
    )

    # 4. Evaluate
    task_type = 'binary' if args.num_classes == 2 else 'multiclass'
    evaluate_model(setup['model'], test_loader, setup['device'], task_type, args.model)


if __name__ == '__main__':
    main()