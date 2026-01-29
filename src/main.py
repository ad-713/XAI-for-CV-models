import torch
import torch.nn as nn
import torch.optim as optim
from model import vgg16_cifar
from dataset import get_dataloaders
from train import train_one_epoch, validate, save_checkpoint
import argparse

def main():
    parser = argparse.ArgumentParser(description='VGG-16 CIFAR-10 Training')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, test_loader, classes = get_dataloaders(batch_size=args.batch_size)

    # Model
    model = vgg16_cifar(num_classes=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch: {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        scheduler.step()
        
        if val_acc > best_acc:
            print(f"Validation accuracy improved from {best_acc:.2f}% to {val_acc:.2f}%")
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, "outputs/models/vgg16_best.pth")
            
    print(f"Training finished. Best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
