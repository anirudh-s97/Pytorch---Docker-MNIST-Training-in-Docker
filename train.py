import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(rank, args, model, device, dataset, dataloader_kwargs):
    torch.manual_seed(args.seed + rank)

    train_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer)

def test(args, model, device, dataset, dataloader_kwargs):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


def train_epoch(epoch, args, model, device, data_loader, optimizer):
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = F.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('{}\\tTrain Epoch: {} [{}/{} ({:.0f}%)]\\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
            if args.dry_run:
                break
    
    PATH = "model_checkpoint.pth"
    torch.save(model.state_dict(), PATH)

def test_epoch(model, device, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

def save_model(model, optimizer, epoch, checkpoint_dir, accuracy):
    try:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "accuracy": accuracy,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        save_path = os.path.join(checkpoint_dir, "model_checkpoint.pth")
        torch.save(checkpoint, save_path)
        print(f"Model saved at {save_path} with accuracy: {accuracy:.2f}%")
    except Exception as e:
        print(f"Error saving model: {e}")


def main():
    # Parser to get command line arguments
    parser = argparse.ArgumentParser(description='MNIST Training Script')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                        help='how many training processes to use (default: 2)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume ckpt')
        
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    args = parser.parse_args()

    device = torch.device("cuda" if use_cuda else "cpu")
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    kwargs = {'batch_size': args.batch_size,
              'shuffle': True}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                      })

    torch.manual_seed(args.seed)
    mp.set_start_method('spawn')

    model = Net().to(device)
    model.share_memory() # gradients are allocated lazily, so they are not shared here

    checkpoint_dir = "."
    checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint.pth")

    if args.resume and os.path.isfile(checkpoint_path):
        try:
            print(f"Loading checkpoint from '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint (epoch {checkpoint['epoch']})")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch.")
    else:
        print(
            f"No checkpoint found or resume not requested. Starting from scratch."
        )

    processes = []
    for rank in range(args.num_processes):
        try:
            p = mp.Process(
                target=train, args=(rank, args, model, device, dataset1, kwargs)
            )
            p.start()
            processes.append(p)
        except Exception as e:
            print(f"Error starting process {rank}: {e}")

    for p in processes:
        p.join()

    # Once training is complete, we can test the model
    final_accuracy = test(args, model, device, dataset2, kwargs)

    # Save the final model
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum
    )
    
    save_model(model, optimizer, args.epochs, checkpoint_dir, final_accuracy)

        
if __name__ == "__main__":
    main()
