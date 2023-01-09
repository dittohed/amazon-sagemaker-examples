import argparse
import json
import logging
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision

from torchvision import transforms


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.model = torchvision.models.resnet50()
        self.model.fc = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)


def _get_train_data_loader(batch_size, root, is_distributed, **kwargs):
    logger.info("Get train data loader")

    transform = [
        transforms.ToTensor(),  # Handles conversion to float32 + dividing by 255
        transforms.RandomResizedCrop(size=32, scale=(0.5, 1),
                                    interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.2, hue=0.1)],
            p=0.8
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3)],
            p=0.5
        ),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]

    dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True,
                                        transform=transforms.Compose(transform))
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **kwargs
    )


def _get_test_data_loader(batch_size, root, **kwargs):
    logger.info("Get test data loader")

    transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]

    dataset = torchvision.datasets.CIFAR10(root=root, train=False,
                                        transform=transforms.Compose(transform))

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )


def _average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def train(args):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ["RANK"] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info(
            "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
                args.backend, dist.get_world_size()
            )
            + "Current host rank is {}. Number of gpus: {}".format(dist.get_rank(), args.num_gpus)
        )

    # Set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader = _get_train_data_loader(args.batch_size, args.data_dir, is_distributed, **kwargs)
    test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir, **kwargs)

    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.debug(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )

    model = Net().to(device)
    if is_distributed and use_cuda:
        # Multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # Single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    best_test_acc = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target) 
            loss.backward()
            if is_distributed and not use_cuda:
                # Average gradients manually for multi-machine cpu case only
                _average_gradients(model)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        test_acc = test(model, test_loader, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc 
    save_model(model, args.model_dir)

    logger.info(f"Best test accuracy: {best_test_acc:.2f}%")


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # Sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )

    return accuracy


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Net())
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for testing (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="M", help="SGD momentum (default: 0.9)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container environment
    parser.add_argument("--hosts", type=list)
    parser.add_argument("--current-host", type=str)
    parser.add_argument("--model-dir", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--num-gpus", type=int)

    train(parser.parse_args())
