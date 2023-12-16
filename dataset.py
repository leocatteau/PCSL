import torch
import torchvision

def mnist(seed_trainset, seed_testset, ptr, pte):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),  # Convert each image into a torch.FloatTensor
        torchvision.transforms.Normalize((0.1307,), (0.2890,))  # Normalize the data to have zero mean and 1 stdv
    ])
    train_set = torchvision.datasets.MNIST('~/.torchvision/datasets/MNIST', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST('~/.torchvision/datasets/MNIST', train=False, transform=transform)

    if len(train_set)<ptr:
        print("The length of the training set is smaller than the required number of points.", flush=True)
        ptr = len(train_set)
    if len(test_set)<pte:
        print("The length of the test set is smaller than the required number of points.", flush=True)
        pte = len(test_set)

    gen       = torch.manual_seed(seed_trainset)
    train_idx = torch.randperm(len(train_set), generator=gen)[:ptr]
    gen       = torch.manual_seed(seed_testset)
    test_idx  = torch.randperm(len(test_set),  generator=gen)[:pte]
    xtr = torch.stack([x for x, _ in train_set])
    ytr = torch.tensor([y for _, y in train_set])
    xte = torch.stack([x for x, _ in test_set])
    yte = torch.tensor([y for _, y in test_set])
    
    return xtr[train_idx], ytr[train_idx], xte[test_idx], yte[test_idx]


def cifar10(seed_trainset, seed_testset, ptr, pte):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # Convert each image into a torch.FloatTensor
        torchvision.transforms.Normalize((0.4914008, 0.4821590, 0.4465309,), (0.2470322, 0.2434851, 0.2615879,))  # Normalize the data to have zero mean and 1 stdv
    ])
    train_set = torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=False, transform=transform)

    if len(train_set)<ptr:
        print("The length of the training set is smaller than the required number of points.", flush=True)
        ptr = len(train_set)
    if len(test_set)<pte:
        print("The length of the test set is smaller than the required number of points.", flush=True)
        pte = len(test_set)

    gen       = torch.manual_seed(seed_trainset)
    train_idx = torch.randperm(len(train_set), generator=gen)[:ptr]
    gen       = torch.manual_seed(seed_testset)
    test_idx  = torch.randperm(len(test_set),  generator=gen)[:pte]
    xtr = torch.stack([x for x, _ in train_set])
    ytr = torch.tensor([y for _, y in train_set])
    xte = torch.stack([x for x, _ in test_set])
    yte = torch.tensor([y for _, y in test_set])
    
    return xtr[train_idx], ytr[train_idx], xte[test_idx], yte[test_idx]

    
def gaussian_multiclass(d, nc, seed_trainset, seed_testset, ptr, pte):
    gen = torch.random.manual_seed(seed_trainset)
    xtr = torch.randn(ptr, d, generator=gen)
    teacher = torch.randn(d, nc)
    ytr = torch.argmax(xtr @ teacher, dim=1)

    gen = torch.random.manual_seed(seed_testset)
    xte = torch.randn(pte, d, generator=gen)
    yte = torch.argmax(xte @ teacher, dim=1)

    return xtr, ytr, xte, yte