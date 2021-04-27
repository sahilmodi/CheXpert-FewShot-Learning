import torch
import torchvision

def load_data(dataset, data_root, batch_size, num_workers):
    if dataset == "Omniglot":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((28, 28))
        ])
        train_ds = torchvision.datasets.Omniglot(
            root=data_root,
            download=True,
            background=True,
            transform=transform
        )
        test_ds = torchvision.datasets.Omniglot(
            root=data_root,
            download=True,
            background=False,
            transform=transform
        )

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

        return train_loader, test_loader
    
    else:
        raise Exception("Unimplemented dataset")