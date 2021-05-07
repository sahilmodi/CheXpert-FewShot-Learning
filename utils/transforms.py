import torchvision.transforms as transforms

def get_transforms(height, width, split):
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        # transforms.ToTensor(),
        # transforms.Normalize(128, 64),
        # transforms.ToPILImage(),
        transforms.Lambda(lambda x: transforms.functional.equalize(x)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    if split == 'train':
        transform = transforms.Compose([
            transform,
            transforms.RandomAffine(
                degrees=(-15, 15),
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
        ])
    return transform

def label_to_binary_class(label, classes):
    return int(f'0b{"".join([str(int(label[c])) for c in classes])}', 2)
