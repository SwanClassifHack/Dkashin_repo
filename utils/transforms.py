import torchvision

resnet_test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(), 
    torchvision.transforms.Resize(
        size=(224, 224)
    ),
    torchvision.transforms.ToTensor(),  
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])