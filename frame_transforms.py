import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional
import random


class RandomRotations():
    def __init__(self, angles:list = [-30, -15, 0, 15, 30]) -> None:
        self.angles = angles
    def __call__(self, x):
        angle = random.choice(self.angles)
        return functional.rotate(x, angle)

class FrameTransforms():
    def __init__(self, frame_shape:int) -> None:
        self.preprocess = transforms.Compose([
            transforms.Resize((frame_shape, frame_shape)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            RandomRotations()
        ])
    
    def __call__(self, x:torch.Tensor):
        x = torch.transpose(x, 0, 2)
        x = self.preprocess(x)
        x = torch.transpose(x, 0, 2)
        return x

if __name__ == "__main__":
    # torch.Size([1, 1080, 1920, 3]
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        #transforms.CenterCrop(224),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
    ])
    #transform = transforms.Compose([
    #    transforms.PILToTensor()
    #    ])
    #resp = requests.get('https://upload.wikimedia.org/wikipedia/commons/b/b6/Image_created_with_a_mobile_phone.png')
    #img = Image.open('img_test.png')

    #img = transform(img)

    #print(img.shape)


    #img = torch.rand(1080, 1920, 3)
    img = torch.rand(1080, 2120, 3)
    img = torch.transpose(img, 0, 2)
    print(img.shape)
    img = preprocess(img)
    print(img.shape)

    img = torch.rand(1080, 2120, 3)
    processer = FrameTransforms(512)
    print(processer(img).shape)