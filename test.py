import torch
from torch.autograd import Variable
import net
import pathlib
from torchvision import transforms, datasets
from PIL import Image

containing_dir = pathlib.Path(__file__).resolve().parent

model = net.model
model.load_state_dict(torch.load(containing_dir / "net.pth"))
model.eval()


def load_image_tensor(filename, batch_size, image_shape=None):
    """Load an image for torch."""
    image = Image.open(filename)
    if image_shape:  # Downsample the image
        image = image.resize(image_shape, Image.ANTIALIAS)
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)])
    # Repeat the image so it matches the batch size for loss computations
    return image_transform(image)[:3].repeat(batch_size, 1, 1, 1)


original = load_image_tensor(
    containing_dir.parent / "images/faces/original.png", 1)
bw = load_image_tensor(
    containing_dir.parent / "images/faces/bw.png", 1)
stylized = load_image_tensor(
    containing_dir.parent / "images/faces/stylized.png", 1)
other = load_image_tensor(
    containing_dir.parent / "images/faces/other.png", 1)

# CPU
f1 = model(original)
f2 = model(bw)
f3 = model(stylized)
f4 = model(other)

print(torch.nn.functional.mse_loss(f1, f1))
print(torch.nn.functional.mse_loss(f1, f2))
print(torch.nn.functional.mse_loss(f1, f3))
print(torch.nn.functional.mse_loss(f1, f4))

breakpoint()
# print('CPU done')

# GPU
# model.cuda()
# input = input.cuda()
# output = model(Variable(input))
# print('GPU done')
