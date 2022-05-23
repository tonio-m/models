import os
import torch
from sys import argv
from PIL import Image
from torchvision import transforms


def preprocess(files):
  _preprocess = transforms.Compose([
      transforms.Resize(299),
      transforms.CenterCrop(299),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  images = [Image.open(f) for f in files]
  batch = [_preprocess(i) for i in images]
  print(batch)
  return batch


def get_top_categories(output,n=3):
  with open("imagenet_classes.txt", "r") as f:
    probabilities = torch.nn.functional.softmax(output, dim=0)
    categories = [s.strip() for s in f.readlines()]
    top_prob, top_catid = torch.topk(probabilities, n)
    print(top_catid)
    return [categories[id] for id in top_catid]


def main(files):
  model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
  input_batch = preprocess(files)
  if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
  with torch.no_grad():
    output = model(input_batch)
  categories = get_top_categories(output,3)
  print(categories)



if __name__ == '__main__':
  args = argv[1:]
  files = []
  for folder in args:
    path = os.path.expanduser(folder)
    files += [os.path.join(path,f) for f in os.listdir(folder) if os.path.isfile(os.path.join(path,f))]
  main(files[:1])

