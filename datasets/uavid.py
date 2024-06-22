import os
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset

class UAVDataset(Dataset):
    def __init__(self, data_folder, transform=None):

      self.data_folder = data_folder
      self.transform = transform
      self.data = []
      self.load_data()

    def load_data(self):
      for seq_dir in tqdm(os.listdir(self.data_folder)):

          image_folder = os.path.join(self.data_folder, seq_dir, "Images")
          label_folder = os.path.join(self.data_folder, seq_dir, "Labels")

          image_paths = os.listdir(image_folder)

          for image_path in image_paths:
            id = image_path.split('.')[0]

            image = str(os.path.join(image_folder, image_path))
            label = str(os.path.join(label_folder, id + ".npy"))

            sample = {
                'seq' : seq_dir,
                'id' : id,
                'image' : image,
                'label' : label,
            }

            self.data.append(sample)

    def __len__(self):
      return len(self.data)

    def get_label(self):
      return {
          'Background clutter' : 0,
          'Building' : 1,
          'Road' : 2,
          'Tree' : 3,
          'Low vegetation' : 4,
          'Moving car' : 5,
          'Static car' : 6,
          'Human' : 7
      }

    def __getitem__(self, index):
      sample = self.data[index]

      image = np.array(Image.open(sample['image']))
      mask = np.load(sample['label'])

      if self.transform is not None:
        transformed = self.transform(image=image, mask=mask) # The keyword must be "mask"
        image = transformed["image"]
        mask = transformed["mask"]

      image = image / 255.0
      mask = mask.long()

      return {
          'seq': sample['seq'],
          'id': sample['id'],
          'image': image,
          'label': mask,
      }

    def __repr__(self) -> str:
      body = f"""Dataset {self.__class__.__name__}
      Data directory: {self.data_folder}
      Number of datapoints: {len(self.data)}"""
      return body