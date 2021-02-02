from pycocotools.coco import COCO
import torch
import cv2

import torchvision.transforms as transforms

import os


def transform(image, dsize=(256, 256)):
    resized_image = cv2.resize(image, dsize)
    torch_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    return torch_transform(resized_image)


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder_path, anns_file_path, tokenizer, transform=transform):
        self.images = img_folder_path
        self.coco = COCO(anns_file_path)
        self.key_map = {index: key for index, key in enumerate(self.coco.anns.keys())}

        self.tokenizer = tokenizer
        self.transform = transform

    def get_image(self, image_id):
        file_name = self.coco.loadImgs(ids=[image_id])[0]["file_name"]
        image = cv2.cvtColor(cv2.imread(self.images + file_name), cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def tokenize_caption(self, caption):
        tokens = self.tokenizer(caption, return_tensors="pt")["input_ids"].cpu().numpy().tolist()[0]
        return tokens

    def __getitem__(self, index):
        coco_object = self.coco.anns[self.key_map[index]]
        image = self.get_image(coco_object["image_id"])
        caption = self.tokenize_caption(coco_object["caption"])
        return image, caption

    def __len__(self):
        return len(os.listdir(self.images))
