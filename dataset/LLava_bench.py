import os
import random
import json
from dataset.base import BaseDataset




class llavabenchDataset(BaseDataset):
    def __init__(self,split="val", data_root = "",sampling="random", num_samples=500):
        super(llavabenchDataset, self).__init__()
        self.data_root = data_root
        self.question_path = data_root.replace("images","questions.jsonl")
        self.questions = []
        with open(self.question_path, 'r') as f:
            for line in f.readlines():
                self.questions.append(json.loads(line.strip()))


    def get_data(self):
        # image_ids = os.listdir(self.data_root)
        # data = []
        # for img_id in image_ids:
        #     img_path = os.path.join(self.data_root, img_id)
        #     data.append(
        #         {
        #             "image_id": img_id,
        #             "image_path": img_path,
        #             "question": 'Please describe this image in detail.',
        #         }
        #     )
        data = []
        for item in self.questions:
            image_id = item['image']
            img_path = os.path.join(self.data_root, image_id)
            question = item['text']
            data.append(
                {
                    "image_id": image_id,
                    "image_path": img_path,
                    "question": question,
                }
            )
        return data
        