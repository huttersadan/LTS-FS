import os
import json
import random

from tqdm import tqdm

from dataset.base import BaseDataset
# class BaseDataset:
#     def __init__(self):
#         pass
    
#     def get_data(self):
#         return []
from utils.func import read_jsonl


class MMEDataset(BaseDataset):
    def __init__(self, split_list=["count"], data_root="", sampling="first", num_samples=None):
        super(MMEDataset, self).__init__()
        testfiles_dir = [data_root + "/" + inst for inst in split_list]
        self.testfiles_dir = testfiles_dir

        self.sampling = sampling
        self.num_samples = num_samples


    def get_data(self):

        val_data = {}

        for testfile in self.testfiles_dir:
            image_with_questions = os.listdir(testfile)
            image_file = [f for f in image_with_questions if not f.endswith('.txt')]
            txt_file = [f for f in image_with_questions if f.endswith('.txt')]

            #image_name = [f[:-4] for f in image_with_questions if not f.endswith('.txt')]

            # with open(testfile, 'r') as f:
            #     inputs = [json.loads(line) for line in f]

            if self.num_samples:
                if self.sampling == "first":
                    sampled_groups = image_file[:self.num_samples] 
                elif self.sampling == "random":
                    if self.num_samples > len(image_file):
                        sampled_groups = image_file
                    else:
                        sampled_groups = random.sample(image_file, self.num_samples) 
                else:
                    raise ValueError(f"Unsupported sampling strategy: {self.sampling}")
            else:
                sampled_groups = image_file

            strategy = testfile.split('/')[-1]
            val_data[strategy] = []
            for single_image_file in sampled_groups:
                with open(os.path.join(testfile, single_image_file.split('.')[0] + '.txt'), 'r') as f:
                    lines = f.readlines()
                question_ls, answer_ls = [], []
                for line in lines:
                    question,answer = line.strip().split('\t')
                    question_ls.append(question)
                    answer_ls.append(answer)
                #
                image_path = os.path.join(testfile,single_image_file)
                for q,a in zip(question_ls, answer_ls):
                    val_data[strategy].append({
                        "image_path": os.path.join(testfile,single_image_file),
                        "question": q,
                        "label": a
                    })
        return val_data

if __name__ == "__main__":
    dataset = MMEDataset(split_list=["count"], data_root="")
    data = dataset.get_data()
    for k in data:
        print(f"{k}: {len(data[k])} samples")
