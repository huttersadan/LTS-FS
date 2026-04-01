dataset_roots = {
    "lure": "",
    "chair": "",
    "pope": "",
    "opope": "",
    "mme":"",
    "llava_bench": "",
    "mmmu":""
}


def build_dataset(dataset_name, split, sampling, num_samples,st_idx,ed_idx):
    if dataset_name == "lure":
        from .LURE import LUREDataset
        dataset = LUREDataset(split, dataset_roots[dataset_name], sampling, num_samples)
    elif dataset_name == "chair":
        from .CHAIR import CHAIRDataset
        dataset = CHAIRDataset(split, dataset_roots[dataset_name], sampling, num_samples)
    elif dataset_name == "pope":
        from .POPE import POPEDataset
        dataset = POPEDataset(split, dataset_roots[dataset_name], sampling, num_samples,st_idx,ed_idx)
    elif dataset_name == "opope":
        from .OPOPE import POPEDataset
        dataset = POPEDataset(split, dataset_roots[dataset_name], sampling, num_samples)
    elif dataset_name == "mme":
        from .MME import MMEDataset
        dataset = MMEDataset(split, dataset_roots[dataset_name], sampling, num_samples)
    elif dataset_name == "llava_bench":
        from .LLava_bench import llavabenchDataset
        dataset = llavabenchDataset(split, dataset_roots[dataset_name], sampling, num_samples)
    elif dataset_name == "mmmu":
        from .MMMU import mmmuDataset
        dataset = mmmuDataset(split,dataset_roots[dataset_name],sampling=sampling, num_samples=num_samples)
    else:
        from .base import BaseDataset
        dataset = BaseDataset()
        
    return dataset.get_data()
