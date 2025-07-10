import os
import torch
import numpy as np
import pandas as pd


def preprocess_clinical_data(clinical_path):
    data_clinical = pd.read_csv(clinical_path, header=None)
    target_data = data_clinical[[6, 7]]
    clin_data_categorical = data_clinical[[1, 2, 3, 4]]
    clin_data_continuous = data_clinical[[5]]
    return clin_data_categorical, clin_data_continuous, target_data, data_clinical[[0]] 


class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode, modalities, data_path, remove_missing=False):
        super(MultiModalDataset, self).__init__()
        split_file = os.path.join(args.split_path, f"split_{args.fold}.npy")
        all_splits = np.load(split_file, allow_pickle=True).reshape(-1)[0]
        self.all_index = all_splits[mode]
        self.data_modalities = modalities
        clin_data_categorical, clin_data_continuous, target_data, clinical_label = preprocess_clinical_data(data_path['clinical'])
        self.target = target_data.to_numpy()[self.all_index]
        self.clinical_label = clinical_label.to_numpy()[self.all_index]
        self.non_zero_index = []

        if 'clinical' in self.data_modalities:
            self.clin_cat = clin_data_categorical.to_numpy()[self.all_index]
            self.clin_cont = clin_data_continuous.to_numpy()[self.all_index]
            self.non_zero_index.append(np.ones(len(self.clin_cont)) == 1)

        if 'mRNA' in self.data_modalities:
            data_mrna = pd.read_csv(data_path['mRNA'], header=None)
            self.data_mrna = data_mrna.to_numpy()[self.all_index]
            self.non_zero_index.append(self.data_mrna.sum(axis=1) != 0)

        if 'miRNA' in self.data_modalities:
            data_mirna = pd.read_csv(data_path['miRNA'], header=None)
            self.data_mirna = data_mirna.to_numpy()[self.all_index]
            self.non_zero_index.append(self.data_mirna.sum(axis=1) != 0)

        if 'WSI' in self.data_modalities:
            self.wsi_dict = torch.load(data_path['WSI'])
            self.all_wsi_data = []
            missing_wsi = []
            for label in self.clinical_label:
                label = label[0]
                if label in self.wsi_dict:
                    wsi_data = self.wsi_dict[label]
                    missing_wsi.append(True)
                else:
                    wsi_data = torch.zeros(args.input_modality_dim['WSI'])
                    missing_wsi.append(False)
                self.all_wsi_data.append(wsi_data)
            self.all_wsi_data = torch.stack(self.all_wsi_data, 0)
            self.non_zero_index.append(np.array(missing_wsi))

        self.agg_missing = np.sum(self.non_zero_index, 0) != 0
        if remove_missing:
            if 'clinical' in self.data_modalities:
                self.clin_cat = self.clin_cat[self.agg_missing]
                self.clin_cont = self.clin_cont[self.agg_missing]

            if 'mRNA' in self.data_modalities:
                self.data_mrna = self.data_mrna[self.agg_missing]

            if 'miRNA' in self.data_modalities:
                self.data_mirna = self.data_mirna[self.agg_missing]

            if 'WSI' in self.data_modalities:
                self.all_wsi_data = self.all_wsi_data[self.agg_missing]
            self.target = self.target[self.agg_missing]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        data = {}
        data_label = {}
        target_y = np.array(self.target[index], dtype='int')
        target_y = torch.from_numpy(target_y)
        data_label = target_y.type(torch.LongTensor)
    
        if 'clinical' in self.data_modalities:
            clin_cate = np.array(self.clin_cat[index]).astype(np.int64)
            clin_cate = torch.from_numpy(clin_cate)
            data['clinical_categorical'] = clin_cate

            clin_conti = np.array(self.clin_cont[index]).astype(np.float32)
            clin_conti = torch.from_numpy(clin_conti)
            data['clinical_continuous'] = clin_conti

        if 'mRNA' in self.data_modalities:
            mrna = np.array(self.data_mrna[index])
            mrna = torch.from_numpy(mrna)
            data['mRNA'] = mrna.type(torch.FloatTensor)

        if 'miRNA' in self.data_modalities:
            mirna = np.array(self.data_mirna[index])
            mirna = torch.from_numpy(mirna)
            data['miRNA'] = mirna.type(torch.FloatTensor)
        
        if 'WSI' in self.data_modalities:
            data['WSI'] = self.all_wsi_data[index]
        
        data_mask = []
        for modality in self.data_modalities:
            if modality == 'clinical':
                data_mask.append(1)
            else:
                if torch.sum(data[modality]) == 0:
                    data_mask.append(0)
                else:
                    data_mask.append(1)
        data_mask = torch.tensor(data_mask)
        return data, data_label, data_mask