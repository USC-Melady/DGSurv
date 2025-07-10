import torch
from torch import nn
from sub_models.mirna_net import MiRNANet
from sub_models.mrna_net import MRNANet
from sub_models.clinical_net import ClinicalNet


class EncoderModel(nn.Module):
	def __init__(self, modalities, modality_fv_len, input_modality_dim, drop_out_p=0.5):
		super(EncoderModel, self).__init__()
		self.data_modalities = modalities
		self.dim = input_modality_dim
		self.modality_pipeline = {}
					
		if 'clinical' in self.data_modalities:
			self.clinical_submodel = ClinicalNet(m_length=modality_fv_len)
			self.modality_pipeline['clinical'] = self.clinical_submodel

		if 'mRNA' in self.data_modalities:
			self.mRNA_submodel = MRNANet(mrna_length=self.dim['mRNA'], m_length=modality_fv_len)
			self.modality_pipeline['mRNA'] = self.mRNA_submodel

		if 'miRNA' in self.data_modalities:
			self.miRNA_submodel = MiRNANet(mirna_length=self.dim['miRNA'], m_length=modality_fv_len)
			self.modality_pipeline['miRNA'] = self.miRNA_submodel
	
		if 'WSI' in self.data_modalities:
			img_fv_len = self.dim['WSI']
			self.img_pipline = torch.nn.Sequential(
					nn.Linear(img_fv_len, modality_fv_len),
					nn.BatchNorm1d(modality_fv_len),
					nn.ReLU(),
					nn.Dropout(drop_out_p)
				)
			self.modality_pipeline['WSI'] = self.img_pipline

		self.modality_count = len(modalities)
			
	def forward(self, x_modality):
		# Extract representations from different modalities
		representation = []
		for modality in self.data_modalities:
			if modality == 'clinical':
				representation.append(self.modality_pipeline['clinical'](x_modality['clinical_categorical'], x_modality['clinical_continuous']))
			else:
				representation.append(self.modality_pipeline[modality](x_modality[modality]))
		return representation