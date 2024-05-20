"""
Anonymous authors, 2023/24

This file contains the implementation of a Featured Transfer Model, which is a PyTorch module used for fine-tuning encoders on validation datasets with full node and edge features. The model consists of generic node and edge encoder modules, as well as a projection layer and an output layer.

The FeaturedTransferModel class is the main module in this file. It takes an encoder module as input, which is responsible for encoding node and edge features. The encoder can be customized with various parameters such as the number of layers, dimensionality of features, and the number of layers in the input head.

The GenericNodeEncoder and GenericEdgeEncoder classes are used to encode node and edge features, respectively. They are generic modules that transform input features into embeddings. These modules can be customized with parameters such as the dimensionality of input and output embeddings, as well as the number of layers in the encoder.

The FeaturedTransferModel class also includes methods for initializing the embeddings of the model and performing the forward pass. The forward method takes input tensors such as batch, node features, edge index, edge attributes, and edge weights (optional), and returns the output tensor and node embeddings tensor.

Large sections are from the AD-GCL paper:

Susheel Suresh, Pan Li, Cong Hao, Georgia Tech, and Jennifer Neville. 2021.
Adversarial Graph Augmentation to Improve Graph Contrastive Learning.

In Advances in Neural Information Processing Systems,
Vol. 34. 15920–15933.
https://github.com/susheels/adgcl
"""

import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F
import numpy as np
from unsupervised.convs import GINEConv
from torch_geometric.nn import GCNConv, GATv2Conv
from utils import normalize_l2


class GenericEdgeEncoder(torch.nn.Module):
	"""
	A generic edge encoder module that transforms input features into embeddings.

	Args:
		emb_dim (int): The dimensionality of the output embeddings.
		feat_dim (int): The dimensionality of the input features.
		n_layers (int, optional): The number of layers in the encoder. Defaults to 1.
		node_feature_dim (int, optional): The dimensionality of the node features. Defaults to 512.
		edge_feature_dim (int, optional): The dimensionality of the edge features. Defaults to 512.
		input_head_layers (int, optional): The number of layers in the input head. Defaults to 3.
	"""

	def __init__(self, emb_dim, feat_dim, n_layers=1,
				 node_feature_dim=512, edge_feature_dim=512,
				 input_head_layers=3
				 ):
		super(GenericEdgeEncoder, self).__init__()
		self.layers = []
		spread_layers = [min(emb_dim, feat_dim) + np.abs(feat_dim - emb_dim) * i for i in range(n_layers - 1)]

		layer_sizes = [feat_dim] + spread_layers + [emb_dim]
		for i in range(n_layers):
			lin = Linear(layer_sizes[i], layer_sizes[i + 1])

			# lin = Linear(feat_dim, emb_dim)
			torch.nn.init.xavier_uniform_(lin.weight.data)
			self.layers.append(lin)

			if i != n_layers:
				self.layers.append(ReLU())

		self.model = Sequential(*self.layers)
		self.init_emb()

	def forward(self, x):
		"""
		Forward pass of the encoder.

		Args:
			x (torch.Tensor): Input features.

		Returns:
			torch.Tensor: Output embeddings.
		"""
		return self.model(x.float())

	def init_emb(self):
		"""
		Initialize the embeddings of the model.
		"""
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

class GenericNodeEncoder(torch.nn.Module):
	"""
	A generic node encoder module.

	Args:
		emb_dim (int): The dimension of the output embeddings.
		feat_dim (int): The dimension of the input features.
		n_layers (int, optional): The number of layers in the encoder. Defaults to 1.
	"""

	def __init__(self, emb_dim, feat_dim, n_layers=1):
		super(GenericNodeEncoder, self).__init__()

		self.layers = []

		spread_layers = [min(emb_dim, feat_dim) + np.abs(feat_dim - emb_dim) * i for i in range(n_layers - 1)]

		layer_sizes = [feat_dim] + spread_layers + [emb_dim]
		for i in range(n_layers):
			lin = Linear(layer_sizes[i], layer_sizes[i + 1])
			torch.nn.init.xavier_uniform_(lin.weight.data)
			self.layers.append(lin)

			if i != n_layers:
				self.layers.append(ReLU())

		self.model = Sequential(*self.layers)

		self.init_emb()

	def forward(self, x):
		"""
		Forward pass of the encoder.

		Args:
			x (torch.Tensor): The input tensor.

		Returns:
			torch.Tensor: The output tensor.
		"""
		return self.model(x.float())

	def init_emb(self):
		"""
		Initialize the embeddings of the model.
		"""
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	

class FeaturedTransferModel(torch.nn.Module):
	"""
	A PyTorch module representing a Featured Transfer Model.

	Args:
		encoder (torch.nn.Module): The encoder module used for node and edge feature encoding.
		proj_hidden_dim (int): The hidden dimension size for the projection layer. Default is 300.
		output_dim (int): The output dimension size. Default is 300.
		features (bool): Whether to use input features or not. Default is False.
		node_feature_dim (int): The dimension size of the node features. Default is 512.
		edge_feature_dim (int): The dimension size of the edge features. Default is 512.
		input_head_layers (int): The number of layers in the input head. Default is 3.
	"""

	def __init__(self, encoder, proj_hidden_dim=300, output_dim=300, features=False,
				 node_feature_dim=512, edge_feature_dim=512, input_head_layers=3):
		super(FeaturedTransferModel, self).__init__()

		self.encoder = encoder
		self.num_gc_layers = encoder.num_gc_layers
		self.convs = encoder.convs
		self.bns = encoder.bns
		self.drop_ratio = encoder.drop_ratio
		self.input_proj_dim = self.encoder.out_graph_dim
		self.features = features
		self.node_feature_dim = node_feature_dim
		self.edge_feature_dim = edge_feature_dim
		self.input_head_layers = input_head_layers

		self.atom_encoder = GenericNodeEncoder(proj_hidden_dim, node_feature_dim, n_layers=self.input_head_layers)
		self.bond_encoder = GenericEdgeEncoder(proj_hidden_dim, edge_feature_dim, n_layers=self.input_head_layers)
		self.output_layer = Sequential(Linear(self.input_proj_dim, proj_hidden_dim), ReLU(inplace=True),
									   Linear(proj_hidden_dim, output_dim))

		self.convolution = encoder.convolution

		self.init_emb()


	def init_emb(self):
		"""
		Initialize the embeddings of the model.
		"""
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, batch, x, edge_index, edge_attr, edge_weight=None, training = False):
		"""
		Forward pass of the Featured Transfer Model.

		Args:
			batch (Tensor): The batch tensor.
			x (Tensor): The input tensor.
			edge_index (LongTensor): The edge index tensor.
			edge_attr (Tensor): The edge attribute tensor.
			edge_weight (Tensor, optional): The edge weight tensor. Default is None.

		Returns:
			z (Tensor): The output tensor.
			node_emb (Tensor): The node embeddings tensor.
		"""
		if not self.features or training:
			x = torch.ones_like(x)
			edge_attr = torch.ones_like(edge_attr)
		x = self.atom_encoder(x.to(torch.int))
		edge_attr = self.bond_encoder(edge_attr.to(torch.int))
		# compute node embeddings using GNN
		xs = []
		for i in range(self.num_gc_layers):

			if edge_weight is None:
				edge_weight = torch.ones((edge_index.shape[1], 1)).to(x.device)
			if self.convolution == GINEConv:
				x = self.convs[i](x, edge_index, edge_attr, edge_weight)
			elif self.convolution == GCNConv:
				x = self.convs[i](x, edge_index, edge_weight)
			elif self.convolution == GATv2Conv:
				x = self.convs[i](x, edge_index)

			x = self.bns[i](x)
			if i == self.num_gc_layers - 1:
				# remove relu for the last layer
				x = F.dropout(x, self.drop_ratio, training=self.training)
			else:
				x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
			xs.append(x)

		node_emb = x
		z = global_add_pool(x, batch)

		normalize_l2(z)

		z = self.output_layer(z)
		return z, node_emb


class FeaturedEncoder(torch.nn.Module):
	"""
	A PyTorch module representing a Featured Transfer Model.

	Args:
		encoder (torch.nn.Module): The encoder module used for node and edge feature encoding.
		proj_hidden_dim (int): The hidden dimension size for the projection layer. Default is 300.
		output_dim (int): The output dimension size. Default is 300.
		features (bool): Whether to use input features or not. Default is False.
		node_feature_dim (int): The dimension size of the node features. Default is 512.
		edge_feature_dim (int): The dimension size of the edge features. Default is 512.
		input_head_layers (int): The number of layers in the input head. Default is 3.
	"""

	def __init__(self, encoder, proj_hidden_dim=300, output_dim=300, features=False,
				 node_feature_dim=512, edge_feature_dim=512, input_head_layers=3):
		super(FeaturedEncoder, self).__init__()

		self.encoder = encoder
		self.num_gc_layers = encoder.num_gc_layers
		self.convs = encoder.convs
		self.bns = encoder.bns
		self.drop_ratio = encoder.drop_ratio
		self.input_proj_dim = self.encoder.out_graph_dim
		self.out_graph_dim = self.input_proj_dim
		self.out_node_dim = self.out_graph_dim
		self.features = features
		self.node_feature_dim = node_feature_dim
		self.edge_feature_dim = edge_feature_dim
		self.input_head_layers = input_head_layers

		

		self.atom_encoder = GenericNodeEncoder(proj_hidden_dim, node_feature_dim, n_layers=self.input_head_layers)
		self.bond_encoder = GenericEdgeEncoder(proj_hidden_dim, edge_feature_dim, n_layers=self.input_head_layers)

		self.convolution = encoder.convolution

		# self.init_emb()

	def init_emb(self):
		"""
		Initialize the embeddings of the model.
		"""
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, batch, x, edge_index, edge_attr, edge_weight=None, training = False):
		"""
		Forward pass of the Featured Transfer Model.

		Args:
			batch (Tensor): The batch tensor.
			x (Tensor): The input tensor.
			edge_index (LongTensor): The edge index tensor.
			edge_attr (Tensor): The edge attribute tensor.
			edge_weight (Tensor, optional): The edge weight tensor. Default is None.

		Returns:
			z (Tensor): The output tensor.
			node_emb (Tensor): The node embeddings tensor.
		"""
		# if not self.features or training:
		# 	x = torch.ones_like(x)
		# 	edge_attr = torch.ones_like(edge_attr)

		x = self.atom_encoder(x.to(torch.int))
		edge_attr = self.bond_encoder(edge_attr.to(torch.int))
		# compute node embeddings using GNN
		xs = []
		for i in range(self.num_gc_layers):

			if edge_weight is None:
				edge_weight = torch.ones((edge_index.shape[1], 1)).to(x.device)
			if self.convolution == GINEConv:
				x = self.convs[i](x, edge_index, edge_attr, edge_weight)
			elif self.convolution == GCNConv:
				x = self.convs[i](x, edge_index, edge_weight)
			elif self.convolution == GATv2Conv:
				x = self.convs[i](x, edge_index)

			x = self.bns[i](x)
			if i == self.num_gc_layers - 1:
				# remove relu for the last layer
				x = F.dropout(x, self.drop_ratio, training=self.training)
			else:
				x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
			xs.append(x)

		node_emb = x
		z = global_add_pool(x, batch)

		z = normalize_l2(z)

		return z, node_emb


	def get_embeddings(self, loader, device, is_rand_label=False, every=1, node_features=False):
		"""
		Computes embeddings for a given data loader.

		Args:
			loader (DataLoader): The data loader.
			device (torch.device): The device to perform computations on.
			is_rand_label (bool, optional): Whether to use random labels. Defaults to False.
			every (int, optional): The interval at which to compute embeddings. Defaults to 1.
			node_features (bool, optional): Whether to use node features. Defaults to False.

		Returns:
			Tuple[np.ndarray, np.ndarray]: The computed embeddings and labels.

		"""
		ret = []
		y = []
		with torch.no_grad():
			for i, data in enumerate(loader):
				if i % every != 0:
					continue

				if isinstance(data, list):
					data = data[0].to(device)

				data = data.to(device)
				batch, x, edge_index, edge_attr = data.batch, data.x, data.edge_index, data.edge_attr

				edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

				if x is None:
					x = torch.ones((batch.shape[0], 1)).to(device)

				x, _ = self.forward(batch, x, edge_index, edge_attr, edge_weight)

				ret.append(x.cpu().numpy())

				try:
					if is_rand_label:
						y.append(data.rand_label.cpu().numpy())
					else:
						y.append(data.y.cpu().numpy())
				except AttributeError:
					y.append(torch.ones(x.shape[0]).to(torch.float))
		ret = np.concatenate(ret, 0)
		y = np.concatenate(y, 0)
		return ret, y