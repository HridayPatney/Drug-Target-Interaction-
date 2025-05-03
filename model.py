import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class MultimodalHyperAttentionDTI(nn.Module):
    def __init__(self, config):
        super(MultimodalHyperAttentionDTI, self).__init__()
        self.dim = config.char_dim
        self.conv = config.conv
        self.drug_MAX_LENGTH = config.drug_max_length
        self.drug_kernel = config.drug_kernel
        self.protein_MAX_LENGTH = config.protein_max_length
        self.protein_kernel = config.protein_kernel
        self.image_size = config.image_size
        self.k_gram_sizes = [2, 3, 4]  # K-gram sizes for sequence processing

        # Embeddings for protein and drug SMILES
        self.protein_embed = nn.Embedding(26, self.dim, padding_idx=0)
        self.drug_embed = nn.Embedding(65, self.dim, padding_idx=0)

        # CNN for drug SMILES processing with k-grams
        self.Drug_CNNs = nn.ModuleList([
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=k)
            for k in self.k_gram_sizes
        ])
        
        # CNN for protein sequence processing with k-grams
        self.Protein_CNNs = nn.ModuleList([
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=k)
            for k in self.k_gram_sizes
        ])
        
        # Max pooling layers
        self.Drug_max_pools = nn.ModuleList([
            nn.MaxPool1d(self.drug_MAX_LENGTH - k + 1)
            for k in self.k_gram_sizes
        ])
        
        self.Protein_max_pools = nn.ModuleList([
            nn.MaxPool1d(self.protein_MAX_LENGTH - k + 1)
            for k in self.k_gram_sizes
        ])

        # Lightweight CNN-based image encoder (MCL-DTI style)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Multi-head self-attention for image features
        self.msa_layer = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        
        # Projection layer to match dimensions with SMILES features
        self.image_projection = nn.Linear(128, self.conv * len(self.k_gram_sizes))

        # Fusion layer for combining SMILES and image features
        self.fusion_layer = nn.Linear(self.conv * len(self.k_gram_sizes), self.conv * len(self.k_gram_sizes))

        # Attention mechanisms
        self.attention_layer = nn.Linear(self.conv * len(self.k_gram_sizes), self.conv * len(self.k_gram_sizes))
        self.protein_attention_layer = nn.Linear(self.conv * len(self.k_gram_sizes), self.conv * len(self.k_gram_sizes))
        self.drug_attention_layer = nn.Linear(self.conv * len(self.k_gram_sizes), self.conv * len(self.k_gram_sizes))

        # Dropout layers
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()

        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(self.conv * len(self.k_gram_sizes) * 2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

        # Image transformation
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_image(self, image):
        """Process a drug image and extract features using MCL-DTI style encoding"""
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Extract features using the image encoder
        batch_size = image.size(0)
        image_features = self.image_encoder(image)
        image_features = image_features.view(batch_size, 128)
        
        # Reshape for multi-head attention
        image_features = image_features.unsqueeze(1)  # [batch_size, 1, 128]
        
        # Apply self-attention
        attn_output, _ = self.msa_layer(image_features, image_features, image_features)
        image_features = attn_output.squeeze(1)  # [batch_size, 128]
        
        # Project to match dimension with SMILES features
        image_features = self.image_projection(image_features)
        
        return image_features.unsqueeze(2)  # [batch_size, conv*k_gram_count, 1]

    def forward(self, drug, protein, drug_image):
        # Process SMILES data
        drugembed = self.drug_embed(drug)
        proteinembed = self.protein_embed(protein)
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)
        
        # Process drug SMILES with k-grams
        drug_conv_outputs = []
        for i, conv in enumerate(self.Drug_CNNs):
            conv_output = self.relu(conv(drugembed))
            pooled_output = self.Drug_max_pools[i](conv_output)
            drug_conv_outputs.append(pooled_output)
        
        # Concatenate k-gram features
        drugConv = torch.cat(drug_conv_outputs, dim=1)
        
        # Process protein sequence with k-grams
        protein_conv_outputs = []
        for i, conv in enumerate(self.Protein_CNNs):
            conv_output = self.relu(conv(proteinembed))
            pooled_output = self.Protein_max_pools[i](conv_output)
            protein_conv_outputs.append(pooled_output)
        
        # Concatenate k-gram features
        proteinConv = torch.cat(protein_conv_outputs, dim=1)

        # Process image data
        image_features = self.process_image(drug_image)
        
        # Reshape image features to match drugConv dimensions
        batch_size = drugConv.shape[0]
        feature_dim = drugConv.shape[1]
        
        # Combine drug SMILES and image features
        combined_drug_features = torch.cat([drugConv, image_features], dim=2)
        
        # Reshape for fusion
        combined_drug_features_reshaped = combined_drug_features.permute(0, 2, 1)
        
        # Apply fusion to each element in the sequence
        batch_size, seq_len, feature_dim = combined_drug_features_reshaped.shape
        combined_drug_features_flat = combined_drug_features_reshaped.reshape(-1, feature_dim)
        fused_drug_features_flat = self.fusion_layer(combined_drug_features_flat)
        fused_drug_features_reshaped = fused_drug_features_flat.reshape(batch_size, seq_len, feature_dim)
        
        # Permute back to [batch_size, features, seq_len]
        fused_drug_features = fused_drug_features_reshaped.permute(0, 2, 1)

        # Apply attention mechanism
        drug_att = self.drug_attention_layer(fused_drug_features.permute(0, 2, 1))
        protein_att = self.protein_attention_layer(proteinConv.permute(0, 2, 1))
        
        # Create attention matrices
        d_att_layers = torch.unsqueeze(drug_att, 2).repeat(1, 1, proteinConv.shape[-1], 1)
        p_att_layers = torch.unsqueeze(protein_att, 1).repeat(1, fused_drug_features.shape[-1], 1, 1)
        
        # Calculate attention matrix
        Atten_matrix = self.attention_layer(self.relu(d_att_layers + p_att_layers))
        
        # Calculate attention weights
        Compound_atte = torch.mean(Atten_matrix, 2)
        Protein_atte = torch.mean(Atten_matrix, 1)
        Compound_atte = self.sigmoid(Compound_atte.permute(0, 2, 1))
        Protein_atte = self.sigmoid(Protein_atte.permute(0, 2, 1))
        
        # Apply attention weights
        fused_drug_features = fused_drug_features * 0.5 + fused_drug_features * Compound_atte
        proteinConv = proteinConv * 0.5 + proteinConv * Protein_atte
        
        # Global average pooling
        drugConv = torch.mean(fused_drug_features, dim=2)
        proteinConv = torch.mean(proteinConv, dim=2)
        
        # Concatenate drug and protein features
        pair = torch.cat([drugConv, proteinConv], dim=1)
        
        # Fully connected layers for prediction
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        
        return predict
