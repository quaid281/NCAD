import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TCNLayer(nn.Module):
    """Temporal Convolutional Network layer with residual connections (PyTorch)."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate, dropout_rate=0.2):
        super(TCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        
        self.causal_padding = (kernel_size - 1) * dilation_rate
        
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0, 
            dilation=dilation_rate
        )
        self.relu1 = nn.ReLU() 
        self.layer_norm1 = nn.LayerNorm(out_channels)
        self.dropout1 = nn.Dropout(dropout_rate) # <<< PYTORCH DROPOUT
        
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0, 
            dilation=dilation_rate
        )
        self.relu2 = nn.ReLU() 
        self.layer_norm2 = nn.LayerNorm(out_channels)
        self.dropout2 = nn.Dropout(dropout_rate) # <<< PYTORCH DROPOUT
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None
    
    def forward(self, inputs):
        x_padded = F.pad(inputs, (self.causal_padding, 0)) 
        x = self.conv1(x_padded)
        x = self.relu1(x)
        # Apply LayerNorm on the (N, C, L) tensor. It normalizes over the C dimension.
        # To do this, we permute, apply norm, and permute back.
        x = x.permute(0, 2, 1) # (N, C, L) -> (N, L, C)
        x = self.layer_norm1(x)
        x = x.permute(0, 2, 1) # (N, L, C) -> (N, C, L)
        x = self.dropout1(x) # This will now use nn.Dropout's forward
        
        x_padded = F.pad(x, (self.causal_padding, 0))
        x = self.conv2(x_padded)
        x = self.relu2(x)
        x = x.permute(0, 2, 1) # (N, C, L) -> (N, L, C)
        x = self.layer_norm2(x)
        x = x.permute(0, 2, 1) # (N, L, C) -> (N, C, L)
        x = self.dropout2(x) # This will now use nn.Dropout's forward
        
        if self.residual_conv:
            res = self.residual_conv(inputs)
        else:
            res = inputs
            
        return self.relu2(x + res) 

    
class TCNEncoder(nn.Module):
    def __init__(self, sequence_length, input_dim, latent_dim=128, projection_dim=64,
                 tcn_layers=4, filters=128, kernel_size=5, pooling_strategy="hybrid"):
        super(TCNEncoder, self).__init__()
        self.sequence_length = sequence_length 
        self.input_dim = input_dim 
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim
        self.pooling_strategy = pooling_strategy  # Options: "last", "hybrid", "traditional"
        
        self.input_projection = nn.Conv1d(input_dim, filters, kernel_size=1) 
        
        tcn_module_list = []
        current_channels = filters
        for i in range(tcn_layers):
            dilation_rate = 2 ** i
            tcn_module_list.append(
                TCNLayer(in_channels=current_channels, 
                         out_channels=filters, 
                         kernel_size=kernel_size, 
                         dilation_rate=dilation_rate)
            )
            current_channels = filters 
        self.encoder_layers = nn.Sequential(*tcn_module_list)
        
        # Keep these for backward compatibility and hybrid approach
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # For "last" or "hybrid" pooling strategies
        if pooling_strategy == "last":
            # For last time step only - no need for any pooling layers
            self.dense1 = nn.Linear(filters, 128)
        elif pooling_strategy == "hybrid":
            # For hybrid approach (last time step + traditional pooled features)
            self.dense1 = nn.Linear(3 * filters, 128)  # last + max + avg
        else:  # "traditional"
            # Original approach with max and avg pooling only
            self.dense1 = nn.Linear(2 * filters, 128)
        
        self.relu_dense1 = nn.ReLU()
        self.ln1 = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(0.3)        
        # --- CAPACITY INCREASE HERE ---
        self.spectral_feature_dim = 9 * filters 
        # Increase the size of the dense layer for spectral features
        self.spectral_dense = nn.Linear(self.spectral_feature_dim, 128)  # From 64 to 128
        self.relu_spectral = nn.ReLU()
        # The LayerNorm must match the new size
        self.spectral_ln = nn.LayerNorm(128)  # From 64 to 128
        
        # The bottleneck input size must be updated
        self.bottleneck_dense = nn.Linear(128 + 128, latent_dim)  # From 128+64 to 128+128
        # --- END CAPACITY INCREASE ---
        
        self.projection_dense1 = nn.Linear(latent_dim, 64)
        self.relu_projection1 = nn.ReLU()
        self.projection_ln = nn.LayerNorm(64)
        self.projection_dropout = nn.Dropout(0.2)
        self.projection_output = nn.Linear(64, projection_dim)

    def _extract_spectral_features(self, x): # x: (batch, filters, seq_len)
        # ... (implementation should be fine, it uses torch ops) ...
        batch_size, n_filters, seq_len = x.shape
        safe_pool_size_16 = min(16, seq_len // 2 if seq_len // 2 > 0 else 1)
        safe_pool_size_4 = min(4, seq_len // 4 if seq_len // 4 > 0 else 1)
        mean_over_time = torch.mean(x, dim=2)                  
        std_over_time = torch.std(x, dim=2)                    
        max_over_time = torch.max(x, dim=2).values             
        min_over_time = torch.min(x, dim=2).values             
        range_over_time = max_over_time - min_over_time        
        
        # --- MODIFIED PADDING ---
        # For nn.AvgPool1d to achieve 'same'-like output dimension with stride=1,
        # padding = (kernel_size - 1) // 2 (if kernel_size is odd)
        # or handle even kernel_size carefully.
        # A simple way is to calculate padding needed to keep dimensions roughly the same.
        # If kernel_size is k, padding p, input L_in, output L_out:
        # L_out = floor((L_in + 2*p - k) / stride) + 1
        # For stride=1, L_out = L_in + 2*p - k + 1.
        # We want L_out ~= L_in. So, 2*p - k + 1 ~= 0 => p ~= (k-1)/2.

        # Padding for low_freq_pool
        padding_low = (safe_pool_size_16 - 1) // 2
        low_freq_pool = nn.AvgPool1d(kernel_size=safe_pool_size_16, stride=1, padding=padding_low)
        low_freq = low_freq_pool(x)
        # No need for F.interpolate if padding is correct for 'same' with stride 1
        # However, due to floor operations, it might still be off by one,
        # so interpolation can be a safety net if exact same length is critical.
        # If exact length is needed and padding_low doesn't achieve it perfectly:
        if low_freq.shape[2] != seq_len: 
            low_freq = F.interpolate(low_freq, size=seq_len, mode='linear', align_corners=False)
        low_freq_energy = torch.mean(torch.square(low_freq), dim=2)
        
        mid_freq = x - low_freq 
        mid_freq_energy = torch.mean(torch.square(mid_freq), dim=2)
        
        # Padding for high_freq_pool
        padding_high = (safe_pool_size_4 - 1) // 2
        high_freq_pool = nn.AvgPool1d(kernel_size=safe_pool_size_4, stride=1, padding=padding_high)
        smoothed_for_high_freq = high_freq_pool(x)
        if smoothed_for_high_freq.shape[2] != seq_len:
             smoothed_for_high_freq = F.interpolate(smoothed_for_high_freq, size=seq_len, mode='linear', align_corners=False)
        high_freq = x - smoothed_for_high_freq 
        high_freq_energy = torch.mean(torch.square(high_freq), dim=2)
        
        signs = torch.sign(x)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs) 
        sign_changes = torch.abs(signs[:, :, 1:] - signs[:, :, :-1])
        zero_crossing_rate = torch.mean(sign_changes, dim=2) * 0.5
        
        spectral_features_list = [
            mean_over_time, std_over_time, 
            max_over_time, min_over_time, range_over_time,
            low_freq_energy, mid_freq_energy, high_freq_energy,
            zero_crossing_rate
        ]
        spectral_features = torch.cat(spectral_features_list, dim=-1) 
        return spectral_features

    def forward(self, inputs, with_projection=True): 
        x = inputs.permute(0, 2, 1)  # (batch, seq_len, features) -> (batch, features, seq_len)
        
        x = self.input_projection(x)
        x = self.encoder_layers(x)  # Output: (batch, filters, seq_len)
        
        # Apply pooling based on strategy
        if self.pooling_strategy == "last":
            # Take only the last time step (most contextual information) - no pooling needed
            # x is (batch, filters, seq_len), we want the last time step
            last_step_features = x[:, :, -1]  # Shape: (batch, filters)
            combined_pooled = last_step_features
        elif self.pooling_strategy == "hybrid":
            # Combine last time step with global pooling for best of both worlds
            last_step_features = x[:, :, -1]  # Shape: (batch, filters)
            max_features = self.max_pool(x).squeeze(-1)  # Shape: (batch, filters)
            avg_features = self.avg_pool(x).squeeze(-1)  # Shape: (batch, filters)
            combined_pooled = torch.cat([last_step_features, max_features, avg_features], dim=-1)
        else:  # "traditional"
            # Original approach with max and avg pooling only
            max_features = self.max_pool(x).squeeze(-1)
            avg_features = self.avg_pool(x).squeeze(-1)
            combined_pooled = torch.cat([max_features, avg_features], dim=-1)
        
        time_features = self.dense1(combined_pooled)
        time_features = self.relu_dense1(time_features)
        time_features = self.ln1(time_features)
        time_features = self.dropout1(time_features)
        
        spectral_features_extracted = self._extract_spectral_features(x)
        spectral_features = self.spectral_dense(spectral_features_extracted)
        spectral_features = self.relu_spectral(spectral_features)
        spectral_features = self.spectral_ln(spectral_features)
        
        combined_features = torch.cat([time_features, spectral_features], dim=-1)
        
        latent = self.bottleneck_dense(combined_features)        
        if not with_projection:
            return latent
            
        projection = self.projection_dense1(latent)
        projection = self.relu_projection1(projection)
        projection = self.projection_ln(projection)
        projection = self.projection_dropout(projection)
        projection = self.projection_output(projection)
        
        projection = F.normalize(projection, p=2, dim=1) 
        
        return latent, projection