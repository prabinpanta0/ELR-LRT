import torch
import torch.nn as nn
from elr_lrt import patch_sequence
import numpy as np

class ELRLRTModel(nn.Module):
    """
    Efficient Low-Resource Latent Reasoning Transformer (ELR-LRT).

    This model is designed to work as a plug-and-play component with default settings,
    enabling deployment without requiring users to customize hyperparameters.
    Advanced users can override these defaults if needed.
    """
    def __init__(self, vocab_size=256, d_model=512, n_layers=6, n_heads=8,
                 theta=1.0, theta_r=0.5, tau=0.1):
        super(ELRLRTModel, self).__init__()
        self.d_model = d_model
        self.theta = theta
        self.theta_r = theta_r
        self.tau = tau
        
        # DBPM (Dynamic Byte Patching Module) parameters
        self.k = 5  # Context window for patching
        
        # Encoder for patch embedding
        self.byte_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True),
            num_layers=n_layers
        )
        
        # CLRM (Continuous Latent Reasoning Module) parameters
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable mixing parameter
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True),
            num_layers=n_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def patch_input(self, byte_seq):
        """Apply dynamic byte patching."""
        byte_list = byte_seq.tolist() if isinstance(byte_seq, torch.Tensor) else byte_seq
        patches = patch_sequence(byte_list, self.k, self.theta, self.theta_r)
        return patches
    
    def encode_patches(self, patches):
        """Encode patches into latent representations using mean pooling."""
        patch_embeddings = []
        for patch in patches:
            emb = self.byte_embedding(torch.tensor(patch))
            emb = emb.mean(dim=0)  # Mean pooling for the patch
            patch_embeddings.append(emb)
        return torch.stack(patch_embeddings)
    
    def gate_function(self, h_t):
        """
        Gating mechanism for CLRM.
        Returns a float tensor (1 if norm exceeds threshold, else 0).
        """
        norm = torch.norm(h_t, p=2, dim=-1)
        return (norm > self.tau).float()
    
    def forward(self, input_bytes, target=None):
        """
        Forward pass with dynamic byte patching and continuous latent reasoning.
        Processes input bytes to produce output logits without additional customization.
        """
        # Step 1: Dynamic Byte Patching
        patches = self.patch_input(input_bytes)
        z = self.encode_patches(patches)
        encoded = self.encoder(z.unsqueeze(0))  # [1, num_patches, d_model]
        
        # Step 2: Continuous Latent Reasoning
        h_t = encoded.squeeze(0)  # [num_patches, d_model]
        outputs = []
        for t in range(len(patches)):
            gate = self.gate_function(h_t[t])
            # Ensure the token is created on the same device as the input
            token = torch.tensor(patches[t][-1]).to(input_bytes.device)
            if gate > 0:
                # Use latent state: weighted combination of latent hidden state and token embedding
                e_t = self.alpha * h_t[t] + (1 - self.alpha) * self.byte_embedding(token)
            else:
                # Use token embedding only
                e_t = self.byte_embedding(token)
            decoded = self.decoder(e_t.unsqueeze(0).unsqueeze(0), encoded)  # [1, 1, d_model]
            output = self.output_layer(decoded.squeeze(0))  # [1, vocab_size]
            outputs.append(output)
        return torch.stack(outputs).squeeze(1)  # [num_patches, vocab_size]

    def rl_finetune(self, input_bytes, target_bytes, num_iterations=10):
        """
        Fine-tuning with supervised cross-entropy loss using a fixed AdamW optimizer.
        
        The defaults here are chosen for a plug-and-play experience. Users can update the optimizer
        or hyperparameters if needed, but in production these defaults should be sufficient.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for _ in range(num_iterations):
            outputs = self.forward(input_bytes)  # [num_patches, vocab_size]
            # Ensure target matches output length (num_patches)
            if len(target_bytes) != outputs.shape[0]:
                if len(target_bytes) > outputs.shape[0]:
                    target = target_bytes[:outputs.shape[0]]
                else:
                    pad = torch.zeros(outputs.shape[0] - len(target_bytes),
                                      dtype=torch.long, device=input_bytes.device)
                    target = torch.cat([target_bytes, pad])
            else:
                target = target_bytes
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
