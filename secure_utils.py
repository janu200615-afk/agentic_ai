# secure_utils.py
# Simple symmetric encryption (AES-GCM) helpers and a toy DP function

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os
import torch

def generate_key():
    return AESGCM.generate_key(bit_length=128)

def encrypt_bytes(key: bytes, plaintext: bytes, associated_data: bytes = None):
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, associated_data)
    return nonce + ct  # prepend nonce

def decrypt_bytes(key: bytes, ciphertext: bytes, associated_data: bytes = None):
    aesgcm = AESGCM(key)
    nonce = ciphertext[:12]
    ct = ciphertext[12:]
    return aesgcm.decrypt(nonce, ct, associated_data)

def add_gaussian_dp_to_state(state_dict, sigma=0.01):
    # state_dict: a dict of tensors
    noisy = {}
    for k,v in state_dict.items():
        # convert to float tensor and add noise
        noise = torch.randn_like(v) * sigma
        noisy[k] = (v + noise)
    return noisy
