from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt, InvalidKey
import os
import base64

class CryptoHandler:
    def __init__(self, passphrase: str):
        self.passphrase = passphrase.encode()  # Convert passphrase to bytes
        self.salt = os.urandom(16)  # Generate a random salt
        self.backend = default_backend()
        self.key = self.derive_key()

    def derive_key(self):
        """Derives a secure key from the passphrase using Scrypt."""
        kdf = Scrypt(
            salt=self.salt,
            length=32,  # AES key length
            n=2**14,
            r=8,
            p=1,
            backend=self.backend
        )
        key = kdf.derive(self.passphrase)  # Derive the key
        return key

    def encrypt(self, plaintext: str) -> str:
        """Encrypts the message."""
        plaintext = plaintext.encode()  # Convert string to bytes
        iv = os.urandom(16)  # Initialization vector for AES
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()

        # Apply padding to the message before encrypting
        padder = PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(plaintext) + padder.finalize()

        # Encrypt the data
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        # Return the encrypted message encoded in base64 along with the IV and salt
        return base64.b64encode(iv + self.salt + ciphertext).decode('utf-8')

    def decrypt(self, encrypted_text: str, passphrase: str) -> str:
        """Decrypts the message using the passphrase."""
        encrypted_data = base64.b64decode(encrypted_text)

        iv = encrypted_data[:16]  # The IV is in the first 16 bytes
        salt = encrypted_data[16:32]  # The salt is in the next 16 bytes
        ciphertext = encrypted_data[32:]  # The rest is the ciphertext

        # Derive the key with the passphrase and salt
        kdf = Scrypt(
            salt=salt,
            length=32,
            n=2**14,
            r=8,
            p=1,
            backend=self.backend
        )
        try:
            key = kdf.derive(passphrase.encode())
        except Exception:
            raise InvalidKey("Invalid passphrase")

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()

        try:
            # Decrypt the data
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()

            # Remove the padding
            unpadder = PKCS7(algorithms.AES.block_size).unpadder()
            data = unpadder.update(padded_data) + unpadder.finalize()
        except ValueError:
            raise InvalidKey("Invalid passphrase or corrupted data")

        return data.decode('utf-8')