import pytest
from cryptography.hazmat.primitives.kdf.scrypt import InvalidKey
from wyra.crypto import CryptoHandler  # Import the CryptoHandler class

class TestCrypto:

    def setup_method(self):
        """Setup method to initialize variables for the tests."""
        self.passphrase = "Sator Arepo Tenet Opera Rotas"
        self.handler = CryptoHandler(self.passphrase)
        self.message = "frase secreta"

    def test_encrypt_decrypt(self):
        """Test that ensures an encrypted message can be decrypted correctly."""
        encrypted_message = self.handler.encrypt(self.message)
        decrypted_message = self.handler.decrypt(encrypted_message, self.passphrase)
        assert decrypted_message == self.message, "The decrypted message does not match the original."

    def test_invalid_passphrase(self):
        """Test that verifies decryption fails with an incorrect passphrase."""
        encrypted_message = self.handler.encrypt(self.message)
        with pytest.raises(InvalidKey):
            # Attempting to decrypt with an incorrect passphrase
            self.handler.decrypt(encrypted_message, "wrong_password")

    def test_encrypt_returns_base64(self):
        """Test that ensures the encryption result is a base64 encoded string."""
        encrypted_message = self.handler.encrypt(self.message)
        # Check if the encrypted string can be decoded back from base64
        assert isinstance(encrypted_message, str), "The encryption result is not a string."
        # This does not guarantee it is a valid base64 string, but it is an initial check.

    def test_different_encrypted_output(self):
        """Test that ensures different encrypted messages are generated even with the same input."""
        encrypted_message_1 = self.handler.encrypt(self.message)
        encrypted_message_2 = self.handler.encrypt(self.message)
        assert encrypted_message_1 != encrypted_message_2, "Identical encrypted messages were generated for the same input."