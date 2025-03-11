import secrets

# Generate a 32-byte random string (256 bits)
jwt_secret = secrets.token_hex(32)
print(f"Your JWT secret key: {jwt_secret}")
