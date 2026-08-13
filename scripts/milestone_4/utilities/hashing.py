import hashlib


def compute_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def compute_config_hash(yaml_path):
    with open(yaml_path, "r") as f:
        content = f.read()
    return hashlib.sha256(content.encode()).hexdigest()
