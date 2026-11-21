from lib.local_paths import load_paths, write_paths

class EnvSettings:
    def __init__(self):
        self.__dict__.update(load_paths())

def create_default_local_file():
    return EnvSettings()

def create_default_local_file_ITP_test(workspace_dir, data_dir, save_dir):
    return write_paths(workspace_dir, data_dir, save_dir)

def env_settings():
    from lib.test.evaluation.local import local_env_settings
    return local_env_settings()
