from lib.local_paths import write_paths

def create_default_local_file():
    from lib.train.admin.local import EnvironmentSettings
    return EnvironmentSettings()

def create_default_local_file_ITP_train(workspace_dir, data_dir):
    return write_paths(workspace_dir, data_dir)

def env_settings():
    from lib.train.admin.local import EnvironmentSettings
    return EnvironmentSettings()
