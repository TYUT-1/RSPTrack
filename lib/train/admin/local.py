from lib.local_paths import load_paths

class EnvironmentSettings:
    def __init__(self):
        self.__dict__.update(load_paths())
