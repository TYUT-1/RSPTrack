import os
import lmdb
import numpy as np
import cv2
import json

LMDB_ENVS = dict()
LMDB_HANDLES = dict()
LMDB_FILELISTS = dict()


def _cache_key(name):
    return os.getpid(), os.path.abspath(name)


def _close_cached_env(key):
    LMDB_HANDLES.pop(key, None)
    env = LMDB_ENVS.pop(key, None)
    if env is not None:
        try:
            env.close()
        except Exception:
            pass


def get_lmdb_handle(name, renew=False):
    global LMDB_HANDLES, LMDB_FILELISTS
    key = _cache_key(name)
    if renew:
        _close_cached_env(key)

    item = LMDB_HANDLES.get(key, None)
    if item is None:
        env = LMDB_ENVS.get(key, None)
        if env is None:
            env = lmdb.open(name, readonly=True, lock=False, readahead=False, meminit=False, max_readers=2048)
            LMDB_ENVS[key] = env
        item = env.begin(write=False)
        LMDB_HANDLES[key] = item

    return item


def _lmdb_get(lmdb_fname, key_name):
    key = key_name.encode()
    for renew in (False, True):
        handle = get_lmdb_handle(lmdb_fname, renew=renew)
        try:
            return handle.get(key)
        except lmdb.Error:
            if renew:
                raise


def decode_img(lmdb_fname, key_name):
    binfile = _lmdb_get(lmdb_fname, key_name)
    if binfile is None:
        print("Illegal data detected. %s %s" % (lmdb_fname, key_name))
    s = np.frombuffer(binfile, np.uint8)
    x = cv2.cvtColor(cv2.imdecode(s, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return x


def decode_str(lmdb_fname, key_name):
    binfile = _lmdb_get(lmdb_fname, key_name)
    string = binfile.decode()
    return string


def decode_json(lmdb_fname, key_name):
    return json.loads(decode_str(lmdb_fname, key_name))


if __name__ == "__main__":
    lmdb_fname = "/data/sda/v-yanbi/iccv21/LittleBoy_clean/data/got10k_lmdb"
    '''Decode image'''
    # key_name = "test/GOT-10k_Test_000001/00000001.jpg"
    # img = decode_img(lmdb_fname, key_name)
    # cv2.imwrite("001.jpg", img)
    '''Decode str'''
    # key_name = "test/list.txt"
    # key_name = "train/GOT-10k_Train_000001/groundtruth.txt"
    key_name = "train/GOT-10k_Train_000001/absence.label"
    str_ = decode_str(lmdb_fname, key_name)
    print(str_)
