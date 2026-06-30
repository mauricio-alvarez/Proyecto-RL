import cv2
import tensorflow as tf
from baselines.common.distributions import make_pdtype

from ma_policy.load_policy import load_policy


def main():
    print("tensorflow:", tf.__version__)
    print("opencv:", cv2.__version__)
    print("baselines.make_pdtype:", callable(make_pdtype))
    print("load_policy:", callable(load_policy))
    print("policy_deps_ok: True")


if __name__ == "__main__":
    main()
