from proj_utils.files import merge_ckpts
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("ckpts", type=str)
    args = parser.parse_args()
    ckpt_paths = [
        os.path.join("outputs", args.model_name, f"model.{ckpt}.pt.tar") for ckpt in args.ckpts.split(',')
    ]
    save_path = os.path.join('outputs', args.model_name, f"model.{args.ckpts}.pt.tar")
    merge_ckpts(ckpt_paths, save_path)