#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Build a reduced AV-HuBERT manifest (test.tsv/test.wrd) containing only the N
samples with the longest transcripts -- the AV-HuBERT-side equivalent of the
Llama-AVSR sweep's `*_micro_50.csv`, for tractable SHAP sweeps.

Usage:
    python make_micro_manifest.py --root /path/to/manifest/dir --split test \
        --num-samples 50 --out-dir /path/to/manifest/dir/micro50
"""

import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='dir containing {split}.tsv, {split}.wrd[, dict.wrd.txt]')
    parser.add_argument('--split', default='test')
    parser.add_argument('--num-samples', type=int, default=50)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--by', choices=['words', 'chars', 'video_frames', 'audio_frames'], default='words',
        help="ranking criterion for 'longest' (words/chars of the .wrd line, "
             "or the nf_video/nf_audio column of the .tsv row)")
    args = parser.parse_args()

    tsv_path = os.path.join(args.root, f'{args.split}.tsv')
    wrd_path = os.path.join(args.root, f'{args.split}.wrd')

    with open(tsv_path) as f:
        tsv_lines = f.read().splitlines()
    header, tsv_rows = tsv_lines[0], tsv_lines[1:]

    with open(wrd_path) as f:
        wrd_rows = f.read().splitlines()

    assert len(tsv_rows) == len(wrd_rows), (
        f"{len(tsv_rows)} tsv data rows vs {len(wrd_rows)} wrd rows -- "
        f"{tsv_path} and {wrd_path} are not aligned (check the .tsv header line)"
    )

    def length_key(i):
        if args.by == 'words':
            return len(wrd_rows[i].split())
        if args.by == 'chars':
            return len(wrd_rows[i])
        items = tsv_rows[i].split('\t')
        return int(items[-2]) if args.by == 'video_frames' else int(items[-1])

    ranked = sorted(range(len(tsv_rows)), key=length_key, reverse=True)
    keep = sorted(ranked[:args.num_samples])  # preserve original file order among the kept rows

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, f'{args.split}.tsv'), 'w') as f:
        f.write(header + '\n')
        for i in keep:
            f.write(tsv_rows[i] + '\n')

    with open(os.path.join(args.out_dir, f'{args.split}.wrd'), 'w') as f:
        for i in keep:
            f.write(wrd_rows[i] + '\n')

    dict_path = os.path.join(args.root, 'dict.wrd.txt')
    if os.path.exists(dict_path):
        shutil.copyfile(dict_path, os.path.join(args.out_dir, 'dict.wrd.txt'))
    else:
        print(f"WARNING: {dict_path} not found -- copy/symlink the correct dict.wrd.txt "
              f"into {args.out_dir} yourself before running infer_s2s_shap.py")

    lengths = sorted((length_key(i) for i in keep), reverse=True)
    print(f"Kept {len(keep)}/{len(tsv_rows)} samples from {args.split}, ranked by={args.by}")
    print(f"Length range kept: {lengths[-1]} .. {lengths[0]}")
    print(f"Wrote {args.out_dir}/{args.split}.tsv and {args.split}.wrd")


if __name__ == '__main__':
    main()
