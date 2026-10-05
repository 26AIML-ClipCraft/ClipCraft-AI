"""
Utility: extract clip-based textual eot feature and textual token feature into a single lmdb file for MAD dataset
"""
import json
import os

import torch
import numpy as np
import tqdm
import lmdb
import msgpack
import io
from clip_extractor import ClipFeatureExtractor
from torch.utils.data import DataLoader, Dataset
import argparse
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def dumps_msgpack(dump):
    return msgpack.dumps(dump, use_bin_type=True)


def dumps_npz(dump, compress=False):
    with io.BytesIO() as writer:
        if compress:
            np.savez_compressed(writer, **dump, allow_pickle=True)
        else:
            np.savez(writer, **dump, allow_pickle=True)
        return writer.getvalue()


class SingleSentenceDataset(Dataset):
    def __init__(self, input_datalist, block_size=512, debug=False):
        self.max_length = block_size
        self.debug = debug
        self.examples = input_datalist

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        if isinstance(self.examples[index]['query'], str):
            self.examples[index]['query'] = self.process_query(self.examples[index]['query'])
        elif isinstance(self.examples[index]['query'], list):
            query_list = []
            for query in self.examples[index]['query']:
                query_list.append(self.process_query(query))
                self.examples[index]['query'] = query_list
        return self.examples[index]

    def process_query(self,question):
        """Process the query to make it canonical."""
        try:
            return question.strip(".").strip(" ").strip("?") + "."
        except Exception as e:
            print(e)


def pad_collate(data):
    batch = {}
    for k in data[0].keys():
        batch[k] = [d[k] for d in data]
    return batch


def extract_mad_text_feature(args):
    split_list = args.splits.split(',')
    total_data = []
    for split in split_list:
        filename = os.path.join(args.chapters_dir, f"chapters_vmr_{split}.jsonl")
        if 'jsonl' in filename:
            with open(filename) as f:
                data = [json.loads(line) for line in f]
        else:
            data = json.load(open(filename))
        total_data.extend(data)
    if args.ids:
        # ClipCraft: encode only the queries of the evaluated/trained videos. val/test rows carry a chapter-index prefix
        # in 'vid' ("<index><11-char id>"), the YouTube id is always the last 11 characters.
        keep = set(json.load(open(args.ids)))
        total_data = [d for d in total_data if d['vid'][-11:] in keep]
    print(len(total_data))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Build models...")
    clip_model_name_or_path = args.clip_path
    feature_extractor = ClipFeatureExtractor(
        framerate=30, size=224, centercrop=True,
        model_name_or_path=clip_model_name_or_path, device=device
    )

    dataset = SingleSentenceDataset(input_datalist=total_data)

    eval_dataloader = DataLoader(dataset, batch_size=60, collate_fn=pad_collate)

    feature_save_path = args.feature_output_path
    os.makedirs(feature_save_path, exist_ok=True)
    text_output_env = lmdb.open(feature_save_path, map_size=1099511627776, lock=False)

    for i, batch in enumerate(tqdm.tqdm(eval_dataloader, desc="Evaluating", total=len(eval_dataloader))):
        query_id_list = [str(qid) for qid in batch["qid"]]
        query_list = batch["query"]
        if args.key_style == 'vid_idx' or isinstance(query_list[0], list):
            new_query_id_list = []
            new_query_list = []
            for i in range(len(query_list)):
                for j in range(len(query_list[i])):
                    new_query_id_list.append(batch['vid'][i]+'_'+ str(j))
                    new_query_list.append(query_list[i][j])
            query_id_list = new_query_id_list
            query_list = new_query_list

        token_features, text_eot_features = feature_extractor.encode_text(query_list)

        # for i in range(len(query_list)):
        #     if i == 0:
        #         token_feat = np.array(token_features[i].detach().cpu()).astype(np.float32)
        #         eot_feat = np.array(text_eot_features[i].detach().cpu()).astype(np.float32)
        #         print("query: ", query_list[i])
        #         #print("query tokenize 0: ", _tokenizer.bpe(query_list[i]))
        #         encode_text = _tokenizer.encode(query_list[i])
        #         print("query tokenize 1: ", encode_text)
        #         print("query tokenize idx: ", clip.tokenize(query_list[i]))
        #         print("decoder query: ", _tokenizer.decode(encode_text))
        #         print("token_feat: ", token_feat.shape)
        #         print("text_eot_features: ", eot_feat.shape)

        with text_output_env.begin(write=True) as text_output_txn:
            for i in range(len(query_list)):
                q_feat = np.array(text_eot_features[i].detach().cpu()).astype(np.float32)
                token_feat = np.array(token_features[i].detach().cpu()).astype(np.float32)
                features_dict = {"cls_features": q_feat, "token_features": token_feat}
                feature_dump = dumps_npz(features_dict, compress=True)
                text_output_txn.put(key=query_id_list[i].encode(), value=feature_dump)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_output_path", required=True, help="output LMDB dir")
    # ClipCraft: paths were hard-coded to /data/chapters; make them arguments.
    parser.add_argument("--chapters_dir", default="/data/chapters", help="dir with chapters_vmr_<split>.jsonl")
    parser.add_argument("--splits", default="train", help="comma list, e.g. 'train' or 'test' or 'val'")
    parser.add_argument("--clip_path", default="ViT-L/14", help="CLIP name or ViT-L-14.pt path")
    parser.add_argument("--key_style", choices=["vid_idx", "qid"], default="vid_idx",
                        help="LMDB key: '<vid>_<j>' for train annotations (chapters_to_activitynet.py) or the "
                             "record's qid for test/val annotations (chapters_test_to_activitynet.py)")
    parser.add_argument("--ids", default=None, help="JSON list of YouTube ids to keep (e.g. subsets/train_sub.json); "
                                                    "default: every query of the split")
    args = parser.parse_args()
    extract_mad_text_feature(args)
