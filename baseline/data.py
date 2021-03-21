import os

import numpy as np
import sklearn
import torch
from torch.utils.data import TensorDataset

import params


def load_audio_partition(partition_csv):
    """
    :param partition_csv: csv file including audio data partition list
    :return:
        features: list of acoustic features
        labels: list of label indices, single-label / multi-label
    """
    relative_path = partition_csv

    # Convert to absolute path
    partition_csv = os.path.join(params.base_dir, partition_csv)
    partition_dir = os.path.dirname(partition_csv)

    # Load partitions
    features, labels = [], []
    with open(file=partition_csv, mode='r', encoding='UTF-8') as _csv_file:
        for _line in _csv_file.readlines():
            _partition = os.path.join(partition_dir, _line.strip())
            try:
                _data = np.load(_partition, allow_pickle=True)
                for _f, _l in zip(_data['features'], _data['labels']):
                    # Pad acoustic features
                    _tmp = np.zeros((params.num_acoustic, params.acoustic_features))
                    _dim1 = min(_tmp.shape[0], _f.shape[0])
                    _dim2 = min(_tmp.shape[1], _f.shape[1])
                    _tmp[:_dim1, :_dim2] = _f[:_dim1, :_dim2]

                    features.append(_tmp)
                    labels.append(_l)
            except Exception as e:
                print(e)

    print(relative_path, 'features:', len(features), 'labels:', len(labels))

    return features, labels


def load_semantic_embedding(embedding_npz):
    """
    :param embedding_npz: npz file including class semantic embeddings
    """
    # Convert to absolute path
    embedding_path = os.path.join(params.base_dir, embedding_npz)

    embeddings = np.load(embedding_path, allow_pickle=True)[0]
    embeddings = {int(_idx): embeddings[_idx] for _idx in embeddings}

    print(embedding_npz, 'LOADED')

    return embeddings


def preprocess(x, y, y_embeddings, class_weight=None):
    """
    :param x: list of acoustic features
    :param y: list of label indices
    :param y_embeddings: dict of label indices (keys) with semantic embeddings (values)
    :return:
    """
    y = np.array(y).flatten()
    y_unique = np.unique(y)

    encoder, embeddings = {}, []
    for _k, _v in enumerate(y_unique):
        encoder[_v] = _k
        embeddings.append(y_embeddings[_v])

    embeddings = np.array(embeddings, dtype=np.float)  # (num_classes, dim_semantic_features)

    y = np.array([encoder[_v] for _v in y])  # (num_samples,)

    y_weights = sklearn.utils.compute_sample_weight(class_weight=class_weight, y=y)  # (num_samples,)

    x = np.array(x)  # (num_samples, num_segments, dim_acoustic_features)

    return x, y, y_weights, embeddings, encoder


def load_data(partition_csv, embedding_npz):
    """
    :param partition_csv: csv file including audio data partition list
    :param embedding_npz: npz file including class semantic embeddings
    """
    features, labels = load_audio_partition(partition_csv)

    embeddings = load_semantic_embedding(embedding_npz)

    features, labels, weights, embeddings, encoder = preprocess(features, labels, embeddings)

    # Convert to tensors
    features, labels, weights, embeddings = map(torch.tensor, (features, labels, weights, embeddings))

    features = features.to(torch.float32)
    labels = labels.to(torch.int64)
    weights = weights.to(torch.float32)
    embeddings = embeddings.to(torch.float32)

    acoustic_features = features.size(-1)
    semantic_features = embeddings.size(-1)

    dataset = TensorDataset(features, labels, weights)

    return acoustic_features, semantic_features, dataset, embeddings, encoder
