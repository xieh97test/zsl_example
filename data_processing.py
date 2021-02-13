import numpy as np


def load_audio_partition(partition_npz):
    """
    :param partition_npz: NPZ file including acoustic features and their corresponding class indices.
    :return:
        features: (num_audio_samples, dim_acoustic_features)
        labels: (num_audio_samples,)
    """

    partition_data = np.load(partition_npz, allow_pickle=True)

    features = partition_data['features']
    labels = partition_data['labels']

    print(partition_npz, features.shape, labels.shape)

    return features, labels


def load_semantic_embedding(embedding_npz):
    """
    :param embedding_npz: NPZ file including class semantic embeddings
    :return: dict of semantic embeddings: {class_index: semantic_embedding}
    """

    semantic_embeddings = np.load(embedding_npz, allow_pickle=True)['semantic_embeddings'][0]

    return semantic_embeddings
