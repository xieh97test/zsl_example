import data_processing

# %%

# Load acoustic features and labels

example_partition = 'data/fold0.npz'

features, labels = data_processing.load_audio_partition(example_partition)

# %%

# Load semantic embeddings

example_embeddings = 'data/WLE.npz'

semantic_embeddings = data_processing.load_semantic_embedding(example_embeddings)

# Semantic embedding for class index 0
cla_0_embedding = semantic_embeddings[0]

# %%

# Load acoustic features transformed with a trained Baseline model (architecture: [FC-128, tanh, FC-300])

example_partition = 'data/fold0_after.npz'

features, labels = data_processing.load_audio_partition(example_partition)
