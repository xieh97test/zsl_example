### Example for zero-shot learning

#### 1. Semantic Embeddings for Sound Classes

- WLE.npz
    - Averaged 300-dimensional word embeddings from a pre-trained Word2Vec.
    - 527 class indices with their corresponding embeddings.

#### 2. Acoustic Embeddings for Audio Samples

- *.npz
    - Averaged 128-dimensional VGGish embeddings of audio samples and their corresponding class indices.

| Class Fold | Total classes | Total samples | Usage          |
| ---------- | ------------- | ------------- | -------------- |
| fold0      | 104           | 23007         | ZSL-Train      |
| fold1      | 104           | 22889         | ZSL-Validation |
| fold2      | 104           | 22762         | ZSL-Test1      |
| fold3      | 104           | 22739         | ZSL-Test2      |
| fold4      | 105           | 21377         | ZSL-Test3      |

- *_after.npz
  - Transformed acoustic embeddings with a trained ZSL model (Baseline: [FC-128, tanh, FC-300]).

#### 3. Baseline Model

upcoming.