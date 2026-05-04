"""

Examples of how cosine similarity is not informative of model probabilities for softmax classifiers

"""


import numpy as np
import matplotlib.pyplot as plt


def three_models_diff_cosines_ex():
    """ Plotting embeddings and unembeddings for three models. 
    Embeddings are identical, so are only plottet once. 
    Unembeddings are the same up to a translation, so the probabilities 
    of the three models are the same, but cosine similarities differ greatly. """

    font_size = 30
    seed = 0
    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((1000, 2))

    original_unembeddings = np.array(
        [
            [1, 0.5],
            [0.5, 1],
            [-1, 0.4],
            [-0.8, -0.8],
            [0.9, -1.2]
        ]
    )
    num_labels = original_unembeddings.shape[0]
    label_0 = original_unembeddings[0]
    label_1 = original_unembeddings[1]
    orig_cosine = np.dot(label_0, label_1)/np.linalg.norm(label_0)**2
    print(orig_cosine) # 0.7999999999999998

    # Plot original unembeddings, cosine between unembeddings for labels 0 and 1 about 0.8
    fig, ax = plt.subplots(1,1, figsize=(5.5, 5))

    for label in range(num_labels):
        ax.scatter(original_unembeddings[label][0], original_unembeddings[label][1],
                   s=180, label=label)

    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.grid(visible=True, linestyle='dashed')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=font_size-10)
    ax.set_title('b)', size= font_size)

    fig.tight_layout()

    ax.legend(fontsize= font_size-10)
    # fig.show()

    dot_products = np.matmul(embeddings, np.transpose(original_unembeddings))
    emb_labels = np.argmax(dot_products, axis=1)


    # plot embeddings coloured by label
    fig, ax = plt.subplots(1,1, figsize=(5.5, 5))
    for label in range(num_labels):
        ax.scatter(
            embeddings[emb_labels == label][:, 0], embeddings[emb_labels == label][:, 1],
            alpha = 0.9, linewidths= 0, s=20, label=label)

    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-3.3, 3.3)
    ax.grid(visible=True, linestyle='dashed')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=font_size-10)
    ax.set_title('a)', size= font_size)

    fig.tight_layout()

    # fig.show()


    # Plot unembeddings of model where unembeddings for labels 0 and 1 have cosine -1
    # By lemma 2.2, we choose v = -a - 0.5(b-a)
    v = -label_0 - 0.5*(label_1 - label_0)
    a_plus_v = label_0 + v
    b_plus_v = label_1 + v
    v_minus_1_cosine = np.dot(a_plus_v, b_plus_v)/(
        np.linalg.norm(a_plus_v)*np.linalg.norm(b_plus_v))
    print(v_minus_1_cosine) # -0.9999999999999998

    v_minus_1_unembeddings = original_unembeddings + v

    fig, ax = plt.subplots(1,1, figsize=(5.5, 5))

    for label in range(num_labels):
        ax.scatter(v_minus_1_unembeddings[label][0], v_minus_1_unembeddings[label][1],
                   s=180, label=label)

    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.grid(visible=True, linestyle='dashed')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=font_size-10)
    ax.set_title('c)', size= font_size)

    fig.tight_layout()

    ax.legend(fontsize= font_size-10)
    # fig.show()

    dot_products_model_2 = np.matmul(embeddings, np.transpose(v_minus_1_unembeddings))
    emb_labels_model_2 = np.argmax(dot_products_model_2, axis=1)
    print(f'All labels classified the same for model 1 and 2: {(emb_labels == emb_labels_model_2).all()}')



    # Plot unembeddings of model where unembeddings for labels 0 and 1 have cosine -1
    # By lemma 2.3, we choose v = -a + 0.5(b-a)
    v = -label_0 + 0.5*(label_1 - label_0)
    a_plus_v = label_0 + v
    b_plus_v = label_1 + v
    v_1_cosine = np.dot(a_plus_v, b_plus_v)/(
        np.linalg.norm(a_plus_v)*np.linalg.norm(b_plus_v))
    print(v_1_cosine) # 1.0

    v_1_unembeddings = original_unembeddings + v

    fig, ax = plt.subplots(1,1, figsize=(5.5, 5))

    for label in range(num_labels):
        ax.scatter(v_1_unembeddings[label][0], v_1_unembeddings[label][1],
                   s=180, label=label)

    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.grid(visible=True, linestyle='dashed')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=font_size-10)
    ax.set_title('d)', size= font_size)

    fig.tight_layout()

    ax.legend(fontsize= font_size-10)
    # fig.show()

    dot_products_model_3 = np.matmul(embeddings, np.transpose(v_1_unembeddings))
    emb_labels_model_3 = np.argmax(dot_products_model_3, axis=1)
    print(
        f'All labels classified the same for model 1 and 3: {(emb_labels == emb_labels_model_3).all()}')


def centered_unembeddings_diff_cosines_ex():
    """
    Plotting an example of centered embeddings where cosine similarity between unembeddings 
    still does not tell us which labels have high probability at the same time.
    """

    font_size = 30

    original_unembeddings = np.array(
        [
            [1.4, -1],
            [1.4, 1],
            #[0.5, 1],
            #[-0.7, 0.7],
            [-0.9, 0.3],
            [-1, 0],
            [-0.9, -0.3],
            #[-0.7, -0.7],
            #[0.5, -1]
        ]
    )
    num_labels = original_unembeddings.shape[0]
    # These unembeddings are centered, since they sum to the zero vector
    print(f'unembedding sum vector: {np.sum(original_unembeddings, axis = 0)}')

    label_0 = original_unembeddings[0]
    label_1 = original_unembeddings[1]
    orig_cosine = np.dot(label_0, label_1)/np.linalg.norm(label_0)**2
    print(f'Cosine between unembeddings for labels 0 and 1: {orig_cosine}') # 0.32432432432432423

    label_2 = original_unembeddings[2]
    cosine_1_2 = np.dot(label_1, label_2)/(np.linalg.norm(label_1)*np.linalg.norm(label_2))
    print(f'Cosine between unembeddings for labels 1 and 2: {cosine_1_2}') # -0.58

    label_4 = original_unembeddings[4]
    cosine_2_4 = np.dot(label_2, label_4)/np.linalg.norm(label_2)**2
    print(f'Cosine between unembeddings for labels 2 and 4: {cosine_2_4}') # 0.8


    fig, ax = plt.subplots(1,1, figsize=(5.5, 5))

    for label in range(num_labels):
        ax.scatter(original_unembeddings[label][0], original_unembeddings[label][1],
                   s=180, label=label)

    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.grid(visible=True, linestyle='dashed')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=font_size-10)
    ax.set_title('Centered Unembeddings', size= font_size)

    fig.tight_layout()

    ax.legend(fontsize= font_size-10)
    fig.show()



def centered_fix_length_unembeddings_diff_cosines_ex():
    """
    Plotting an example of centered embeddings with fixed length where cosine similarity 
    between unembeddings still does not tell us which labels have high probability at the same time.
    """

    font_size = 30
    # np.sqrt(1-(0.95**2)) approx 0.3

    original_unembeddings = np.array(
        [
            [0.95, np.sqrt(1-(0.95**2))],
            [-0.95, np.sqrt(1-(0.95**2))],
            [-1, 0],
            [-0.95, -np.sqrt(1-(0.95**2))],
            [0.95, -np.sqrt(1-(0.95**2))],
            [1, 0]
        ]
    )
    num_labels = original_unembeddings.shape[0]
    # These unembeddings are centered, since they sum to the zero vector
    print(f'unembedding sum vector: {np.sum(original_unembeddings, axis = 0)}')

    # These unembeddings all have length 1
    print(f'Lengths of unembeddings: {np.linalg.norm(original_unembeddings, axis=1)}')

    label_0 = original_unembeddings[0]
    label_1 = original_unembeddings[1]
    orig_cosine = np.dot(label_0, label_1)/np.linalg.norm(label_0)**2
    print(f'Cosine between unembeddings for labels 0 and 1: {orig_cosine}') # -0.805

    label_1 = original_unembeddings[1]
    label_3 = original_unembeddings[3]
    cosine_1_3 = np.dot(label_1, label_3)/np.linalg.norm(label_1)**2
    print(f'Cosine between unembeddings for labels 1 and 3: {cosine_1_3}') # 0.805


    fig, ax = plt.subplots(1,1, figsize=(5.5, 5))

    for label in range(num_labels):
        ax.scatter(original_unembeddings[label][0], original_unembeddings[label][1],
                   s=180, label=label)

    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.grid(visible=True, linestyle='dashed')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=font_size-10)
    ax.set_title('Centered Unemb length 1', size= font_size)

    fig.tight_layout()

    ax.legend(fontsize= font_size-10)
    # fig.show()


if __name__ == "__main__":
    three_models_diff_cosines_ex()
    plt.show()
    centered_unembeddings_diff_cosines_ex()
    plt.show()
    centered_fix_length_unembeddings_diff_cosines_ex()
    plt.show()
