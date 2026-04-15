# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# %%
def center(X: np.ndarray):
    return X - X.mean(axis=0, keepdims=True)


def effective_rank(X: np.ndarray, eps: float = 1e-12):
    """
    participation ratio based effective rank
    """
    # SVD on centered data
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S2 = S ** 2
    num = (S2.sum()) ** 2
    den = (S2 ** 2).sum() + eps
    return num / den


def k_energy(X: np.ndarray, energy: float = 0.9):
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    S2 = S ** 2
    cum_energy = np.cumsum(S2) / S2.sum()
    return int(np.searchsorted(cum_energy, energy) + 1)

# %%
def load_layer_embeddings(base_path: str):
    """
    base_path:
        Data/Representation/Qwen3-VL-4B-Thinking
    return:
        layers: dict[layer_id] -> np.ndarray (N, D)
    """
    layers = {}
    for fname in os.listdir(base_path):
        if fname.startswith("embeds_") and fname.endswith(".npy"):
            layer_id = int(fname.replace("embeds_", "").replace(".npy", ""))
            layers[layer_id] = np.load(os.path.join(base_path, fname))
    return dict(sorted(layers.items()))

# %%
def analyze_rank(
    with_image_path: str,
    without_image_path: str,
    energy: float = 0.9,
):
    with_layers = load_layer_embeddings(with_image_path)
    no_layers   = load_layer_embeddings(without_image_path)

    assert with_layers.keys() == no_layers.keys(), "Layer mismatch!"

    results = {
        "layer": [],
        "eff_rank_img": [],
        "eff_rank_no": [],
        "k90_img": [],
        "k90_no": [],
    }

    for l in with_layers.keys():
        X_img = center(with_layers[l])
        X_no  = center(no_layers[l])

        r_img = effective_rank(X_img)
        r_no  = effective_rank(X_no)

        k_img = k_energy(X_img, energy)
        k_no  = k_energy(X_no, energy)

        results["layer"].append(l)
        results["eff_rank_img"].append(r_img)
        results["eff_rank_no"].append(r_no)
        results["k90_img"].append(k_img)
        results["k90_no"].append(k_no)

        print(
            f"Layer {l:02d} | "
            f"r_eff(img)={r_img:.1f}, r_eff(no)={r_no:.1f} | "
            f"k90(img)={k_img}, k90(no)={k_no}"
        )

    return results

# %%
with_image_path = "Data/Representation/MMMU_Test/Qwen3-VL-4B-Thinking"
without_image_path = "Data/Representation/MMMU_Test-no-image/Qwen3-VL-4B-Thinking-no-image"

# %%
results = analyze_rank(with_image_path, without_image_path)

# %%
with_image_path_2b = "Data/Representation/MMMU_Test/Qwen3-VL-2B-Thinking"
without_image_path_2b = "Data/Representation/MMMU_Test-no-image/Qwen3-VL-2B-Thinking-no-image"

# %%
with_image_path_2b = "Data/Representation/MMMU_Test/Qwen3-VL-2B-Thinking"
without_image_path_2b = "Data/Representation/MMMU_Test-no-image/Qwen3-VL-2B-Thinking-no-image"

results_2b = analyze_rank(with_image_path_2b, without_image_path_2b)

# %%
with_image_path_8b = "Data/Representation/MMMU_Test/Qwen3-VL-8B-Thinking"
without_image_path_8b = "Data/Representation/MMMU_Test-no-image/Qwen3-VL-8B-Thinking-no-image"

# %%
with_image_path_8b = "Data/Representation/MMMU_Test/Qwen3-VL-8B-Thinking"
without_image_path_8b = "Data/Representation/MMMU_Test-no-image/Qwen3-VL-8B-Thinking-no-image"

results_8b = analyze_rank(with_image_path_8b, without_image_path_8b)

# %%
with_image_path_glm = "Data/Representation/MMMU_Test/GLM-4.1V-9B-Thinking"
without_image_path_glm = "Data/Representation/MMMU_Test-no-image/GLM-4.1V-9B-Thinking-no-image"

# %%
with_image_path_glm = "Data/Representation/MMMU_Test/GLM-4.1V-9B-Thinking"
without_image_path_glm = "Data/Representation/MMMU_Test-no-image/GLM-4.1V-9B-Thinking-no-image"

results_glm = analyze_rank(with_image_path_glm, without_image_path_glm)

# %%
def _to_np(results, key):
    return np.array(results[key], dtype=float)

def plot_rank_results(results, skip_degenerate=True, title="(Qwen3-VL-4B-Thinking)"):
    layer = _to_np(results, "layer")
    r_img = _to_np(results, "eff_rank_img")
    r_no  = _to_np(results, "eff_rank_no")
    k_img = _to_np(results, "k90_img")
    k_no  = _to_np(results, "k90_no")

    if skip_degenerate:
        mask = ~((np.isfinite(r_img) & np.isfinite(r_no) & (r_img < 1e-6) & (r_no < 1e-6)))
        layer, r_img, r_no, k_img, k_no = layer[mask], r_img[mask], r_no[mask], k_img[mask], k_no[mask]

    """
    # ---- Figure 1: effective rank ----
    plt.figure()
    plt.plot(layer, r_img, marker="o", label="with image")
    plt.plot(layer, r_no,  marker="o", label="no image")
    plt.xlabel("Layer")
    plt.ylabel("Effective rank")
    plt.title(f"Layer-wise effective rank {title}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    
    # ---- Figure 2: k90 ----
    plt.figure()
    plt.plot(layer, k_img, marker="o", label="w/ image")
    plt.plot(layer, k_no,  marker="o", label="w/o image")
    plt.xlabel("Layer")
    plt.ylabel("90% explained variance dimension")
    plt.title(f"Layer-wise 90% explained variance dimension {title}")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # ---- Figure 3: differences ----
    dr = r_img - r_no
    dk = k_img - k_no

    """
    plt.figure()
    plt.plot(layer, dr, marker="o")
    plt.axhline(0, linewidth=1)
    plt.xlabel("Layer")
    plt.ylabel("Δ effective rank (with - no)")
    plt.title("Difference in effective rank")
    plt.tight_layout()
    plt.show()
    """
    
    plt.figure()
    plt.plot(layer, dk, marker="o")
    plt.axhline(0, linewidth=1, color = "gray", linestyle="--")
    plt.xlabel("Layer")
    plt.ylabel("Δ dimensionality (90% variance)")
    plt.title("Difference in representation dimensionality")
    plt.tight_layout()
    plt.show()


# %%
results = pd.read_csv("Results/results_4b.csv")

# %%
plot_rank_results(results, title="(Qwen3-VL-4B-Thinking)")

# %%
plot_rank_results(results)

# %%
results_2b = pd.read_csv("Results/results_2b.csv")

# %%
plot_rank_results(results_2b, title="(Qwen3-VL-2B-Thinking)")

# %%
results_8b = pd.read_csv("Results/results_8b.csv")

# %%
plot_rank_results(results_8b, title="(Qwen3-VL-8B-Thinking)")

# %%
results_glm = pd.read_csv("Results/results_glm.csv")

# %%
plot_rank_results(results_glm, title="(GLM-4.1V-9B-Thinking)")

# %%
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np


def probe_layer_mlp(X_img, X_no):

    # build dataset
    X = np.concatenate([X_img, X_no], axis=0)

    y = np.concatenate([
        np.ones(len(X_img)),
        np.zeros(len(X_no))
    ])

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=0
    )

    # MLP probe (similar to paper repo)
    clf = MLPClassifier(
        hidden_layer_sizes=(256,),
        max_iter=200,
        batch_size=64
    )

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)

    return acc

# %%
def analyze_probe(with_image_path, without_image_path):

    with_layers = load_layer_embeddings(with_image_path)
    no_layers = load_layer_embeddings(without_image_path)

    accs = []

    for l in with_layers.keys():

        X_img = with_layers[l]
        X_no = no_layers[l]

        acc = probe_layer_mlp(X_img, X_no)

        accs.append(acc)

        print(f"Layer {l:02d} probe accuracy: {acc:.3f}")

    return accs

# %%
def plot_probe(accs):

    layers = list(range(len(accs)))

    plt.figure()

    plt.plot(layers, accs, marker="o")

    plt.xlabel("Layer")
    plt.ylabel("Classification accuracy")
    plt.title("Visual context separability")

    plt.ylim(0.5, 1)

    plt.show()

# %%
analyze_probe(with_image_path, without_image_path)

# %%
from scipy.linalg import subspace_angles

def compute_subspace_angle(X_img, X_no, k=5):

    X_img = X_img - X_img.mean(0)
    X_no  = X_no  - X_no.mean(0)

    _, _, Vt_img = np.linalg.svd(X_img, full_matrices=False)
    _, _, Vt_no  = np.linalg.svd(X_no,  full_matrices=False)

    A = Vt_img[:k].T
    B = Vt_no[:k].T

    angles = subspace_angles(A, B)

    return float(np.degrees(angles[0]))

# %%
def mean_cosine(X_img, X_no):

    mu_img = X_img.mean(axis=0)
    mu_no  = X_no.mean(axis=0)

    cos = np.dot(mu_img, mu_no) / (
        np.linalg.norm(mu_img) * np.linalg.norm(mu_no)
    )

    return cos

# %%
def analyze_subspace(with_image_path, without_image_path):

    with_layers = load_layer_embeddings(with_image_path)
    no_layers   = load_layer_embeddings(without_image_path)

    layers = []
    angles = []
    cosines = []

    for l in with_layers.keys():

        X_img = with_layers[l]
        X_no  = no_layers[l]

        angle = compute_subspace_angle(X_img, X_no)
        cosv  = mean_cosine(X_img, X_no)

        layers.append(l)
        angles.append(angle)
        cosines.append(cosv)

        print(
            f"Layer {l:02d} | "
            f"subspace angle = {angle:.2f}° | "
            f"mean cosine = {cosv:.3f}"
        )

    return layers, angles, cosines

# %%
def plot_subspace(layers, angles, cosines):

    plt.figure()
    plt.plot(layers, angles, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Subspace angle (deg)")
    plt.title("Image vs No-image Subspace Difference")
    plt.show()

    plt.figure()
    plt.plot(layers, cosines, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Cos(mean_img, mean_no)")
    plt.title("Mean Representation Alignment")
    plt.show()

# %%
layers, angles, cosines = analyze_subspace(
    with_image_path,
    without_image_path
)

plot_subspace(layers, angles, cosines)

# %%
def variance_explained_by_noimage_pc(X_img, X_no, max_k=20):

    # center
    X_img = X_img - X_img.mean(0)
    X_no  = X_no  - X_no.mean(0)

    # PCA on no-image
    _, _, Vt = np.linalg.svd(X_no, full_matrices=False)

    total_var = np.sum(X_img**2)

    ratios = []

    for k in range(1, max_k + 1):

        V = Vt[:k].T

        proj = X_img @ V @ V.T

        var = np.sum(proj**2)

        ratios.append(var / total_var)

    return ratios

# %%
def variance_explained_by_image_pc(X_img, max_k=20):

    # center
    X_img = X_img - X_img.mean(0)

    # PCA on no-image
    _, _, Vt = np.linalg.svd(X_img, full_matrices=False)

    total_var = np.sum(X_img**2)

    ratios = []

    for k in range(1, max_k + 1):

        V = Vt[:k].T

        proj = X_img @ V @ V.T

        var = np.sum(proj**2)

        ratios.append(var / total_var)

    return ratios

# %%
layer = 20

with_layers = load_layer_embeddings(with_image_path)
no_layers   = load_layer_embeddings(without_image_path)


X_img = with_layers[layer]
X_no  = no_layers[layer]

ratios = variance_explained_by_image_pc(X_img)

# %%
layer = 20

with_layers = load_layer_embeddings(with_image_path)
no_layers   = load_layer_embeddings(without_image_path)


X_img = with_layers[layer]
X_no  = no_layers[layer]

ratios_2 = variance_explained_by_noimage_pc(X_img, X_no)

print("k=1:", ratios[0])
print("k=3:", ratios[2])
print("k=5:", ratios[4])
print("k=10:", ratios[9])

# %%
def layerwise_variance_analysis(
    hidden_states_img,
    hidden_states_no,
    max_k=20
):
    results_noimg_pc = []
    #results_img_pc = []

    num_layers = len(hidden_states_img)

    for layer in range(num_layers):

        X_img = hidden_states_img[layer]
        X_no  = hidden_states_no[layer]

        ratios_noimg_pc = variance_explained_by_noimage_pc(
            X_img,
            X_no,
            max_k=max_k
        )

        """
        ratios_img_pc = variance_explained_by_image_pc(
            X_img,
            max_k=max_k
        )
        """

        results_noimg_pc.append(ratios_noimg_pc)
        # results_img_pc.append(ratios_img_pc)

    #return results_noimg_pc, results_img_pc
    return results_noimg_pc

# %%
with_layers = load_layer_embeddings(with_image_path)
no_layers   = load_layer_embeddings(without_image_path)

#results_noimg_pc, results_img_pc = layerwise_variance_analysis(
results_noimg_pc = layerwise_variance_analysis(
    with_layers,
    no_layers,
    max_k=20
)

# %%
results_noimg_pc

# %%
layer = 20

print("\nLayer", layer)

print("\nno-image PCA explains image variance")
print("k=1 :", results_noimg_pc[0][layer][0])
print("k=3 :", results_noimg_pc[0][layer][2])
print("k=5 :", results_noimg_pc[0][layer][4])
print("k=10:", results_noimg_pc[0][layer][9])
"""
print("\nimage PCA explains image variance")
print("k=1 :", results_img_pc[layer][0])
print("k=3 :", results_img_pc[layer][2])
print("k=5 :", results_img_pc[layer][4])
print("k=10:", results_img_pc[layer][9])
"""



# %%
results_no_img_pc_df = pd.DataFrame(results_noimg_pc[0])

# %%
results_noimg_pc = pd.read_csv("Results/results_no_img_pc_df_4b_200.csv").values.tolist()

# %%
############################################
## Plot layer-wise curve
############################################

layers = range(len(results_noimg_pc))

plt.figure(figsize=(8,5))

plt.plot(
    layers,
    [results_noimg_pc[l][0] for l in layers],
    label="k=1"
)

plt.plot(
    layers,
    [results_noimg_pc[l][2] for l in layers],
    label="k=3"
)

plt.plot(
    layers,
    [results_noimg_pc[l][4] for l in layers],
    label="k=5"
)

plt.plot(
    layers,
    [results_noimg_pc[l][9] for l in layers],
    label="k=10"
)

plt.plot(
    layers,
    [results_noimg_pc[l][24] for l in layers],
    label="k=25"
)

plt.xlabel("Layer")
plt.ylabel("Proportion of variance explained")
plt.title("Layer-wise variance explained by image-absent PCs (Qwen3-VL-4B-Thinking)")

plt.legend()
plt.show()

# %%
with_layers_2b = load_layer_embeddings(with_image_path_2b)
no_layers_2b   = load_layer_embeddings(without_image_path_2b)

#results_noimg_pc, results_img_pc = layerwise_variance_analysis(
results_noimg_pc_2b = layerwise_variance_analysis(
    with_layers_2b,
    no_layers_2b,
    max_k=200
)

# %%
layer = 20

print("\nLayer", layer)

print("\nno-image PCA explains image variance")
print("k=1 :", results_noimg_pc_2b[layer][0])
print("k=3 :", results_noimg_pc_2b[layer][2])
print("k=5 :", results_noimg_pc_2b[layer][4])
print("k=10:", results_noimg_pc_2b[layer][9])
"""
print("\nimage PCA explains image variance")
print("k=1 :", results_img_pc[layer][0])
print("k=3 :", results_img_pc[layer][2])
print("k=5 :", results_img_pc[layer][4])
print("k=10:", results_img_pc[layer][9])
"""



# %%
results_no_img_pc_df_2b = pd.DataFrame(results_noimg_pc_2b)

# %%
results_noimg_pc_2b = pd.read_csv("Results/results_no_img_pc_df_2b_200.csv").values.tolist()

# %%
############################################
## Plot layer-wise curve
############################################

layers_2b = range(len(results_noimg_pc_2b))

plt.figure(figsize=(8,5))

plt.plot(
    layers_2b,
    [results_noimg_pc_2b[l][0] for l in layers_2b],
    label="k=1"
)

plt.plot(
    layers_2b,
    [results_noimg_pc_2b[l][2] for l in layers_2b],
    label="k=3"
)

plt.plot(
    layers_2b,
    [results_noimg_pc_2b[l][4] for l in layers_2b],
    label="k=5"
)

plt.plot(
    layers_2b,
    [results_noimg_pc_2b[l][9] for l in layers_2b],
    label="k=10"
)

plt.plot(
    layers_2b,
    [results_noimg_pc_2b[l][19] for l in layers_2b],
    label="k=20"
)

plt.xlabel("Layer")
plt.ylabel("Proportion of variance explained")
plt.title("Layer-wise variance explained by image-absent PCs (Qwen3-VL-2B-Thinking)")

plt.legend()
plt.show()

# %%
with_layers_8b = load_layer_embeddings(with_image_path_8b)
no_layers_8b   = load_layer_embeddings(without_image_path_8b)

#results_noimg_pc, results_img_pc = layerwise_variance_analysis(
results_noimg_pc_8b = layerwise_variance_analysis(
    with_layers_8b,
    no_layers_8b,
    max_k=200
)

# %%
layer = 20

print("\nLayer", layer)

print("\nno-image PCA explains image variance")
print("k=1 :", results_noimg_pc_8b[layer][0])
print("k=3 :", results_noimg_pc_8b[layer][2])
print("k=5 :", results_noimg_pc_8b[layer][4])
print("k=10:", results_noimg_pc_8b[layer][9])
"""
print("\nimage PCA explains image variance")
print("k=1 :", results_img_pc[layer][0])
print("k=3 :", results_img_pc[layer][2])
print("k=5 :", results_img_pc[layer][4])
print("k=10:", results_img_pc[layer][9])
"""



# %%
results_no_img_pc_df_8b = pd.DataFrame(results_noimg_pc_8b)

# %%
results_noimg_pc_8b = pd.read_csv("Results/results_no_img_pc_df_8b_200.csv").values.tolist()

# %%
############################################
## Plot layer-wise curve
############################################

layers_8b = range(len(results_noimg_pc_8b))

plt.figure(figsize=(8,5))

plt.plot(
    layers_8b,
    [results_noimg_pc_8b[l][0] for l in layers_8b],
    label="k=1"
)

plt.plot(
    layers_8b,
    [results_noimg_pc_8b[l][2] for l in layers_8b],
    label="k=3"
)

plt.plot(
    layers_8b,
    [results_noimg_pc_8b[l][9] for l in layers_8b],
    label="k=10"
)

plt.plot(
    layers_8b,
    [results_noimg_pc_8b[l][19] for l in layers_8b],
    label="k=20"
)

plt.plot(
    layers_8b,
    [results_noimg_pc_8b[l][39] for l in layers_8b],
    label="k=40"
)

plt.xlabel("Layer")
plt.ylabel("Proportion of variance explained")
plt.title("Layer-wise variance explained by image-absent PCs (Qwen3-VL-8B-Thinking)")

plt.legend()
plt.show()

# %%
with_layers_glm = load_layer_embeddings(with_image_path_glm)
no_layers_glm   = load_layer_embeddings(without_image_path_glm)

#results_noimg_pc, results_img_pc = layerwise_variance_analysis(
results_noimg_pc_glm = layerwise_variance_analysis(
    with_layers_glm,
    no_layers_glm,
    max_k=500
)

# %%
layer = 20

print("\nLayer", layer)

print("\nno-image PCA explains image variance")
print("k=1 :", results_noimg_pc_glm[layer][0])
print("k=3 :", results_noimg_pc_glm[layer][2])
print("k=5 :", results_noimg_pc_glm[layer][4])
print("k=10:", results_noimg_pc_glm[layer][9])
"""
print("\nimage PCA explains image variance")
print("k=1 :", results_img_pc[layer][0])
print("k=3 :", results_img_pc[layer][2])
print("k=5 :", results_img_pc[layer][4])
print("k=10:", results_img_pc[layer][9])
"""



# %%
results_no_img_pc_df_glm = pd.DataFrame(results_noimg_pc_glm)

# %%
results_noimg_pc_glm = pd.read_csv("Results/results_no_img_pc_df_glm_500.csv").values.tolist()

# %%
############################################
## Plot layer-wise curve
############################################

layers_glm = range(len(results_noimg_pc_glm))

plt.figure(figsize=(8,5))

plt.plot(
    layers_glm,
    [results_noimg_pc_glm[l][0] for l in layers_glm],
    label="k=1"
)

plt.plot(
    layers_glm,
    [results_noimg_pc_glm[l][2] for l in layers_glm],
    label="k=3"
)

plt.plot(
    layers_glm,
    [results_noimg_pc_glm[l][9] for l in layers_glm],
    label="k=10"
)

plt.plot(
    layers_glm,
    [results_noimg_pc_glm[l][19] for l in layers_glm],
    label="k=20"
)

plt.plot(
    layers_glm,
    [results_noimg_pc_glm[l][39] for l in layers_glm],
    label="k=40"
)

plt.xlabel("Layer")
plt.ylabel("Proportion of variance explained")
plt.title("Layer-wise variance explained by image-absent PCs (GLM-4.1V-9B-Thinking)")

plt.legend()
plt.show()

# %%
def subspace_overlap(X_img, X_no, k):

    X_img = X_img - X_img.mean(0)
    X_no  = X_no  - X_no.mean(0)

    _, _, Vt_img = np.linalg.svd(X_img, full_matrices=False)
    _, _, Vt_no  = np.linalg.svd(X_no,  full_matrices=False)

    V = Vt_img[:k].T
    U = Vt_no[:k].T

    s = np.linalg.svd(V.T @ U, compute_uv=False)

    return s  # cos(theta_i)

# %%
def layerwise_subspace_overlap(
    hidden_states_img,
    hidden_states_no,
    k=20
):

    results = []

    num_layers = len(hidden_states_img)

    for layer in range(num_layers):

        X_img = hidden_states_img[layer]
        X_no  = hidden_states_no[layer]

        s = subspace_overlap(
            X_img,
            X_no,
            k=k
        )

        results.append(s)

    return results

# %%
with_layers_4b = load_layer_embeddings(with_image_path)
no_layers_4b = load_layer_embeddings(without_image_path)

results_overlap_4b = layerwise_subspace_overlap(
    with_layers_4b,
    no_layers_4b,
    k=180
)

# %%
with_layers_glm = load_layer_embeddings(with_image_path_glm)
no_layers_glm   = load_layer_embeddings(without_image_path_glm)

results_overlap_glm = layerwise_subspace_overlap(
    with_layers_glm,
    no_layers_glm,
    k=500
)

# %%
layer = 20

print("Layer", layer)

print("mean overlap:",
      results_overlap_glm[layer].mean())

print("top 5 cos(theta):")
print(results_overlap_glm[layer][:5])

# %%
results_overlap_glm = pd.read_csv("Results/results_overlap_glm_500.csv").iloc[1:41, :].values.tolist()

# %%
overlap_mean_glm = [
    # np.mean(s)
    np.mean(np.array(s) ** 2)
    for s in results_overlap_glm
]

# %%
plt.plot(overlap_mean_glm)

# baselines
baseline_rho098 = 0.8932953353547871
baseline_rho095 = 0.7949711497711195
baseline_rho09 = 0.6760455622126994

plt.axhline(baseline_rho098, linestyle="--", label="ρ=0.98 baseline", color="red")
plt.axhline(baseline_rho095, linestyle="--", label="ρ=0.95 baseline", color="green")
plt.axhline(baseline_rho09, linestyle="--", label="ρ=0.9 baseline", color="blue")


plt.xlabel("Layer")
plt.ylabel("Mean cos² principal angle")

plt.title(
"Layer-wise mean overlap (GLM-4.1V-9B-Thinking)"
)

plt.legend()

plt.show()

# %%
delta_overlap_glm = np.diff(overlap_mean_glm)

plt.plot(delta_overlap_glm)

plt.axhline(0, color="gray", linestyle="--")

plt.xlabel("Layer")

plt.ylabel("Δ subspace overlap")

plt.title(
"Layer-wise change in mean overlap (GLM-4.1V-9B-Thinking)"
)

plt.show()

# %%
with_layers_8b = load_layer_embeddings(with_image_path_8b)
no_layers_8b   = load_layer_embeddings(without_image_path_8b)

results_overlap_8b = layerwise_subspace_overlap(
    with_layers_8b,
    no_layers_8b,
    k=160
)

# %%
results_overlap_8b = pd.read_csv("Results/results_overlap_8b_160.csv").iloc[1:37, :].values.tolist()

# %%
overlap_mean_8b = [
    # np.mean(s)
    np.mean(np.array(s) ** 2)
    for s in results_overlap_8b
]

# %%
plt.plot(overlap_mean_8b)

# baselines
baseline_rho09 = 0.5514632281864306
baseline_rho085 = 0.43874334818315236
baseline_rho08 = 0.354059947734277

plt.axhline(baseline_rho09, linestyle="--", label="ρ=0.95 baseline", color="red")
plt.axhline(baseline_rho085, linestyle="--", label="ρ=0.9 baseline", color="green")
plt.axhline(baseline_rho08, linestyle="--", label="ρ=0.8 baseline", color="blue")


plt.xlabel("Layer")
plt.ylabel("Mean cos² principal angle")

plt.title(
"Layer-wise mean overlap (Qwen3-VL-8B-Thinking)"
)

plt.legend()

plt.show()

# %%
delta_overlap_8b = np.diff(overlap_mean_8b)

plt.plot(delta_overlap_8b)

plt.axhline(0, color="gray", linestyle="--")

plt.xlabel("Layer")

plt.ylabel("Δ subspace overlap")

plt.title(
"Layer-wise change in mean overlap (Qwen3-VL-8B-Thinking)"
)

plt.show()

# %%
with_layers_2b = load_layer_embeddings(with_image_path_2b)
no_layers_2b   = load_layer_embeddings(without_image_path_2b)

results_overlap_2b = layerwise_subspace_overlap(
    with_layers_2b,
    no_layers_2b,
    k=100
)

# %%
results_overlap_2b = pd.read_csv("Results/results_overlap_2b_100.csv").iloc[1:29, :].values.tolist()

# %%
overlap_mean_2b = [
    # np.mean(s)
    np.mean(np.array(s) ** 2)
    for s in results_overlap_2b
]

# %%
plt.plot(overlap_mean_2b)

# baselines
baseline_rho095 = 0.7137046225786019
baseline_rho09 = 0.5621312512554327
baseline_rho08 = 0.36662567973860943

plt.axhline(baseline_rho095, linestyle="--", label="ρ=0.95 baseline", color="red")
plt.axhline(baseline_rho09, linestyle="--", label="ρ=0.9 baseline", color="green")
plt.axhline(baseline_rho08, linestyle="--", label="ρ=0.8 baseline", color="blue")

plt.xlabel("Layer")
plt.ylabel("Mean cos² principal angle")

plt.title(
"Layer-wise mean overlap (Qwen3-VL-2B-Thinking)"
)

plt.legend()

plt.show()

# %%
delta_overlap_2b = np.diff(overlap_mean_2b)

plt.plot(delta_overlap_2b)

plt.axhline(0, color="gray", linestyle="--")

plt.xlabel("Layer")

plt.ylabel("Δ subspace overlap")

plt.title(
"Layer-wise change in mean overlap (Qwen3-VL-2B-Thinking)"
)

plt.show()

# %%
results_overlap_4b = pd.read_csv("Results/results_overlap_4b_180.csv").iloc[1:37, :].values.tolist()

# %%
overlap_mean_4b = [
    # np.mean(s)
    np.mean(np.array(s) ** 2)
    for s in results_overlap_4b
]

# %%
plt.plot(overlap_mean_4b)

# baselines
baseline_rho09 = 0.6059633875775221
baseline_rho08 = 0.4209154410415107
baseline_rho07 = 0.3011568144923316

plt.axhline(baseline_rho09, linestyle="--", label="ρ=0.9 baseline", color="red")
plt.axhline(baseline_rho08, linestyle="--", label="ρ=0.8 baseline", color="green")
plt.axhline(baseline_rho07, linestyle="--", label="ρ=0.7 baseline", color="blue")

plt.xlabel("Layer")
plt.ylabel("Mean cos² principal angle")

plt.title("Layer-wise mean overlap (Qwen3-VL-4B-Thinking)")

plt.legend()

plt.show()

# %%
plt.plot(overlap_mean_4b)

plt.xlabel("Layer")
plt.ylabel("Mean cos² principal angle")

plt.title(
"Layer-wise mean overlap (Qwen3-VL-4B-Thinking)"
)

plt.show()

# %%
delta_overlap_4b = np.diff(overlap_mean_4b)

plt.plot(delta_overlap_4b)

plt.axhline(0, color="gray", linestyle="--")

plt.xlabel("Layer")

plt.ylabel("Δ subspace overlap")

plt.title(
"Layer-wise change in mean overlap (Qwen3-VL-4B-Thinking)"
)

plt.show()

# %%
def random_noimage_baseline(
    X_img,
    max_k=20
):
    """
    keep image hidden states real
    replace no-image hidden states with random vectors
    """

    n_samples, dim = X_img.shape

    X_no_random = np.random.randn(
        n_samples,
        dim
    )

    return variance_explained_by_noimage_pc(
        X_img,
        X_no_random,
        max_k=max_k
    )

# %%
def layerwise_random_noimage_baseline(
    hidden_states_img,
    max_k=20
):

    results = []

    for layer in range(len(hidden_states_img)):

        X_img = hidden_states_img[layer]

        ratios = random_noimage_baseline(
            X_img,
            max_k=max_k
        )

        results.append(ratios)

    return results

# %%
with_layers_2b = load_layer_embeddings(with_image_path_2b)
no_layers_2b   = load_layer_embeddings(without_image_path_2b)

# %%
random_baseline_2b = layerwise_random_noimage_baseline(
    with_layers_2b,
    max_k=200
)

# %%
with_layers_4b = load_layer_embeddings(with_image_path)
no_layers_4b   = load_layer_embeddings(without_image_path)

# %%
random_baseline_4b = layerwise_random_noimage_baseline(
    with_layers_4b,
    max_k=200
)

# %%
with_layers_8b = load_layer_embeddings(with_image_path_8b)
no_layers_8b   = load_layer_embeddings(without_image_path_8b)

# %%
random_baseline_8b = layerwise_random_noimage_baseline(
    with_layers_8b,
    max_k=200
)

# %%
with_layers_glm = load_layer_embeddings(with_image_path_glm)
no_layers_glm   = load_layer_embeddings(without_image_path_glm)

# %%
random_baseline_glm = layerwise_random_noimage_baseline(
    with_layers_glm,
    max_k=500
)

# %%
def generate_correlated_gaussian(n, d, rho):
    U = np.random.randn(n, d)
    U_tilde = np.random.randn(n, d)
    V = rho * U + np.sqrt(1 - rho**2) * U_tilde
    return U, V


def principal_angle_baseline(n, d, k=40, rho=0.3, num_trials=100):
    vals = []

    k_eff = min(k, n - 1, d)

    for _ in range(num_trials):
        X_img_rand, X_no_rand = generate_correlated_gaussian(n, d, rho)
        s = subspace_overlap(X_img_rand, X_no_rand, k_eff)
        vals.append(np.mean(np.array(s) ** 2))

    return np.mean(vals)

# %%
n, d = with_layers_4b[0].shape

baseline_const_4b = principal_angle_baseline(
    n=n,
    d=d,
    k=25,
    rho=0.3,
    num_trials=10
)

# %%
print("Baseline mean cos² principal angle (4B):", baseline_const_4b)

# %%
n, d = with_layers_4b[0].shape

baseline_const_4b_08 = principal_angle_baseline(
    n=n,
    d=d,
    k=25,
    rho=0.8,
    num_trials=10
)

print("Baseline mean cos² principal angle (4B, rho=0.8):", baseline_const_4b_08)

# %%
n, d = with_layers_4b[0].shape

baseline_const_4b_09 = principal_angle_baseline(
    n=n,
    d=d,
    k=25,
    rho=0.9,
    num_trials=10
)

print("Baseline mean cos² principal angle (4B, rho=0.9):", baseline_const_4b_09)

# %%
n, d = with_layers_4b[0].shape

baseline_const_4b_095 = principal_angle_baseline(
    n=n,
    d=d,
    k=25,
    rho=0.95,
    num_trials=10
)

print("Baseline mean cos² principal angle (4B, rho=0.95):", baseline_const_4b_095)

# %%
n, d = with_layers_4b[0].shape

baseline_const_4b_098 = principal_angle_baseline(
    n=n,
    d=d,
    k=25,
    rho=0.98,
    num_trials=10
)

print("Baseline mean cos² principal angle (4B, rho=0.98):", baseline_const_4b_098)

# %%
n, d = with_layers_4b[0].shape

baseline_const_4b_099 = principal_angle_baseline(
    n=n,
    d=d,
    k=25,
    rho=0.99,
    num_trials=10
)

print("Baseline mean cos² principal angle (4B, rho=0.99):", baseline_const_4b_099)

# %%
n, d = with_layers_4b[0].shape

baseline_const_4b_098 = principal_angle_baseline(
    n=n,
    d=d,
    k=25,
    rho=0.98,
    num_trials=10
)

print("Baseline mean cos² principal angle (4B, rho=0.98):", baseline_const_4b_098)

# %%
n, d = with_layers_4b[0].shape

baseline_const_4b_k180_09 = principal_angle_baseline(
    n=n,
    d=d,
    k=180,
    rho=0.9,
    num_trials=10
)

print("Baseline mean cos² principal angle (4B, k=180, rho=0.9):", baseline_const_4b_k180_09)

# %%
n, d = with_layers_4b[0].shape

baseline_const_4b_k180_095 = principal_angle_baseline(
    n=n,
    d=d,
    k=180,
    rho=0.95,
    num_trials=10
)

print("Baseline mean cos² principal angle (4B, k=180, rho=0.95):", baseline_const_4b_k180_095)

# %%
n, d = with_layers_4b[0].shape

baseline_const_4b_k180_08 = principal_angle_baseline(
    n=n,
    d=d,
    k=180,
    rho=0.8,
    num_trials=10
)

print("Baseline mean cos² principal angle (4B, k=180, rho=0.8):", baseline_const_4b_k180_08)

# %%
n, d = with_layers_4b[0].shape

baseline_const_4b_k180_07 = principal_angle_baseline(
    n=n,
    d=d,
    k=180,
    rho=0.7,
    num_trials=10
)

print("Baseline mean cos² principal angle (4B, k=180, rho=0.7):", baseline_const_4b_k180_07)

# %%
n, d = with_layers_4b[0].shape

baseline_const_4b_k180_06 = principal_angle_baseline(
    n=n,
    d=d,
    k=180,
    rho=0.6,
    num_trials=10
)

print("Baseline mean cos² principal angle (4B, k=180, rho=0.6):", baseline_const_4b_k180_06)

# %%
n, d = with_layers_4b[0].shape

baseline_const_4b_k180_09 = principal_angle_baseline(
    n=n,
    d=d,
    k=180,
    rho=0.9,
    num_trials=10
)

print("Baseline mean cos² principal angle (4B, k=180, rho=0.9):", baseline_const_4b_k180_09)

# %%
n, d = with_layers_2b[0].shape

baseline_const_2b_k100_09 = principal_angle_baseline(
    n=n,
    d=d,
    k=100,
    rho=0.9,
    num_trials=10
)

print("Baseline mean cos² principal angle (2B, k=100, rho=0.9):", baseline_const_2b_k100_09)

# %%
n, d = with_layers_2b[0].shape

baseline_const_2b_k100_095 = principal_angle_baseline(
    n=n,
    d=d,
    k=100,
    rho=0.95,
    num_trials=10
)

print("Baseline mean cos² principal angle (2B, k=100, rho=0.95):", baseline_const_2b_k100_095)

# %%
n, d = with_layers_2b[0].shape

baseline_const_2b_k100_098 = principal_angle_baseline(
    n=n,
    d=d,
    k=100,
    rho=0.98,
    num_trials=10
)

print("Baseline mean cos² principal angle (2B, k=100, rho=0.98):", baseline_const_2b_k100_098)

# %%
n, d = with_layers_2b[0].shape

baseline_const_2b_k100_08 = principal_angle_baseline(
    n=n,
    d=d,
    k=100,
    rho=0.8,
    num_trials=10
)

print("Baseline mean cos² principal angle (2B, k=100, rho=0.8):", baseline_const_2b_k100_08)

# %%
with_layers_8b = load_layer_embeddings(with_image_path_8b)
no_layers_8b   = load_layer_embeddings(without_image_path_8b)

# %%
n, d = with_layers_8b[0].shape

baseline_const_8b_k160_08 = principal_angle_baseline(
    n=n,
    d=d,
    k=160,
    rho=0.8,
    num_trials=10
)

print("Baseline mean cos² principal angle (8B, k=160, rho=0.8):", baseline_const_8b_k160_08)

# %%
n, d = with_layers_8b[0].shape

baseline_const_8b_k160_09 = principal_angle_baseline(
    n=n,
    d=d,
    k=160,
    rho=0.9,
    num_trials=10
)

print("Baseline mean cos² principal angle (8B, k=160, rho=0.9):", baseline_const_8b_k160_09)

# %%
n, d = with_layers_8b[0].shape

baseline_const_8b_k160_085 = principal_angle_baseline(
    n=n,
    d=d,
    k=160,
    rho=0.85,
    num_trials=10
)

print("Baseline mean cos² principal angle (8B, k=160, rho=0.85):", baseline_const_8b_k160_085)

# %%
with_layers_glm = load_layer_embeddings(with_image_path_glm)
no_layers_glm   = load_layer_embeddings(without_image_path_glm)

# %%
n, d = with_layers_glm[0].shape

baseline_const_glm_k500_085 = principal_angle_baseline(
    n=n,
    d=d,
    k=500,
    rho=0.85,
    num_trials=10
)

print("Baseline mean cos² principal angle (GLM, k=500, rho=0.85):", baseline_const_glm_k500_085)

# %%
n, d = with_layers_glm[0].shape

baseline_const_glm_k500_08 = principal_angle_baseline(
    n=n,
    d=d,
    k=500,
    rho=0.8,
    num_trials=10
)

print("Baseline mean cos² principal angle (GLM, k=500, rho=0.8):", baseline_const_glm_k500_08)

# %%
n, d = with_layers_glm[0].shape

baseline_const_glm_k500_09 = principal_angle_baseline(
    n=n,
    d=d,
    k=500,
    rho=0.9,
    num_trials=10
)

print("Baseline mean cos² principal angle (GLM, k=500, rho=0.9):", baseline_const_glm_k500_09)

# %%
n, d = with_layers_glm[0].shape

baseline_const_glm_k500_095 = principal_angle_baseline(
    n=n,
    d=d,
    k=500,
    rho=0.95,
    num_trials=10
)

print("Baseline mean cos² principal angle (GLM, k=500, rho=0.95):", baseline_const_glm_k500_095)

# %%
n, d = with_layers_glm[0].shape

baseline_const_glm_k500_098 = principal_angle_baseline(
    n=n,
    d=d,
    k=500,
    rho=0.98,
    num_trials=10
)

print("Baseline mean cos² principal angle (GLM, k=500, rho=0.98):", baseline_const_glm_k500_098)

# %%
plt.plot(overlap_mean_glm, label="real")
plt.axhline(
    baseline_const,
    linestyle="--",
    label="Gaussian baseline"
)
plt.xlabel("Layer")
plt.ylabel("Mean cos^2 principal angle")
plt.legend()
plt.show()

# %%
def analyze_visual_subspace(with_layers, no_layers):

    results = {}

    for l in with_layers.keys():

        X_img = with_layers[l]
        X_no  = no_layers[l]

        ratios = variance_explained_by_noimage_pc(X_img, X_no)

        results[l] = ratios

        print(
            f"Layer {l:02d} | "
            f"k=1 {ratios[0]:.3f} | "
            f"k=3 {ratios[2]:.3f} | "
            f"k=5 {ratios[4]:.3f}"
        )

    return results

# %%
def plot_visual_subspace(results):

    for l, ratios in results.items():

        plt.plot(range(1, len(ratios)+1), ratios, label=f"L{l}")

    plt.xlabel("k (no-image PCs)")
    plt.ylabel("Explained variance on image states")
    plt.title("How many PCs explain visual signal?")
    plt.legend()
    plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

def mean_shift_stats(X_img, X_no):
    """
    X_img, X_no: (N, D)

    Returns:
        mu_img: mean vector of image states
        mu_no: mean vector of no-image states
        v: steering / mean shift vector
        shift_norm: ||v||
        base_norm: ||mu_no||
        ratio: ||v|| / ||mu_no||
        cosine: cos(mu_img, mu_no)
    """
    mu_img = X_img.mean(axis=0)
    mu_no = X_no.mean(axis=0)

    v = mu_img - mu_no

    shift_norm = np.linalg.norm(v)
    base_norm = np.linalg.norm(mu_no)
    ratio = shift_norm / (base_norm + 1e-12)

    cosine = np.dot(mu_img, mu_no) / (
        (np.linalg.norm(mu_img) * np.linalg.norm(mu_no)) + 1e-12
    )

    return {
        "mu_img": mu_img,
        "mu_no": mu_no,
        "v": v,
        "shift_norm": float(shift_norm),
        "base_norm": float(base_norm),
        "ratio": float(ratio),
        "cosine": float(cosine),
    }


def analyze_mean_shift(with_image_path, without_image_path):
    with_layers = load_layer_embeddings(with_image_path)
    no_layers = load_layer_embeddings(without_image_path)

    assert with_layers.keys() == no_layers.keys(), "Layer mismatch!"

    results = {
        "layer": [],
        "shift_norm": [],
        "base_norm": [],
        "ratio": [],
        "cosine": [],
    }

    for l in with_layers.keys():
        X_img = with_layers[l]
        X_no = no_layers[l]

        stats = mean_shift_stats(X_img, X_no)

        results["layer"].append(l)
        results["shift_norm"].append(stats["shift_norm"])
        results["base_norm"].append(stats["base_norm"])
        results["ratio"].append(stats["ratio"])
        results["cosine"].append(stats["cosine"])

        print(
            f"Layer {l:02d} | "
            f"||v||={stats['shift_norm']:.4f} | "
            f"||mu_no||={stats['base_norm']:.4f} | "
            f"ratio={stats['ratio']:.4%} | "
            f"cos={stats['cosine']:.4f}"
        )

    return results


def plot_mean_shift_results(results, skip_layer0=False):
    layer = np.array(results["layer"], dtype=float)
    shift_norm = np.array(results["shift_norm"], dtype=float)
    base_norm = np.array(results["base_norm"], dtype=float)
    ratio = np.array(results["ratio"], dtype=float)
    cosine = np.array(results["cosine"], dtype=float)

    if skip_layer0:
        mask = layer > 0
        layer = layer[mask]
        shift_norm = shift_norm[mask]
        base_norm = base_norm[mask]
        ratio = ratio[mask]
        cosine = cosine[mask]

    plt.figure()
    plt.plot(layer, ratio, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("||mu_img - mu_no|| / ||mu_no||")
    plt.title("Relative Mean Shift by Layer")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(layer, cosine, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("cos(mu_img, mu_no)")
    plt.title("Mean Cosine by Layer")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(layer, shift_norm, marker="o", label="||mu_img - mu_no||")
    plt.plot(layer, base_norm, marker="o", label="||mu_no||")
    plt.xlabel("Layer")
    plt.ylabel("Norm")
    plt.title("Mean Vector Norms by Layer")
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%
mean_shift_results = analyze_mean_shift(
    with_image_path,
    without_image_path
)

plot_mean_shift_results(mean_shift_results, skip_layer0=True)

# %%
def analyze_delta_rank(
    with_image_path: str,
    without_image_path: str,
    energy: float = 0.9,
):
    with_layers = load_layer_embeddings(with_image_path)
    no_layers   = load_layer_embeddings(without_image_path)

    assert with_layers.keys() == no_layers.keys()

    results = {
        "layer": [],
        "eff_rank_delta": [],
        "k90_delta": [],
    }

    for l in with_layers.keys():

        X_img = with_layers[l]
        X_no  = no_layers[l]

        # paired difference
        Delta = X_img - X_no

        # reuse your existing functions
        Delta_c = center(Delta)

        r_delta = effective_rank(Delta_c)
        k_delta = k_energy(Delta_c, energy)

        results["layer"].append(l)
        results["eff_rank_delta"].append(r_delta)
        results["k90_delta"].append(k_delta)

        print(
            f"Layer {l:02d} | "
            #f"r_eff(Δ)={r_delta:.2f} | "
            f"k90(Δ)={k_delta}"
        )

    return results

# %%
delta_results = analyze_delta_rank(
    with_image_path,
    without_image_path
)

# %%
def delta_spectrum(X_img, X_no):
    Delta = X_img - X_no
    Delta = center(Delta)

    _, S, _ = np.linalg.svd(Delta, full_matrices=False)

    energy = S**2
    ratio = energy / energy.sum()

    print("top1:", ratio[0])
    print("top3:", ratio[:3].sum())
    print("top5:", ratio[:5].sum())

    return S

# %%
delta_spectrum(X_img, X_no)

# %%
def steering_alignment(X_img, X_no):
    Delta = center(X_img - X_no)

    _, _, Vt = np.linalg.svd(Delta, full_matrices=False)

    principal = Vt[0]

    v = (X_no.mean(0) - X_img.mean(0))
    v = v / np.linalg.norm(v)

    cos = np.dot(v, principal)

    print("cos(v, first PC of Δh) =", cos)

# %%
steering_alignment(X_img, X_no)


