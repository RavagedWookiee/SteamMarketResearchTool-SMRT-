import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from time import time

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings

import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk

df = None
popup_window = None
genre_listbox = None
mechanic_listbox = None

warnings.filterwarnings("ignore")

try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Synonym map
synonym_map = {
    "roguelike": "roguelike",
    "rogue-lite": "roguelike",
    "rogue lite": "roguelike",
    "soulslike": "souls-like",
    "souls-like": "souls-like",
    "deckbuilder": "deck-building",
    "deck builder": "deck-building",
    "deck-building": "deck-building",
    "multiplayer co-op": "co-op",
    "coop": "co-op",
    "co-op": "co-op",
    "fps": "first-person shooter",
    "first-person shooter": "first-person shooter",
    "tps": "third-person shooter",
    "third-person shooter": "third-person shooter",
    "open world": "open-world",
    "open-world": "open-world",
    "4x": "4x strategy",
    "4x strategy": "4x strategy",
    "jrpg": "j-rpg",
    "j-rpg": "j-rpg",
}

# Define the mechanics keyword dictionary and mechanic extraction function

MECHANIC_KEYWORDS = {
    "crafting": ["craft", "crafting", "gather", "gathering"],
    "base building": ["base building", "build your base", "construct", "construction"],
    "exploration": ["explore", "exploration", "open world", "roam", "discover"],
    "stealth": ["stealth", "sneak", "hide", "avoid detection"],
    "combat": ["fight", "combat", "battle", "attack", "kick", "shoot", "weapon", "enemy"],
    "multiplayer": ["multiplayer", "co-op", "online play", "head to head", "versus", "pvp"],
    "puzzle": ["puzzle", "logic challenge", "brain teaser", "problem solving"],
    "story": ["story", "campaign", "narrative", "plot"],
    "progression": ["upgrade", "level up", "unlock", "experience", "xp", "progress"],
    "physics": ["physics-based", "ragdoll", "momentum", "gravity"],
    "timing": ["reaction", "reflex", "timing-based", "quick time"],
    "achievements": ["achievement", "steam achievements", "trophy"],
    "vr": ["vr", "virtual reality"],
    "simulation": ["simulate", "simulation", "realistic", "management"],
    "sandbox": ["sandbox", "creative mode", "free build"],
    "strategy": ["strategy", "tactics", "planning", "turn-based"],
    "survival": ["survive", "survival", "hardcore", "hunger", "thirst"],
    "customization": ["customize", "customization", "character creation", "skins"],
    "minigames": ["minigame", "side game", "bonus game"],
    "rpg elements": ["stats", "abilities", "role-playing", "rpg", "talent tree"],
    "platforming": ["platformer", "jump", "double jump", "obstacle"],
    "driving": ["drive", "vehicle", "car", "race"],
}

def extract_mechanics_from_description(text, keyword_dict=MECHANIC_KEYWORDS, genre=None):
    if pd.isna(text):
        return []
    text = text.lower()
    found = set()
    for mechanic, keywords in keyword_dict.items():
        if any(kw in text for kw in keywords):
            found.add(mechanic)
    if genre:
        found.add(genre)  # Add the genre explicitly to the set if specified
    return list(found)


def plot_mechanic_cooccurrence(df):
    mechanic_matrix = pd.DataFrame(columns=df['Game'], index=df['Mechanics'])
    for idx, game in df.iterrows():
        for mechanic in game['Mechanics']:
            mechanic_matrix.at[mechanic, game['Game']] = 1
    co_occurrence = mechanic_matrix.dot(mechanic_matrix.T)
    sns.heatmap(co_occurrence, cmap="coolwarm", annot=True, fmt="d")
    plt.title("Mechanic Co-occurrence Heatmap")
    plt.show()

# Load and normalize
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Loaded:", df.shape)
    return df

# Load genres from the CSV
def load_genres_from_csv(file_path):
    df = pd.read_csv(file_path)
    genres = df['Genres'].dropna().unique()
    return sorted(genres)


def parse_and_normalize(text):
    if pd.isna(text):
        return []
    tags = [tag.strip().lower() for tag in re.split(',|;', text)]
    normalized_tags = []
    for tag in tags:
        # Replace with synonym if available
        tag = synonym_map.get(tag, tag)
        normalized_tags.append(tag)
    return list(set(normalized_tags))


# Feature Engineering
def preprocess(df, min_df=10, max_df_ratio=0.7):
    df = df[["Name", "Genres", "Tags", "Categories"]].dropna(subset=["Tags", "Genres"])
    df["Parsed_Tags"] = df["Tags"].apply(parse_and_normalize)
    df["Parsed_Genres"] = df["Genres"].apply(parse_and_normalize)
    df["Parsed_Categories"] = df["Categories"].apply(parse_and_normalize)
    df["All_Features"] = df["Parsed_Tags"] + df["Parsed_Genres"] + df["Parsed_Categories"]

    mlb = MultiLabelBinarizer()
    feature_matrix = mlb.fit_transform(df["All_Features"])
    feature_df = pd.DataFrame(feature_matrix, columns=mlb.classes_)

    tag_counts = feature_df.sum()
    max_df = max_df_ratio * len(feature_df)
    filtered_columns = tag_counts[(tag_counts >= min_df) & (tag_counts <= max_df)].index
    feature_df = feature_df[filtered_columns]

    return df.reset_index(drop=True), feature_df


# Evaluation
def evaluate_clusters(data, labels):
    print("Evaluation Metrics:")
    print("Silhouette Score:", silhouette_score(data, labels))
    print("Davies-Bouldin Score:", davies_bouldin_score(data, labels))
    print("Calinski-Harabasz Index:", calinski_harabasz_score(data, labels))


# Dimensionality reduction
def reduce_umap(data, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    return reducer.fit_transform(data)


# Clustering methods
def run_kmeans(data, k=6):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(data)
    print("\n--- k-Means ---")
    evaluate_clusters(data, labels)
    return labels


def run_gmm(data, k=6):
    model = GaussianMixture(n_components=k, random_state=42)
    labels = model.fit_predict(data)
    print("\n--- Gaussian Mixture Model ---")
    evaluate_clusters(data, labels)
    return labels


def run_dbscan(data):
    model = DBSCAN(eps=0.5, min_samples=5)
    labels = model.fit_predict(data)
    print("\n--- DBSCAN ---")
    evaluate_clusters(data, labels)
    return labels


def run_hdbscan(data):
    if not HDBSCAN_AVAILABLE:
        print("HDBSCAN is not installed.")
        return np.zeros(len(data))
    model = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = model.fit_predict(data)
    print("\n--- HDBSCAN ---")
    evaluate_clusters(data, labels)
    return labels


def run_hierarchical(data, n_clusters=6):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(data)
    print("\n--- Hierarchical Clustering ---")
    evaluate_clusters(data, labels)
    link = linkage(data, method='ward')
    plt.figure(figsize=(10, 5))
    dendrogram(link, truncate_mode='level', p=5)
    plt.title("Dendrogram (Hierarchical Clustering)")
    plt.tight_layout()
    plt.show()
    return labels


# Visuals
def plot_clusters(reduced, labels, title="Cluster Plot"):
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.show()


def plot_tag_heatmap(feature_df):
    corr = feature_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", xticklabels=False, yticklabels=False)
    plt.title("Tag Co-occurrence Heatmap")
    plt.tight_layout()
    plt.show()


# New: Cluster tag summarization
def summarize_cluster_tags(df, cluster_column="KMeans_Cluster", tag_column="Tags", top_n=10):
    print(f"\n📊 Top tags per cluster ({cluster_column}):")
    for cluster in sorted(df[cluster_column].dropna().unique()):
        cluster_df = df[df[cluster_column] == cluster]
        tags = cluster_df[tag_column].dropna().str.cat(sep=",").lower().split(",")
        tags = [tag.strip() for tag in tags if tag.strip()]
        tag_counts = pd.Series(tags).value_counts().head(top_n)
        print(f"\n🔹 Cluster {cluster} ({len(cluster_df)} games):")
        print(tag_counts)


import matplotlib.pyplot as plt
import seaborn as sns


def plot_clean_tag_heatmap(feature_df, max_tags=30):
    """
    Plots a clean, focused tag co-occurrence heatmap using the most common tags.
    This improves readability over plotting all possible tags.
    """
    # Get top N most common tags
    top_tags = feature_df.sum().sort_values(ascending=False).head(max_tags).index
    filtered_df = feature_df[top_tags]

    # Compute co-occurrence matrix
    co_occurrence = filtered_df.T.dot(filtered_df)
    np.fill_diagonal(co_occurrence.values, 0)  # Hide self-co-occurrence

    # Normalize to % of co-occurrence over total appearances for interpretability
    normed_co = co_occurrence.div(co_occurrence.sum(axis=1), axis=0).fillna(0)

    # Plot
    plt.figure(figsize=(14, 12))
    sns.heatmap(normed_co, cmap="coolwarm", linewidths=0.5, linecolor='gray',
                xticklabels=top_tags, yticklabels=top_tags, annot=False)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("🔍 Tag Co-occurrence Heatmap (Normalized %)", fontsize=14)
    plt.tight_layout()
    plt.show()


def optimize_kmeans_k(data, k_min=2, k_max=20):
    scores = []
    ks = list(range(k_min, k_max + 1))

    print("\n🔎 Optimizing k for KMeans...")

    for k in ks:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append(score)
        print(f"k={k}: Silhouette Score={score:.4f}")

    best_k = ks[np.argmax(scores)]
    print(f"\n✅ Best k found: {best_k} (Silhouette Score={max(scores):.4f})")

    # Plot Silhouette Score vs k
    plt.figure(figsize=(8, 5))
    plt.plot(ks, scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters (k)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return best_k


def detect_small_clusters(df, cluster_columns, threshold=5):

    print("\n🔍 Detecting small clusters (white space candidates)...")

    for cluster_col in cluster_columns:
        print(f"\n🔎 {cluster_col}:")
        cluster_sizes = df[cluster_col].value_counts()

        small_clusters = cluster_sizes[cluster_sizes <= threshold]
        if small_clusters.empty:
            print(f"✅ No small clusters found in {cluster_col}.")
        else:
            print(f"⚡ Found {len(small_clusters)} small clusters:")
            for cluster_id, size in small_clusters.items():
                print(f"   - Cluster {cluster_id}: {size} games")

            small_games = df[df[cluster_col].isin(small_clusters.index)][["Name", cluster_col]]
            print("\nExample games in small clusters:")
            print(small_games.head(10))


def plot_umap_density(reduced, bins=100, title="UMAP Density Map"):
    """
    Plots a density heatmap of the UMAP projection to visualize sparse regions (white space).
    """
    x = reduced[:, 0]
    y = reduced[:, 1]

    plt.figure(figsize=(10, 8))
    plt.hist2d(x, y, bins=bins, cmap="viridis", cmin=1)
    plt.colorbar(label="Game Density")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title(title)
    plt.tight_layout()
    plt.show()



# Main
def main():
    print("📥 Loading dataset...")
    df = load_data("games.csv")
    df = df.sample(n=35000, random_state=42).reset_index(drop=True)

    print("🧹 Preprocessing features...")
    df, feature_df = preprocess(df)

    print("📊 Generating tag co-occurrence heatmap...")
    plot_clean_tag_heatmap(feature_df, max_tags=30)

    if UMAP_AVAILABLE:
        print("🔽 Reducing dimensions with UMAP...")
        start = time()
        reduced = reduce_umap(feature_df)
        print(f"✅ UMAP completed in {time() - start:.2f} seconds.")

        # NEW: Plot density map to visualize sparse regions (white space)
        print("📡 Plotting UMAP density...")
        plot_umap_density(reduced, title="UMAP Density Heatmap (10k Games)")
    else:
        print("UMAP not available, skipping dimensionality reduction.")
        return

    # Choose whether to optimize k
    optimize_k = True  # Change this to False if you want to skip optimization
    if optimize_k:
        best_k = optimize_kmeans_k(feature_df, k_min=2, k_max=20)
    else:
        best_k = 6  # Default k

    clusterers = {
        "KMeans": lambda data: run_kmeans(data, k=best_k),
        "GMM": lambda data: run_gmm(data, k=best_k),
        "Hierarchical": lambda data: run_hierarchical(data, n_clusters=best_k),
        "DBSCAN": run_dbscan
    }

    if HDBSCAN_AVAILABLE:
        clusterers["HDBSCAN"] = run_hdbscan

    for name, method in clusterers.items():
        print(f"\n🔄 Running {name} clustering...")
        start = time()
        labels = method(feature_df)
        elapsed = time() - start
        print(f"✅ {name} clustering completed in {elapsed:.2f} seconds.")
        print(f"🖼️ Plotting {name} results...")
        plot_clusters(reduced, labels, title=f"{name} Clustering (UMAP Projection)")
        df[name + "_Cluster"] = labels

        summarize_cluster_tags(df, cluster_column=f"{name}_Cluster")

    print("💾 Saving results to full_clustered_output.csv...")
    df.to_csv("full_clustered_output.csv", index=False)
    print("✅ All clustering complete.")

    print("🛰️ Scanning for white space...")
    detect_small_clusters(df, cluster_columns=[
        "KMeans_Cluster",
        "GMM_Cluster",
        "Hierarchical_Cluster",
        "DBSCAN_Cluster",
    ], threshold=5)


# Genre and Mechanic extraction
def extract_mechanics_for_genre(selected_genres, df):
    # Ensure Genres is a string or empty, otherwise set to empty
    df['Genres'] = df['Genres'].apply(lambda x: x if isinstance(x, str) else '')

    # Filter relevant games based on selected genres
    relevant_games = df[df['Genres'].apply(lambda x: any(genre in x for genre in selected_genres))]

    mechanics = []
    for _, game in relevant_games.iterrows():
        mechanics.extend(extract_mechanics_from_description(game['Tags'], MECHANIC_KEYWORDS))  # Adjusting for mechanics

    mechanic_counts = {mechanic: mechanics.count(mechanic) for mechanic in set(mechanics)}

    return mechanic_counts


# Define the on_genre_select function
def on_genre_select():
    global df, genre_listbox, mechanic_listbox
    # Get selected genres from the listbox
    selected_genres = [genre_listbox.get(i) for i in genre_listbox.curselection()]

    # If at least one genre is selected
    if selected_genres:
        # Get mechanic counts for the selected genres
        mechanic_counts = extract_mechanics_for_genre(selected_genres, df)

        # Update the mechanic listbox with the mechanic counts
        mechanic_listbox.delete(0, tk.END)  # Clear existing items
        for mechanic, count in mechanic_counts.items():
            mechanic_listbox.insert(tk.END, f"{mechanic} (Count: {count})")
    else:
        # If no genre is selected, show a warning
        messagebox.showwarning("No Genre Selected", "Please select at least one genre.")


# Modify the show_genre_popup function
def show_genre_popup(df):
    global popup_window, genre_listbox, mechanic_listbox

    if popup_window is None or not popup_window.winfo_exists():
        popup_window = tk.Toplevel()
        popup_window.title("Explore Game Mechanics by Genre")
        popup_window.geometry("480x450")
        popup_window.resizable(False, False)

        style = ttk.Style()
        style.theme_use('clam')  # or 'alt', 'default', etc.

        ttk.Label(popup_window, text="🎮 Select Genres:", font=("Segoe UI", 11, "bold")).pack(pady=(10, 5))

        genre_listbox = tk.Listbox(popup_window, selectmode=tk.MULTIPLE, height=10, font=("Segoe UI", 10), exportselection=False)
        genres = load_genres_from_csv("games.csv")
        for genre in genres:
            genre_listbox.insert(tk.END, genre)
        genre_listbox.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        ttk.Button(popup_window, text="Show Associated Mechanics", command=on_genre_select).pack(pady=10)

        ttk.Label(popup_window, text="🧩 Mechanics Found:", font=("Segoe UI", 11, "bold")).pack(pady=(10, 5))
        mechanic_listbox = tk.Listbox(popup_window, height=10, font=("Segoe UI", 10))
        mechanic_listbox.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        popup_window.focus_force()
    else:
        popup_window.lift()




# Main function to load data and launch the popup
def launch_gui():
    global df

    # Load and sample the data
    df = load_data("games.csv")
    df = df.sample(n=35000, random_state=42).reset_index(drop=True)

    # Setup root window but keep it hidden
    root = tk.Tk()
    root.withdraw()  # Hide the main root window

    # Show popup
    show_genre_popup(df)

    # Start the Tkinter event loop
    root.mainloop()


if __name__ == "__main__":
    main()         # Runs clustering, saves CSV, etc.
    launch_gui()   # Then opens GUI for exploration
