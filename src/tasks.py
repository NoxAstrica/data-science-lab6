import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sns.set_style('whitegrid')


def load_data(red_path: str, white_path: str) -> pd.DataFrame:
    df_red = pd.read_csv(red_path, sep=';' if ';' in open(red_path).readline() else ',')
    df_white = pd.read_csv(white_path, sep=';' if ';' in open(white_path).readline() else ',')
    
    df_red['type'] = 'red'
    df_white['type'] = 'white'
    
    data = pd.concat([df_red, df_white], ignore_index=True)
    return data


def remove_outliers(df: pd.DataFrame, features: pd.Index) -> pd.DataFrame:
    """Removing outliers from a dataset using the IQR method."""
    df_clean = df.copy()
    
    numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df_clean[f])]
    
    Q1 = df_clean[numeric_features].quantile(0.25)
    Q3 = df_clean[numeric_features].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    condition = ~((df_clean[numeric_features] < lower_bound) | (df_clean[numeric_features] > upper_bound)).any(axis=1)
    df_clean = df_clean[condition]

    return df_clean


def encode_categorical_variables(data: pd.DataFrame) -> pd.DataFrame:
    """Encoding of categorical variables."""
    data_encoded = data.copy()

    if 'type' in data_encoded.columns:
        data_encoded = pd.get_dummies(data_encoded, columns=['type'], dtype=int)
        
    return data_encoded


def scale_features(
    X_train: pd.DataFrame
) -> np.ndarray:
    """Standartization (scaling) of features"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    return X_train_scaled


def apply_pca(X: pd.DataFrame) -> np.ndarray:
    """
    Application of the principal component analysis (PCA) method.
    """
    X_scaled = scale_features(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca


def perform_clustering(X_pca: np.ndarray) -> dict[str, float | np.ndarray]:
    best_score = -1
    best_k = 2
    best_labels = None

    # Finding optimal number of clusters from 2 to 10 using silhouette score
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    results = {'labels': best_labels, 'silhouette_score': float(best_score)}
    return results


def main():
    # Loading data
    data = load_data('winequality-red.csv', 'winequality-white.csv')

    # Removing outliers
    features = data.columns[:-2]  # Without 'quality' and 'type'
    data = remove_outliers(data, features)

    # Encoding categorical variables
    data = encode_categorical_variables(data)

    # Unsupervised learning task
    X_features = data.drop(['quality'], axis=1)
    X_pca = apply_pca(X_features)
    clustering_results = perform_clustering(X_pca)
    print("\nClustering Results:")
    print(f"Silhouette coefficient: {clustering_results['silhouette_score']:.4f}")

    # Visualization of clusters
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0],
                    y=X_pca[:, 1],
                    hue=clustering_results['labels'],
                    palette='viridis')
    plt.title('K-means Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


if __name__ == "__main__":
    main()
