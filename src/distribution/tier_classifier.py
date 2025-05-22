"""
Advanced Tier Classification System - Multi-dimensional market segmentation

This module provides a sophisticated market tier classification system that segments
countries and regions based on multiple characteristics, automatically determines
the optimal number of segments, and provides interpretable classification results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedTierClassifier:
    """
    Advanced market tier classification for countries and regions
    
    This class implements a multi-dimensional classification system that segments
    markets based on multiple characteristics and adapts to different market structures.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the AdvancedTierClassifier
        
        Args:
            config_manager: Optional configuration manager instance
        """
        self.config_manager = config_manager
        
        # Initialize configuration
        self.classification_settings = {}
        self.feature_weights = {}
        self.tier_labels = {}
        self.tier_thresholds = {}
        
        # Initialize clustering results
        self.tier_assignments = {}
        self.cluster_centers = None
        self.feature_importance = {}
        self.optimal_clusters = None
        self.dimensionality_reduction = None
        self.reduced_features = None
        
        # Load settings from config manager if provided
        if config_manager:
            self._load_settings()
    
    def _load_settings(self):
        """Load classification settings from configuration"""
        if not self.config_manager:
            return
            
        # Get tier classification settings
        distribution_settings = self.config_manager.get_market_distribution_settings()
        self.classification_settings = distribution_settings.get('tier_classification', {})
        
        # Get clustering method
        self.clustering_method = self.classification_settings.get('clustering_method', 'kmeans')
        
        # Get dimensionality reduction settings
        self.dim_reduction_method = self.classification_settings.get('dimensionality_reduction', 'pca')
        self.target_dimensions = self.classification_settings.get('target_dimensions', 2)
        
        # Get feature settings
        self.features_to_use = self.classification_settings.get('features', ['market_share', 'growth_rate'])
        self.feature_weights = self.classification_settings.get('feature_weights', {})
        
        # Set default feature weights if not specified
        for feature in self.features_to_use:
            if feature not in self.feature_weights:
                self.feature_weights[feature] = 1.0
        
        # Get tier settings
        self.min_tiers = self.classification_settings.get('min_tiers', 3)
        self.max_tiers = self.classification_settings.get('max_tiers', 8)
        self.auto_determine_tiers = self.classification_settings.get('auto_determine_tiers', True)
        
        # Get tier labels from settings if provided
        self.tier_labels = self.classification_settings.get('tier_labels', {})
        if not self.tier_labels:
            # Default tier labels
            self.tier_labels = {
                0: "Premium Tier",
                1: "High Growth Tier",
                2: "Mid-Market Tier",
                3: "Emerging Tier",
                4: "Base Tier"
            }
    
    def fit(self, market_data: pd.DataFrame, reference_year: Optional[int] = None, 
           features: Optional[List[str]] = None, include_regions: bool = True) -> Dict[str, int]:
        """
        Fit the tier classification model and assign tiers to countries and regions
        
        Args:
            market_data: DataFrame with country/region market data
            reference_year: Year to use for classification (defaults to latest)
            features: List of features to use for classification (defaults to config)
            include_regions: Whether to include regions in classification
            
        Returns:
            Dictionary mapping country/region names to tier indices
        """
        # Make a copy to avoid modifying the original
        data = market_data.copy()
        
        # Use reference year or latest year
        if reference_year is None:
            if 'Year' in data.columns:
                reference_year = data['Year'].max()
            else:
                # Use all data if no year column
                reference_year = None
        
        # Filter data by year if applicable
        if reference_year is not None and 'Year' in data.columns:
            data = data[data['Year'] == reference_year]
        
        # Filter data by region_type if applicable
        if not include_regions and 'region_type' in data.columns:
            data = data[data['region_type'] == 'country']
        
        # Get country/region name column
        if self.config_manager:
            country_mapping = self.config_manager.get_column_mapping('country_historical')
            country_col = country_mapping.get('country_column', 'Country')
        else:
            # Try to infer country column
            for col in ['Country', 'country', 'Name', 'name']:
                if col in data.columns:
                    country_col = col
                    break
            else:
                raise ValueError("Could not determine country column name")
        
        # Use provided features or default from config
        if features:
            self.features_to_use = features
        
        # Check if all required features are available
        for feature in self.features_to_use:
            if feature not in data.columns:
                if feature == 'growth_rate' and 'Value' in data.columns and 'Year' in data.columns:
                    # Calculate growth rate if not present
                    logger.info("Calculating growth rate from market values")
                    data = self._calculate_growth_rates(data)
                elif feature == 'market_share' and 'Value' in data.columns:
                    # Calculate market share if not present
                    logger.info("Calculating market share from market values")
                    data = self._calculate_market_shares(data)
                else:
                    logger.warning(f"Required feature '{feature}' not available in data")
                    # Remove missing feature from list
                    self.features_to_use.remove(feature)
        
        if not self.features_to_use:
            raise ValueError("No valid features available for classification")
        
        # Extract feature data
        feature_data = data[self.features_to_use].copy()
        
        # Apply feature weights
        for feature in self.features_to_use:
            if feature in self.feature_weights:
                feature_data[feature] = feature_data[feature] * self.feature_weights[feature]
        
        # Handle missing values
        feature_data = feature_data.fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        # Apply dimensionality reduction if needed
        if len(self.features_to_use) > 2 and self.target_dimensions < len(self.features_to_use):
            if self.dim_reduction_method == 'pca':
                reducer = PCA(n_components=self.target_dimensions)
            elif self.dim_reduction_method == 'svd':
                reducer = TruncatedSVD(n_components=self.target_dimensions)
            elif self.dim_reduction_method == 'umap':
                reducer = umap.UMAP(n_components=self.target_dimensions)
            else:
                # Default to PCA
                reducer = PCA(n_components=self.target_dimensions)
            
            reduced_features = reducer.fit_transform(scaled_features)
            
            # Store reducer and reduced features for later use
            self.dimensionality_reduction = reducer
            self.reduced_features = reduced_features
        else:
            # No dimensionality reduction needed
            reduced_features = scaled_features
            self.reduced_features = reduced_features
        
        # Determine optimal number of clusters if auto-determination is enabled
        if self.auto_determine_tiers:
            self.optimal_clusters = self._determine_optimal_clusters(reduced_features)
        else:
            # Use fixed number of clusters from config
            fixed_tiers = self.classification_settings.get('fixed_tiers', 5)
            self.optimal_clusters = min(max(fixed_tiers, self.min_tiers), self.max_tiers)
        
        logger.info(f"Using {self.optimal_clusters} tiers for classification")
        
        # Apply clustering method
        if self.clustering_method == 'kmeans':
            clusterer = KMeans(n_clusters=self.optimal_clusters, random_state=42, n_init=10)
        elif self.clustering_method == 'dbscan':
            # Determine epsilon from data
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(reduced_features)
            distances, _ = nn.kneighbors(reduced_features)
            epsilon = np.percentile(distances[:,1], 90)  # Use 90th percentile as heuristic
            clusterer = DBSCAN(eps=epsilon, min_samples=3)
        elif self.clustering_method == 'hdbscan':
            clusterer = HDBSCAN(min_cluster_size=3, min_samples=2)
        elif self.clustering_method == 'gmm':
            clusterer = GaussianMixture(n_components=self.optimal_clusters, random_state=42)
        else:
            # Default to KMeans
            clusterer = KMeans(n_clusters=self.optimal_clusters, random_state=42, n_init=10)
        
        # Fit the clustering model
        cluster_labels = clusterer.fit_predict(reduced_features)
        
        # For models like DBSCAN that may return -1 for noise points,
        # we'll assign those to a new cluster
        if -1 in cluster_labels:
            max_cluster = cluster_labels.max()
            cluster_labels[cluster_labels == -1] = max_cluster + 1
        
        # Store cluster centers if the model has them
        if hasattr(clusterer, 'cluster_centers_'):
            self.cluster_centers = clusterer.cluster_centers_
        
        # Calculate cluster metrics for each feature
        feature_importance = {}
        for i, feature in enumerate(self.features_to_use):
            # Calculate how well each feature discriminates the clusters
            # Using normalized mutual information or similar metric
            importance = self._calculate_feature_importance(scaled_features[:, i], cluster_labels)
            feature_importance[feature] = importance
        
        self.feature_importance = feature_importance
        
        # Create mapping from entities to cluster labels
        entities = data[country_col].tolist()
        tier_assignments = {entity: int(label) for entity, label in zip(entities, cluster_labels)}
        
        # Rearrange clusters by importance (higher market share = lower tier number)
        if 'market_share' in self.features_to_use:
            market_share_index = self.features_to_use.index('market_share')
            
            # Calculate average market share for each cluster
            cluster_shares = {}
            for cluster_id in range(self.optimal_clusters):
                mask = cluster_labels == cluster_id
                if np.any(mask):
                    avg_share = np.mean(scaled_features[mask, market_share_index])
                    cluster_shares[cluster_id] = avg_share
            
            # Sort clusters by average market share (descending)
            sorted_clusters = sorted(cluster_shares.items(), key=lambda x: -x[1])
            
            # Create mapping from old to new cluster IDs
            cluster_mapping = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_clusters)}
            
            # Remap tier assignments
            tier_assignments = {entity: cluster_mapping.get(label, label) 
                              for entity, label in tier_assignments.items()}
        
        self.tier_assignments = tier_assignments
        
        # Generate tier thresholds for reporting
        self._calculate_tier_thresholds(data, feature_data, tier_assignments, country_col)
        
        return tier_assignments
    
    def _determine_optimal_clusters(self, features: np.ndarray) -> int:
        """
        Determine the optimal number of clusters (tiers) using silhouette score
        
        Args:
            features: Array of feature values to cluster
            
        Returns:
            Optimal number of clusters
        """
        # Calculate silhouette score for different numbers of clusters
        silhouette_scores = []
        
        # Try different numbers of clusters within the specified range
        for n_clusters in range(self.min_tiers, self.max_tiers + 1):
            # Skip if we have fewer data points than clusters
            if features.shape[0] <= n_clusters:
                continue
                
            try:
                # Use KMeans for evaluation
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                
                # Calculate silhouette score
                score = silhouette_score(features, cluster_labels)
                silhouette_scores.append((n_clusters, score))
                
                logger.debug(f"Silhouette score for {n_clusters} clusters: {score:.4f}")
            except Exception as e:
                logger.warning(f"Error calculating silhouette score for {n_clusters} clusters: {str(e)}")
        
        if not silhouette_scores:
            # Fall back to default
            logger.warning("Could not determine optimal number of clusters, using default")
            return self.min_tiers
        
        # Find the number of clusters with the highest silhouette score
        optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
        
        return optimal_clusters
    
    def _calculate_feature_importance(self, feature_values: np.ndarray, 
                                    cluster_labels: np.ndarray) -> float:
        """
        Calculate the importance of a feature for discriminating clusters
        
        Args:
            feature_values: Array of feature values
            cluster_labels: Array of cluster labels
            
        Returns:
            Importance score for the feature
        """
        from sklearn.metrics import mutual_info_score
        
        # Discretize feature values for mutual information calculation
        feature_bins = np.linspace(feature_values.min(), feature_values.max(), 10)
        feature_discrete = np.digitize(feature_values, feature_bins)
        
        # Calculate mutual information
        mi = mutual_info_score(feature_discrete, cluster_labels)
        
        # Normalize by entropy
        from scipy.stats import entropy
        feature_entropy = entropy(np.bincount(feature_discrete) / len(feature_discrete))
        if feature_entropy > 0:
            normalized_mi = mi / feature_entropy
        else:
            normalized_mi = 0.0
        
        return normalized_mi
    
    def _calculate_growth_rates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate growth rates from market values
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with added growth_rate column
        """
        result = data.copy()
        
        # Check if we have Year and country/region columns
        if 'Year' not in result.columns:
            # Can't calculate growth rate without year
            return result
        
        # Try to identify country column
        country_col = None
        for col in ['Country', 'country', 'Name', 'name', 'Region', 'region']:
            if col in result.columns:
                country_col = col
                break
        
        if not country_col:
            # Can't calculate growth rate without country/region
            return result
        
        # Calculate growth rate for each country/region
        result['growth_rate'] = 0.0
        
        for entity in result[country_col].unique():
            entity_data = result[result[country_col] == entity].sort_values('Year')
            if len(entity_data) > 1:
                entity_data['growth_rate'] = entity_data['Value'].pct_change() * 100
                
                # Update the main dataframe
                for idx, row in entity_data.iterrows():
                    result.loc[idx, 'growth_rate'] = row['growth_rate']
        
        return result
    
    def _calculate_market_shares(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market shares from market values
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with added market_share column
        """
        result = data.copy()
        
        # Calculate market shares for each year
        if 'Year' in result.columns:
            for year in result['Year'].unique():
                year_data = result[result['Year'] == year]
                total_value = year_data['Value'].sum()
                if total_value > 0:
                    result.loc[result['Year'] == year, 'market_share'] = \
                        result.loc[result['Year'] == year, 'Value'] / total_value * 100
        else:
            # Calculate overall market share
            total_value = result['Value'].sum()
            if total_value > 0:
                result['market_share'] = result['Value'] / total_value * 100
        
        return result
    
    def _calculate_tier_thresholds(self, data: pd.DataFrame, feature_data: pd.DataFrame,
                                 tier_assignments: Dict[str, int], country_col: str) -> None:
        """
        Calculate threshold values for each tier based on feature distributions
        
        Args:
            data: Original market data
            feature_data: Feature data used for clustering
            tier_assignments: Mapping from entities to tier indices
            country_col: Name of the country/region column
        """
        # Create reverse mapping from tier to entities
        tier_entities = {}
        for entity, tier in tier_assignments.items():
            if tier not in tier_entities:
                tier_entities[tier] = []
            tier_entities[tier].append(entity)
        
        # Initialize thresholds dictionary
        self.tier_thresholds = {}
        
        # Calculate thresholds for each feature and tier
        for feature in self.features_to_use:
            feature_thresholds = {}
            
            for tier in sorted(tier_entities.keys()):
                entities = tier_entities[tier]
                
                # Get feature values for entities in this tier
                mask = data[country_col].isin(entities)
                values = data.loc[mask, feature]
                
                if not values.empty:
                    # Calculate min, mean, and max
                    min_val = values.min()
                    mean_val = values.mean()
                    max_val = values.max()
                    
                    feature_thresholds[tier] = {
                        'min': min_val,
                        'mean': mean_val,
                        'max': max_val
                    }
            
            self.tier_thresholds[feature] = feature_thresholds
    
    def predict(self, new_data: pd.DataFrame, country_col: str = 'Country') -> Dict[str, int]:
        """
        Predict tiers for new data based on the fitted model
        
        Args:
            new_data: DataFrame with new market data
            country_col: Name of the country/region column
            
        Returns:
            Dictionary mapping country/region names to tier indices
        """
        if self.reduced_features is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Extract feature data from new data
        feature_data = new_data[self.features_to_use].copy()
        
        # Apply feature weights
        for feature in self.features_to_use:
            if feature in self.feature_weights:
                feature_data[feature] = feature_data[feature] * self.feature_weights[feature]
        
        # Handle missing values
        feature_data = feature_data.fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        # Apply dimensionality reduction if needed
        if self.dimensionality_reduction is not None:
            reduced_features = self.dimensionality_reduction.transform(scaled_features)
        else:
            reduced_features = scaled_features
        
        # Use KMeans with existing centers for prediction
        if self.cluster_centers is not None:
            kmeans = KMeans(n_clusters=len(self.cluster_centers), n_init=1)
            kmeans.cluster_centers_ = self.cluster_centers
            kmeans.labels_ = np.zeros(len(self.reduced_features), dtype=int)  # Dummy labels
            
            # Predict clusters
            cluster_labels = kmeans.predict(reduced_features)
        else:
            # Use the whole clustering process again
            kmeans = KMeans(n_clusters=self.optimal_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(reduced_features)
        
        # Create mapping from entities to cluster labels
        entities = new_data[country_col].tolist()
        tier_assignments = {entity: int(label) for entity, label in zip(entities, cluster_labels)}
        
        return tier_assignments
    
    def get_tier_characteristics(self) -> Dict[int, Dict[str, Any]]:
        """
        Get characteristics of each tier based on feature distributions
        
        Returns:
            Dictionary mapping tier indices to characteristic dictionaries
        """
        if not self.tier_thresholds:
            return {}
        
        characteristics = {}
        
        for tier in range(self.optimal_clusters):
            tier_data = {}
            
            for feature, thresholds in self.tier_thresholds.items():
                if tier in thresholds:
                    tier_data[feature] = thresholds[tier]
            
            # Add tier label
            tier_data['label'] = self.tier_labels.get(tier, f"Tier {tier+1}")
            
            characteristics[tier] = tier_data
        
        return characteristics
    
    def visualize_tiers(self, market_data: pd.DataFrame, 
                      features: Optional[Tuple[str, str]] = None, 
                      country_col: str = 'Country') -> plt.Figure:
        """
        Create visualization of tier classification
        
        Args:
            market_data: DataFrame with market data
            features: Optional tuple of two feature names to plot (defaults to first two)
            country_col: Name of the country/region column
            
        Returns:
            Matplotlib figure with visualization
        """
        if not self.tier_assignments:
            raise RuntimeError("Model must be fitted before visualization")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Choose features to plot
        if features and len(features) == 2 and features[0] in market_data.columns and features[1] in market_data.columns:
            x_feature, y_feature = features
        elif len(self.features_to_use) >= 2:
            x_feature, y_feature = self.features_to_use[:2]
        else:
            # Default to market share and growth rate if available
            x_feature = 'market_share' if 'market_share' in market_data.columns else self.features_to_use[0]
            y_feature = 'growth_rate' if 'growth_rate' in market_data.columns else self.features_to_use[0]
        
        # Create a mapping from tier to color
        colormap = plt.cm.viridis
        colors = colormap(np.linspace(0, 1, self.optimal_clusters))
        
        # Plot each tier with different color
        for tier in range(self.optimal_clusters):
            # Get entities in this tier
            tier_entities = [entity for entity, t in self.tier_assignments.items() if t == tier]
            
            # Filter data for these entities
            tier_data = market_data[market_data[country_col].isin(tier_entities)]
            
            # Skip if no data
            if tier_data.empty:
                continue
            
            # Plot data points with size proportional to market value if available
            if 'Value' in tier_data.columns:
                sizes = tier_data['Value'] / tier_data['Value'].max() * 200 + 30
            else:
                sizes = 100
            
            # Plot scatter
            ax.scatter(tier_data[x_feature], tier_data[y_feature], 
                      c=[colors[tier]], label=self.tier_labels.get(tier, f"Tier {tier+1}"),
                      s=sizes, alpha=0.7)
            
            # Annotate points with country/region names
            for idx, row in tier_data.iterrows():
                ax.annotate(row[country_col], 
                           (row[x_feature], row[y_feature]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8)
        
        # Add labels and legend
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title("Market Tier Classification")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def get_tier_transition_probabilities(self, market_data: pd.DataFrame, 
                                        country_col: str = 'Country') -> pd.DataFrame:
        """
        Calculate tier transition probabilities based on historical data
        
        Args:
            market_data: DataFrame with historical market data
            country_col: Name of the country/region column
            
        Returns:
            DataFrame with tier transition probabilities
        """
        if not self.tier_assignments or 'Year' not in market_data.columns:
            return pd.DataFrame()
        
        # Calculate tier for each country/region and year
        all_tiers = {}
        
        # Get unique years
        years = sorted(market_data['Year'].unique())
        
        for year in years:
            # Fit the model for this year
            year_data = market_data[market_data['Year'] == year]
            year_tiers = self.fit(year_data, reference_year=year)
            
            # Store tiers
            for entity, tier in year_tiers.items():
                if entity not in all_tiers:
                    all_tiers[entity] = {}
                all_tiers[entity][year] = tier
        
        # Calculate transitions
        transitions = []
        
        for entity, year_tiers in all_tiers.items():
            for i in range(len(years) - 1):
                from_year = years[i]
                to_year = years[i+1]
                
                if from_year in year_tiers and to_year in year_tiers:
                    from_tier = year_tiers[from_year]
                    to_tier = year_tiers[to_year]
                    
                    transitions.append((from_tier, to_tier))
        
        # Calculate transition probability matrix
        transition_matrix = np.zeros((self.optimal_clusters, self.optimal_clusters))
        
        for from_tier, to_tier in transitions:
            transition_matrix[from_tier, to_tier] += 1
        
        # Normalize by row sums
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                    out=np.zeros_like(transition_matrix), where=row_sums != 0)
        
        # Convert to DataFrame
        transition_df = pd.DataFrame(transition_matrix, 
                                   index=[self.tier_labels.get(i, f"Tier {i+1}") for i in range(self.optimal_clusters)],
                                   columns=[self.tier_labels.get(i, f"Tier {i+1}") for i in range(self.optimal_clusters)])
        
        return transition_df