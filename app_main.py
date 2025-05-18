import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import torch.nn.functional as F
import torch.nn as nn
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import kneighbors_graph
import community as community_louvain
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from io import BytesIO
import base64
from PIL import Image, ImageDraw, ImageFont

# Set page config
st.set_page_config(page_title="GNN Polarization Detector", layout="wide")

# Main title
st.title("🧠 GNN-Based Community & Polarization Detector")
st.markdown("Analyze how communities form across social networks using Graph Neural Networks.")

# Sidebar
st.sidebar.header("📁 Dataset & Controls")
DATASETS = {
    "Athletes": "athletes_edges.csv",
    "Government": "government_edges.csv",
    "News Sites": "news_sites_edges.csv",
    "Public Figures": "public_figure_edges.csv",
    "TV Shows": "tvshow_edges.csv",
    "Artists": "artist_edges.csv",
    "Politicians": "politician_edges.csv",
    "Companies": "company_edges.csv"
}
selected_label = st.sidebar.selectbox("Select dataset:", list(DATASETS.keys()))
k = st.sidebar.slider("Number of communities (KMeans)", 2, 10, 4)
view = st.sidebar.radio("Select view:", ["Graph View", "Embedding View", "Metrics Dashboard", "Export Report", "Comparative Analysis"])

selected_file = f"{DATASETS[selected_label]}"

# Data loading and processing functions
@st.cache_data
def load_and_preprocess_data(file_path, compute_features=True):
    """
    Load a graph from an edge list file, clean it, and compute node features.
    
    Args:
        file_path: Path to the edge list CSV file
        compute_features: Whether to compute node features
        
    Returns:
        G: NetworkX graph
        features: DataFrame of node features
    """
    # Load edges
    print(f"Loading data from {file_path}...")
    df_edges = pd.read_csv(file_path)
    
    # Construct graph
    print("Constructing graph...")
    G = nx.from_pandas_edgelist(df_edges, source='node_1', target='node_2')
    
    # Clean graph
    print("Cleaning graph...")
    G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops
    
    # Ensure graph is connected (take largest component)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"Extracted largest connected component with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Compute node features
    if compute_features:
        print("Computing node features...")
        degree_dict = dict(G.degree())
        
        # Calculate local clustering coefficients (can be time-consuming for large graphs)
        print("Computing clustering coefficients...")
        clustering_dict = nx.clustering(G)
        
        # Calculate PageRank
        print("Computing PageRank...")
        pagerank_dict = nx.pagerank(G, alpha=0.85, max_iter=100)
        
        # Assemble features
        features = pd.DataFrame({
            'node': list(G.nodes()),
            'degree': [degree_dict[n] for n in G.nodes()],
            'clustering': [clustering_dict[n] for n in G.nodes()],
            'pagerank': [pagerank_dict[n] for n in G.nodes()]
        })
        
        # Normalize features
        for col in features.columns:
            if col != 'node':
                features[col] = features[col] / features[col].max()
        
        return G, features
    
    return G, None

def create_pytorch_geometric_data(G, features):
    """Convert NetworkX graph and features to PyTorch Geometric Data object"""
    # Create node feature matrix
    x = torch.tensor(features.drop('node', axis=1).values, dtype=torch.float)
    
    # Create edge index
    edge_list = list(G.edges())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)
    
    return data

# GNN models
class EnhancedGCN(torch.nn.Module):
    """Enhanced Graph Convolutional Network with skip connections and batch normalization"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super(EnhancedGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First layer
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        # Second layer with skip connection
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2 = x2 + x1  # Skip connection
        
        # Output layer
        x3 = self.conv3(x2, edge_index)
        
        return x3

# Community detection
def detect_communities(embeddings, method='kmeans', k=5, **kwargs):
    """
    Apply different community detection methods to embeddings
    
    Args:
        embeddings: Node embeddings (numpy array)
        method: 'kmeans', 'dbscan', 'hierarchical', or 'louvain'
        k: Number of clusters (for kmeans and hierarchical)
        **kwargs: Additional parameters for clustering methods
        
    Returns:
        labels: Community assignments
        score: Silhouette score for clustering quality
    """
    if method == 'kmeans':
        model = KMeans(n_clusters=k, random_state=42, **kwargs)
        labels = model.fit_predict(embeddings)
        
    elif method == 'dbscan':
        # Automatically determine eps if not provided
        if 'eps' not in kwargs:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(50, len(embeddings)))
            nn.fit(embeddings)
            distances, _ = nn.kneighbors(embeddings)
            knee_point = np.sort(distances[:, -1])[int(0.8 * len(distances))]
            kwargs['eps'] = knee_point
            
        model = DBSCAN(**kwargs)
        labels = model.fit_predict(embeddings)
        
        # Handle outliers (-1 labels) by assigning to nearest cluster
        if -1 in labels:
            outliers = np.where(labels == -1)[0]
            valid_clusters = np.unique(labels[labels != -1])
            for i in outliers:
                # Find the closest non-outlier point
                valid_indices = np.where(labels != -1)[0]
                distances = np.linalg.norm(embeddings[i].reshape(1, -1) - embeddings[valid_indices], axis=1)
                closest = valid_indices[np.argmin(distances)]
                labels[i] = labels[closest]
    
    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=k, **kwargs)
        labels = model.fit_predict(embeddings)
    
    elif method == 'louvain':
        # For Louvain, we need to construct a graph from the embeddings
        # using a k-nearest neighbor approach
        n_neighbors = kwargs.get('n_neighbors', 10)
        A = kneighbors_graph(embeddings, n_neighbors, mode='distance')
        G = nx.from_scipy_sparse_matrix(A)
        
        # Apply Louvain algorithm
        partition = community_louvain.best_partition(G)
        labels = np.array([partition[n] for n in range(len(embeddings))])
    
    else:
        raise ValueError(f"Unknown community detection method: {method}")
    
    # Calculate clustering quality scores
    if len(np.unique(labels)) > 1:  # Can only calculate scores with multiple clusters
        silhouette = silhouette_score(embeddings, labels)
        calinski = calinski_harabasz_score(embeddings, labels)
        print(f"Clustering quality - Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.1f}")
    else:
        silhouette = 0
        calinski = 0
        print("Warning: Only one cluster found, scores will be 0")
    
    return labels, (silhouette, calinski)

# Polarization metrics
def calculate_polarization_metrics(G, community_labels):
    """
    Calculate various polarization metrics based on community structure
    
    Args:
        G: NetworkX graph
        community_labels: Node community assignments
        
    Returns:
        metrics: Dictionary of polarization metrics
    """
    # Create a map from node to community
    node_list = list(G.nodes())
    node_community = {node_list[i]: community_labels[i] for i in range(len(node_list))}
    
    # Calculate inter and intra-cluster edges
    inter_edges = sum(1 for u, v in G.edges() if node_community[u] != node_community[v])
    intra_edges = sum(1 for u, v in G.edges() if node_community[u] == node_community[v])
    total_edges = inter_edges + intra_edges
    
    # Basic polarization score (ratio of inter-cluster edges)
    polarization_score = inter_edges / total_edges if total_edges > 0 else 0
    
    # Calculate modularity
    modularity = 0
    m = G.number_of_edges()
    for u, v in G.edges():
        if node_community[u] == node_community[v]:
            modularity += 1 - (G.degree(u) * G.degree(v)) / (2 * m)
        else:
            modularity += 0 - (G.degree(u) * G.degree(v)) / (2 * m)
    modularity /= (2 * m)
    
    # Calculate community sizes
    communities = {}
    for node, comm in node_community.items():
        if comm not in communities:
            communities[comm] = 0
        communities[comm] += 1
    
    community_sizes = list(communities.values())
    
    # Calculate community size metrics
    size_mean = np.mean(community_sizes)
    size_std = np.std(community_sizes)
    size_min = np.min(community_sizes)
    size_max = np.max(community_sizes)
    size_imbalance = size_std / size_mean if size_mean > 0 else 0
    
    # Calculate E-I index (external-internal index) for each community
    ei_indices = {}
    for comm in set(node_community.values()):
        comm_nodes = {n for n, c in node_community.items() if c == comm}
        external = 0
        internal = 0
        for u, v in G.edges():
            if u in comm_nodes and v in comm_nodes:
                internal += 1
            elif u in comm_nodes or v in comm_nodes:
                external += 1
        ei_indices[comm] = (external - internal) / (external + internal) if (external + internal) > 0 else 0
    
    # Assemble all metrics
    metrics = {
        "polarization_score": polarization_score,
        "modularity": modularity,
        "inter_cluster_edges": inter_edges,
        "intra_cluster_edges": intra_edges,
        "total_edges": total_edges,
        "community_count": len(communities),
        "community_sizes": community_sizes,
        "community_size_mean": size_mean,
        "community_size_std": size_std,
        "community_size_min": size_min,
        "community_size_max": size_max,
        "community_size_imbalance": size_imbalance,
        "ei_indices": ei_indices,
        "average_ei_index": np.mean(list(ei_indices.values()))
    }
    
    return metrics

# Visualization
def plot_interactive_embeddings(embeddings, labels, pagerank, node_ids=None, search_node=None, 
                               title="Node Embeddings", width=800, height=600):
    """
    Create interactive plotly visualization of node embeddings
    
    Args:
        embeddings: 2D node embeddings (n_samples, 2)
        labels: Community labels for each node
        pagerank: PageRank values for sizing nodes
        node_ids: Original node IDs (if None, uses index as ID)
        search_node: Highlighted node (if any)
        title: Plot title
        width, height: Plot dimensions
    """
    # Create dataframe for plotting
    node_ids = node_ids if node_ids is not None else list(range(len(embeddings)))
    df = pd.DataFrame({
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'community': [f"Community {l}" for l in labels],
        'pagerank': pagerank,
        'node_id': node_ids
    })
    
    # Scale node sizes based on pagerank
    df['node_size'] = 5 + 100 * (df['pagerank'] - df['pagerank'].min()) / (df['pagerank'].max() - df['pagerank'].min() + 1e-10)
    
    # Create plot
    fig = px.scatter(
        df, x='x', y='y', color='community', size='node_size',
        hover_data=['node_id', 'pagerank'], 
        title=title,
        opacity=0.8,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    # Highlight search node if provided
    if search_node is not None and search_node in node_ids:
        idx = node_ids.index(search_node)
        fig.add_trace(
            go.Scatter(
                x=[embeddings[idx, 0]], 
                y=[embeddings[idx, 1]],
                mode='markers',
                marker=dict(
                    color='black',
                    size=20,
                    line=dict(width=2, color='white')
                ),
                name=f"Node {search_node}"
            )
        )
    
    # Update layout
    fig.update_layout(
        width=width,
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Main data processing function
@st.cache_resource
def generate_embeddings_cached(file_path, k):
    """Process the data and generate embeddings and communities"""
    # Load and preprocess data
    G, features = load_and_preprocess_data(file_path)
    
    # Create PyTorch Geometric data
    data = create_pytorch_geometric_data(G, features)
    
    # Get the pagerank for node sizing
    pagerank = features['pagerank'].values
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and initialize model
    model = EnhancedGCN(in_channels=3, hidden_channels=32, out_channels=8).to(device)
    data = data.to(device)
    
    # Forward pass (no training needed for this demo)
    model.eval()
    with torch.no_grad():
        out = model(data)
    
    # Get embeddings
    embeddings = out.cpu().numpy()
    
    # Apply UMAP for dimensionality reduction
    reducer = UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    
    # Perform community detection
    labels, quality_scores = detect_communities(reduced, method='kmeans', k=k)
    
    # Calculate polarization metrics
    metrics = calculate_polarization_metrics(G, labels)
    
    # Add silhouette score to metrics
    metrics['silhouette_score'] = quality_scores[0]
    
    return G, reduced, labels, pagerank, metrics, features['node'].values

# Load the data and generate embeddings
with st.spinner("Processing data and computing embeddings..."):
    G, embedding_2d, labels, pagerank, metrics, node_ids = generate_embeddings_cached(selected_file, k)

# View selection based on sidebar
if view == "Graph View":
    st.subheader("📊 Raw Graph Info")
    st.success(f"Loaded `{selected_label}` dataset with {G.number_of_edges():,} edges.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Nodes", G.number_of_nodes())
        st.metric("Density", f"{nx.density(G):.6f}")
    
    with col2:
        st.metric("Edges", G.number_of_edges())
        st.metric("Transitivity", f"{nx.transitivity(G):.4f}")
    
    # Show most central nodes
    st.subheader("Most Influential Pages (by PageRank)")
    top_nodes = sorted(list(zip(node_ids, pagerank)), key=lambda x: x[1], reverse=True)[:10]
    
    top_nodes_df = pd.DataFrame({
        "Rank": range(1, 11),
        "Node ID": [n[0] for n in top_nodes],
        "PageRank": [f"{n[1]:.6f}" for n in top_nodes]
    })
    
    st.dataframe(top_nodes_df)

elif view == "Embedding View":
    st.subheader("💫 Community Embedding Visualization")
    
    search_node_input = st.text_input("🔍 Search for Node ID (supports numeric IDs):")

    try:
        search_node = int(search_node_input)
    except ValueError:
        search_node = search_node_input if search_node_input else None

    node_ids_list = list(node_ids)

    # Plot the embeddings
    fig = plot_interactive_embeddings(
        embedding_2d, 
        labels, 
        pagerank, 
        node_ids_list,  # pass list here too
        search_node,
        f"{k} GCN Communities in {selected_label} Network"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Node details if searched and exists
    if search_node is not None and search_node in node_ids_list:
        idx = node_ids_list.index(search_node)

        st.info(f"""
        **Node ID**: {search_node}  
        **Community**: {labels[idx]}  
        **PageRank**: {pagerank[idx]:.6f}
        """)
    elif search_node is not None:
        st.warning(f"Node ID '{search_node}' not found in the graph.")



elif view == "Metrics Dashboard":
    st.subheader("📋 Business Insights & Polarization Score")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(
            "Polarization Score", 
            f"{metrics['polarization_score']:.3f}",
            help="0 = unified (most connections within communities), 1 = isolated (most connections across communities)"
        )
        
        st.metric(
            "Modularity", 
            f"{metrics['modularity']:.3f}",
            help="Higher modularity indicates stronger community structure"
        )
        
        st.metric(
            "Silhouette Score", 
            f"{metrics['silhouette_score']:.3f}",
            help="Measures how well-separated the communities are (-1 to 1, higher is better)"
        )
    
    with col2:
        # Create a pie chart of inter vs intra cluster edges
        labels_pie = ['Inter-cluster Edges', 'Intra-cluster Edges']
        values_pie = [metrics['inter_cluster_edges'], metrics['intra_cluster_edges']]
        
        fig = px.pie(
            values=values_pie,
            names=labels_pie,
            title=f"Edge Distribution in {selected_label} Network",
            color_discrete_sequence=['#ff9999', '#66b3ff']
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Community metrics
    st.markdown(f"""
    ### Network Statistics:
    
    - **Total Nodes**: {G.number_of_nodes()}  
    - **Total Edges**: {G.number_of_edges()}  
    - **Number of Communities**: {k}  
    - **Largest Community Size**: {metrics['community_size_max']}  
    - **Smallest Community Size**: {metrics['community_size_min']}  
    - **Inter-cluster Edges**: {metrics['inter_cluster_edges']}  
    - **Intra-cluster Edges**: {metrics['intra_cluster_edges']}  
    """)
    
    # Community size distribution
    st.subheader("Community Size Distribution")
    
    # Count nodes in each community
    community_counts = pd.Series(labels).value_counts().sort_index()
    
    community_df = pd.DataFrame({
        "Community": community_counts.index,
        "Node Count": community_counts.values
    })
    
    fig = px.bar(
        community_df,
        x='Community',
        y='Node Count',
        title=f"Community Sizes in {selected_label} Network",
        color='Node Count',
        color_continuous_scale='viridis'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # E-I indices
    if 'ei_indices' in metrics:
        st.subheader("External-Internal Index by Community")
        
        ei_indices = metrics['ei_indices']
        ei_df = pd.DataFrame({
            "Community": list(ei_indices.keys()),
            "E-I Index": list(ei_indices.values())
        })
        
        fig = px.bar(
            ei_df,
            x='Community',
            y='E-I Index',
            title="E-I Index by Community (higher means more external connections)",
            color='E-I Index',
            color_continuous_scale='RdBu'
        )
        
        # Add a horizontal line at 0
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(ei_indices) - 0.5,
            y0=0,
            y1=0,
            line=dict(color="black", width=1, dash="dash")
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif view == "Export Report":
    st.subheader("📤 Export Analysis")
    
    # Export options
    export_format = st.radio("Select export format:", ["CSV", "Interactive HTML", "PDF Report"])
    
    if export_format == "CSV":
        # Create CSV export
        csv_export = pd.DataFrame({
            "Node": node_ids,
            "Community": labels,
            "PageRank": pagerank
        })
        
        csv = csv_export.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{selected_label}_communities.csv">Download CSV Report</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Preview
        st.dataframe(csv_export.head(10))
    
    elif export_format == "Interactive HTML":
        # Create interactive Plotly visualization
        fig = plot_interactive_embeddings(
            embedding_2d, 
            labels, 
            pagerank, 
            node_ids, 
            None,
            f"{k} Communities in {selected_label} Network"
        )
        
        # Convert to HTML
        html_bytes = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Add some extra HTML for a better standalone experience
        html_str = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{selected_label} Network Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .metrics {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <h1>{k} Communities in {selected_label} Network</h1>
            <div class="metrics">
                <h3>Network Metrics:</h3>
                <p><b>Nodes:</b> {G.number_of_nodes()}</p>
                <p><b>Edges:</b> {G.number_of_edges()}</p>
                <p><b>Polarization Score:</b> {metrics['polarization_score']:.3f}</p>
                <p><b>Modularity:</b> {metrics['modularity']:.3f}</p>
                <p><b>Inter-cluster Edges:</b> {metrics['inter_cluster_edges']} ({metrics['inter_cluster_edges']/metrics['total_edges']*100:.1f}%)</p>
                <p><b>Intra-cluster Edges:</b> {metrics['intra_cluster_edges']} ({metrics['intra_cluster_edges']/metrics['total_edges']*100:.1f}%)</p>
            </div>
            {html_bytes}
        </body>
        </html>
        """
        
        b64 = base64.b64encode(html_str.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="{selected_label}_visualization.html">Download Interactive HTML</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Preview
        st.plotly_chart(fig, use_container_width=True)
    
    elif export_format == "PDF Report":
        st.info("Generating a comprehensive PDF report with visualizations and analysis...")
        
        # This would typically call the generate_pdf_report function
        # For this demo, we'll just show a placeholder
        st.success("PDF report generation would be implemented here.")
        st.markdown("The report would include:")
        st.markdown("- Network statistics and metrics")
        st.markdown("- Community visualization with annotations")
        st.markdown("- Polarization analysis")
        st.markdown("- Community distribution charts")
        st.markdown("- Most influential nodes")

elif view == "Comparative Analysis":
    st.subheader("🔍 Comparative Network Analysis")
    st.info("This view allows you to compare different networks or the same network with different parameters.")
    
    # Select networks to compare
    networks_to_compare = st.multiselect(
        "Select networks to compare:",
        list(DATASETS.keys()),
        default=[selected_label]
    )
    
    if not networks_to_compare:
        st.warning("Please select at least one network to analyze.")
    else:
        # Number of communities for comparison
        comp_k = st.slider("Number of communities for comparison:", 2, 10, k)
        
        # Process data for each selected network
        with st.spinner("Processing multiple networks..."):
            results = []
            
            for network_name in networks_to_compare:
                file_path = DATASETS[network_name]
                G_net, embedding_net, labels_net, pagerank_net, metrics_net, node_ids_net = generate_embeddings_cached(file_path, comp_k)
                
                results.append({
                    'name': network_name,
                    'G': G_net,
                    'embedding': embedding_net,
                    'labels': labels_net,
                    'pagerank': pagerank_net,
                    'metrics': metrics_net,
                    'node_ids': node_ids_net
                })
        
        # Create comparison visualizations
        
        # 1. Polarization score comparison
        pol_scores = [r['metrics']['polarization_score'] for r in results]
        mod_scores = [r['metrics']['modularity'] for r in results]
        
        comparison_df = pd.DataFrame({
            'Network': [r['name'] for r in results],
            'Polarization Score': pol_scores,
            'Modularity': mod_scores,
            'Nodes': [r['G'].number_of_nodes() for r in results],
            'Edges': [r['G'].number_of_edges() for r in results],
            'Density': [nx.density(r['G']) for r in results],
            'Transitivity': [nx.transitivity(r['G']) for r in results],
            'Avg Community Size': [sum(r['metrics']['community_sizes'])/len(r['metrics']['community_sizes']) for r in results],
            'Size Imbalance': [r['metrics']['community_size_imbalance'] for r in results]
        })
        
        st.dataframe(comparison_df)
        
        # 2. Metrics comparison chart
        fig = px.bar(
            comparison_df,
            x='Network',
            y=['Polarization Score', 'Modularity', 'Transitivity'],
            title=f"Network Metrics Comparison ({comp_k} communities)",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. Edge distribution comparison
        edge_data = []
        for r in results:
            edge_data.append({
                'Network': r['name'],
                'Edge Type': 'Inter-cluster',
                'Count': r['metrics']['inter_cluster_edges']
            })
            edge_data.append({
                'Network': r['name'],
                'Edge Type': 'Intra-cluster',
                'Count': r['metrics']['intra_cluster_edges']
            })
        
        edge_df = pd.DataFrame(edge_data)
        
        fig = px.bar(
            edge_df,
            x='Network',
            y='Count',
            color='Edge Type',
            title="Edge Distribution Comparison",
            barmode='stack'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. Community size distribution
        comm_sizes = []
        
        for r in results:
            for i, size in enumerate(r['metrics']['community_sizes']):
                comm_sizes.append({
                    'Network': r['name'],
                    'Community': i,
                    'Size': size
                })
        
        comm_df = pd.DataFrame(comm_sizes)
        
        fig = px.box(
            comm_df,
            x='Network',
            y='Size',
            title="Community Size Distribution",
            points='all'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 5. Side-by-side embedding visualizations
        st.subheader("Community Structure Visualizations")
        
        cols = st.columns(min(3, len(results)))
        
        for i, result in enumerate(results):
            with cols[i % len(cols)]:
                fig, ax = plt.subplots(figsize=(5, 5))
                scatter = ax.scatter(
                    result['embedding'][:, 0],
                    result['embedding'][:, 1],
                    c=result['labels'],
                    s=10,
                    alpha=0.7,
                    cmap='tab10'
                )
                
                ax.set_title(f"{result['name']} ({comp_k} communities)")
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")
                plt.tight_layout()
                
                st.pyplot(fig)
                
                st.metric("Polarization", f"{result['metrics']['polarization_score']:.3f}")
        
        # 6. Network statistics comparison
        st.subheader("Detailed Network Statistics")
        
        # Create detailed comparison table
        detailed_stats = []
        
        for r in results:
            detailed_stats.append({
                'Network': r['name'],
                'Nodes': r['G'].number_of_nodes(),
                'Edges': r['G'].number_of_edges(),
                'Density': nx.density(r['G']),
                'Transitivity': nx.transitivity(r['G']),
                'Avg. Clustering': nx.average_clustering(r['G']),
                'Polarization': r['metrics']['polarization_score'],
                'Modularity': r['metrics']['modularity'],
                'Communities': comp_k,
                'Inter-cluster Edges': r['metrics']['inter_cluster_edges'],
                'Intra-cluster Edges': r['metrics']['intra_cluster_edges'],
                'Largest Community': r['metrics']['community_size_max'],
                'Smallest Community': r['metrics']['community_size_min']
            })
        
        detailed_df = pd.DataFrame(detailed_stats)
        st.dataframe(detailed_df)
        
        # 7. Correlation between transitivity and polarization
        if len(results) >= 3:  # Only show if we have enough data points
            st.subheader("Transitivity vs. Polarization")
            
            fig = px.scatter(
                comparison_df,
                x='Transitivity',
                y='Polarization Score',
                text='Network',
                title="Relationship between Transitivity and Polarization",
                size='Nodes',
                color='Modularity',
                trendline='ols'
            )
            
            fig.update_traces(textposition='top center')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation
            if comparison_df['Transitivity'].corr(comparison_df['Polarization Score']) < 0:
                st.info("Networks with higher transitivity (more 'friends of friends are friends') tend to have lower polarization scores, suggesting tighter, more internally-connected communities.")
            else:
                st.info("Interestingly, transitivity and polarization show a positive correlation in these networks, which differs from typical network theory expectations.")

# Footer
st.markdown("---")
st.markdown("🧠 **GNN-Based Community & Polarization Detector** | Developed by Team")