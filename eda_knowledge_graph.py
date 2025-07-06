#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Dermatology Knowledge Graph

This script performs comprehensive analysis of the constructed knowledge graph,
including statistical analysis, network analysis, and visualizations.

Usage:
    python eda_knowledge_graph.py [graph_file.graphml]
"""

import os
import sys
import argparse
import logging
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Any
import warnings
warnings.filterwarnings('ignore')

# For advanced network analysis
try:
    import community as community_louvain
    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False
    print("Warning: python-louvain not installed. Community detection will be skipped.")

# For layout algorithms
try:
    import pygraphviz
    HAS_PYGRAPHVIZ = True
except ImportError:
    HAS_PYGRAPHVIZ = False
    print("Warning: pygraphviz not installed. Some layout algorithms will be limited.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for plots
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')
sns.set_palette("husl")


class KnowledgeGraphEDA:
    """
    Comprehensive EDA class for analyzing knowledge graphs.
    """
    
    def __init__(self, graph_path: str):
        """
        Initialize the EDA with a graph file.
        
        Args:
            graph_path: Path to the GraphML file
        """
        self.graph_path = graph_path
        self.graph = None
        self.report = {}
        self.figures = []
        
    def load_graph(self):
        """Load the knowledge graph from GraphML file."""
        logger.info(f"Loading graph from: {self.graph_path}")
        try:
            self.graph = nx.read_graphml(self.graph_path)
            logger.info(f"✓ Graph loaded successfully")
            logger.info(f"  Nodes: {len(self.graph.nodes):,}")
            logger.info(f"  Edges: {len(self.graph.edges):,}")
        except Exception as e:
            logger.error(f"❌ Failed to load graph: {e}")
            raise
            
    def basic_statistics(self):
        """Calculate basic graph statistics."""
        logger.info("Calculating basic graph statistics...")
        
        # Basic counts
        num_nodes = len(self.graph.nodes)
        num_edges = len(self.graph.edges)
        
        # Density
        density = nx.density(self.graph)
        
        # Connectivity
        if nx.is_directed(self.graph):
            is_connected = nx.is_weakly_connected(self.graph)
            num_components = nx.number_weakly_connected_components(self.graph)
            largest_component = max(nx.weakly_connected_components(self.graph), key=len)
        else:
            is_connected = nx.is_connected(self.graph)
            num_components = nx.number_connected_components(self.graph)
            largest_component = max(nx.connected_components(self.graph), key=len)
            
        largest_component_size = len(largest_component)
        
        # Degree statistics
        degrees = dict(self.graph.degree())
        degree_values = list(degrees.values())
        
        stats = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'is_connected': is_connected,
            'num_components': num_components,
            'largest_component_size': largest_component_size,
            'largest_component_fraction': largest_component_size / num_nodes,
            'avg_degree': np.mean(degree_values),
            'median_degree': np.median(degree_values),
            'max_degree': max(degree_values),
            'min_degree': min(degree_values),
            'std_degree': np.std(degree_values),
            'is_directed': nx.is_directed(self.graph),
            'is_multigraph': isinstance(self.graph, (nx.MultiGraph, nx.MultiDiGraph)),
        }
        
        self.report['basic_stats'] = stats
        
        # Print summary
        print("\n" + "="*60)
        print("BASIC GRAPH STATISTICS")
        print("="*60)
        print(f"Nodes: {stats['num_nodes']:,}")
        print(f"Edges: {stats['num_edges']:,}")
        print(f"Density: {stats['density']:.6f}")
        print(f"Connected: {stats['is_connected']}")
        print(f"Components: {stats['num_components']:,}")
        print(f"Largest component: {stats['largest_component_size']:,} ({stats['largest_component_fraction']:.1%})")
        print(f"Average degree: {stats['avg_degree']:.2f}")
        print(f"Degree range: {stats['min_degree']} - {stats['max_degree']}")
        print(f"Directed: {stats['is_directed']}")
        
    def analyze_node_attributes(self):
        """Analyze node attributes and metadata."""
        logger.info("Analyzing node attributes...")
        
        # Collect all node attributes
        node_attrs = defaultdict(list)
        attribute_stats = {}
        
        for node, data in self.graph.nodes(data=True):
            for key, value in data.items():
                node_attrs[key].append(value)
        
        # Analyze each attribute
        for attr_name, values in node_attrs.items():
            if not values:
                continue
                
            # Count unique values
            unique_values = set(str(v) for v in values if v is not None)
            value_counts = Counter(str(v) for v in values if v is not None)
            
            attribute_stats[attr_name] = {
                'total_values': len(values),
                'unique_values': len(unique_values),
                'missing_values': sum(1 for v in values if v is None or str(v).strip() == ''),
                'most_common': value_counts.most_common(10),
                'data_type': type(values[0]).__name__ if values else 'unknown'
            }
        
        self.report['node_attributes'] = attribute_stats
        
        # Print summary
        print("\n" + "="*60)
        print("NODE ATTRIBUTES ANALYSIS")
        print("="*60)
        
        for attr_name, stats in attribute_stats.items():
            print(f"\n{attr_name}:")
            print(f"  Total values: {stats['total_values']:,}")
            print(f"  Unique values: {stats['unique_values']:,}")
            print(f"  Missing values: {stats['missing_values']:,}")
            print(f"  Data type: {stats['data_type']}")
            
            if stats['most_common'] and len(stats['most_common']) < 50:
                print(f"  Most common values:")
                for value, count in stats['most_common'][:10]:
                    print(f"    {value}: {count:,}")
                    
    def analyze_semantic_types(self):
        """Analyze semantic types distribution."""
        logger.info("Analyzing semantic types...")
        
        # Get semantic type information
        semantic_types = []
        for node, data in self.graph.nodes(data=True):
            if 'semantic_type' in data and data['semantic_type']:
                semantic_types.append(data['semantic_type'])
        
        if not semantic_types:
            print("No semantic type information found in nodes.")
            return
            
        # Count semantic types
        semantic_type_counts = Counter(semantic_types)
        
        # Store in report
        self.report['semantic_types'] = {
            'total_with_semantic_type': len(semantic_types),
            'unique_semantic_types': len(semantic_type_counts),
            'distribution': dict(semantic_type_counts)
        }
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot of top semantic types
        top_types = semantic_type_counts.most_common(20)
        types, counts = zip(*top_types)
        
        ax1.bar(range(len(types)), counts)
        ax1.set_xticks(range(len(types)))
        ax1.set_xticklabels(types, rotation=45, ha='right')
        ax1.set_title('Top 20 Semantic Types by Count')
        ax1.set_xlabel('Semantic Type')
        ax1.set_ylabel('Count')
        
        # Pie chart of top 10 semantic types
        top_10_types = semantic_type_counts.most_common(10)
        types_pie, counts_pie = zip(*top_10_types)
        
        ax2.pie(counts_pie, labels=types_pie, autopct='%1.1f%%')
        ax2.set_title('Top 10 Semantic Types Distribution')
        
        plt.tight_layout()
        plt.savefig('semantic_types_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        self.figures.append('semantic_types_analysis.png')
        
        # Print summary
        print("\n" + "="*60)
        print("SEMANTIC TYPES ANALYSIS")
        print("="*60)
        print(f"Nodes with semantic types: {len(semantic_types):,}")
        print(f"Unique semantic types: {len(semantic_type_counts):,}")
        print(f"\nTop 15 semantic types:")
        for sem_type, count in semantic_type_counts.most_common(15):
            percentage = (count / len(semantic_types)) * 100
            print(f"  {sem_type}: {count:,} ({percentage:.1f}%)")
            
    def analyze_edge_attributes(self):
        """Analyze edge attributes and relationship types."""
        logger.info("Analyzing edge attributes...")
        
        # Collect edge attributes
        edge_attrs = defaultdict(list)
        
        for u, v, data in self.graph.edges(data=True):
            for key, value in data.items():
                edge_attrs[key].append(value)
        
        # Analyze relationship types
        if 'relationship_type' in edge_attrs:
            rel_types = edge_attrs['relationship_type']
            rel_type_counts = Counter(rel_types)
            
            self.report['relationship_types'] = {
                'total_relationships': len(rel_types),
                'unique_rel_types': len(rel_type_counts),
                'distribution': dict(rel_type_counts)
            }
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Bar plot of relationship types
            top_rels = rel_type_counts.most_common(20)
            rel_names, rel_counts = zip(*top_rels)
            
            ax1.bar(range(len(rel_names)), rel_counts)
            ax1.set_xticks(range(len(rel_names)))
            ax1.set_xticklabels(rel_names, rotation=45, ha='right')
            ax1.set_title('Top 20 Relationship Types by Count')
            ax1.set_xlabel('Relationship Type')
            ax1.set_ylabel('Count')
            
            # Pie chart of top 10 relationship types
            top_10_rels = rel_type_counts.most_common(10)
            rel_names_pie, rel_counts_pie = zip(*top_10_rels)
            
            ax2.pie(rel_counts_pie, labels=rel_names_pie, autopct='%1.1f%%')
            ax2.set_title('Top 10 Relationship Types Distribution')
            
            plt.tight_layout()
            plt.savefig('relationship_types_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            self.figures.append('relationship_types_analysis.png')
            
            # Print summary
            print("\n" + "="*60)
            print("RELATIONSHIP TYPES ANALYSIS")
            print("="*60)
            print(f"Total relationships: {len(rel_types):,}")
            print(f"Unique relationship types: {len(rel_type_counts):,}")
            print(f"\nTop 15 relationship types:")
            for rel_type, count in rel_type_counts.most_common(15):
                percentage = (count / len(rel_types)) * 100
                print(f"  {rel_type}: {count:,} ({percentage:.1f}%)")
                
    def analyze_degree_distribution(self):
        """Analyze degree distribution of nodes."""
        logger.info("Analyzing degree distribution...")
        
        # Get degree information
        degrees = dict(self.graph.degree())
        degree_values = list(degrees.values())
        
        # Calculate degree statistics
        degree_stats = {
            'mean': np.mean(degree_values),
            'median': np.median(degree_values),
            'std': np.std(degree_values),
            'min': min(degree_values),
            'max': max(degree_values),
            'q25': np.percentile(degree_values, 25),
            'q75': np.percentile(degree_values, 75),
        }
        
        # Find high-degree nodes (hubs)
        high_degree_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
        
        self.report['degree_analysis'] = {
            'degree_stats': degree_stats,
            'high_degree_nodes': high_degree_nodes
        }
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Degree distribution histogram
        axes[0, 0].hist(degree_values, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Degree Distribution')
        axes[0, 0].set_xlabel('Degree')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_yscale('log')
        
        # Log-log plot for power law analysis
        degree_counts = Counter(degree_values)
        degrees_sorted = sorted(degree_counts.keys())
        counts = [degree_counts[d] for d in degrees_sorted]
        
        axes[0, 1].loglog(degrees_sorted, counts, 'bo-', alpha=0.7)
        axes[0, 1].set_title('Degree Distribution (Log-Log)')
        axes[0, 1].set_xlabel('Degree (log scale)')
        axes[0, 1].set_ylabel('Count (log scale)')
        
        # Box plot of degree distribution
        axes[1, 0].boxplot(degree_values, vert=True)
        axes[1, 0].set_title('Degree Distribution Box Plot')
        axes[1, 0].set_ylabel('Degree')
        
        # Top nodes by degree
        top_nodes = high_degree_nodes[:15]
        node_names = [node[:20] + '...' if len(node) > 20 else node for node, _ in top_nodes]
        node_degrees = [degree for _, degree in top_nodes]
        
        axes[1, 1].barh(range(len(node_names)), node_degrees)
        axes[1, 1].set_yticks(range(len(node_names)))
        axes[1, 1].set_yticklabels(node_names)
        axes[1, 1].set_title('Top 15 Nodes by Degree')
        axes[1, 1].set_xlabel('Degree')
        
        plt.tight_layout()
        plt.savefig('degree_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        self.figures.append('degree_distribution_analysis.png')
        
        # Print summary
        print("\n" + "="*60)
        print("DEGREE DISTRIBUTION ANALYSIS")
        print("="*60)
        print(f"Mean degree: {degree_stats['mean']:.2f}")
        print(f"Median degree: {degree_stats['median']:.2f}")
        print(f"Standard deviation: {degree_stats['std']:.2f}")
        print(f"Min degree: {degree_stats['min']}")
        print(f"Max degree: {degree_stats['max']}")
        print(f"25th percentile: {degree_stats['q25']:.2f}")
        print(f"75th percentile: {degree_stats['q75']:.2f}")
        
        print(f"\nTop 10 highest degree nodes:")
        for node, degree in high_degree_nodes[:10]:
            node_name = self.graph.nodes[node].get('name', node)
            print(f"  {node} ({node_name[:50]}...): {degree}")
            
    def analyze_connectivity(self):
        """Analyze graph connectivity and components."""
        logger.info("Analyzing graph connectivity...")
        
        # Connected components analysis
        if nx.is_directed(self.graph):
            weak_components = list(nx.weakly_connected_components(self.graph))
            strong_components = list(nx.strongly_connected_components(self.graph))
            
            component_sizes_weak = [len(comp) for comp in weak_components]
            component_sizes_strong = [len(comp) for comp in strong_components]
            
            connectivity_stats = {
                'num_weak_components': len(weak_components),
                'num_strong_components': len(strong_components),
                'largest_weak_component': max(component_sizes_weak),
                'largest_strong_component': max(component_sizes_strong),
                'weak_component_sizes': component_sizes_weak,
                'strong_component_sizes': component_sizes_strong,
            }
        else:
            components = list(nx.connected_components(self.graph))
            component_sizes = [len(comp) for comp in components]
            
            connectivity_stats = {
                'num_components': len(components),
                'largest_component': max(component_sizes),
                'component_sizes': component_sizes,
            }
        
        self.report['connectivity'] = connectivity_stats
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Component size distribution
        if nx.is_directed(self.graph):
            axes[0].hist(component_sizes_weak, bins=min(50, len(component_sizes_weak)), 
                        alpha=0.7, label='Weak Components')
            axes[0].hist(component_sizes_strong, bins=min(50, len(component_sizes_strong)), 
                        alpha=0.7, label='Strong Components')
            axes[0].legend()
            axes[0].set_title('Component Size Distribution')
        else:
            axes[0].hist(component_sizes, bins=min(50, len(component_sizes)), alpha=0.7)
            axes[0].set_title('Connected Component Size Distribution')
        
        axes[0].set_xlabel('Component Size')
        axes[0].set_ylabel('Frequency')
        axes[0].set_yscale('log')
        
        # Largest components analysis
        if nx.is_directed(self.graph):
            sizes_to_plot = sorted(component_sizes_weak, reverse=True)[:20]
            axes[1].bar(range(len(sizes_to_plot)), sizes_to_plot)
            axes[1].set_title('Top 20 Largest Weak Components')
        else:
            sizes_to_plot = sorted(component_sizes, reverse=True)[:20]
            axes[1].bar(range(len(sizes_to_plot)), sizes_to_plot)
            axes[1].set_title('Top 20 Largest Components')
        
        axes[1].set_xlabel('Component Rank')
        axes[1].set_ylabel('Component Size')
        
        plt.tight_layout()
        plt.savefig('connectivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        self.figures.append('connectivity_analysis.png')
        
        # Print summary
        print("\n" + "="*60)
        print("CONNECTIVITY ANALYSIS")
        print("="*60)
        
        if nx.is_directed(self.graph):
            print(f"Weakly connected components: {connectivity_stats['num_weak_components']:,}")
            print(f"Strongly connected components: {connectivity_stats['num_strong_components']:,}")
            print(f"Largest weak component: {connectivity_stats['largest_weak_component']:,}")
            print(f"Largest strong component: {connectivity_stats['largest_strong_component']:,}")
        else:
            print(f"Connected components: {connectivity_stats['num_components']:,}")
            print(f"Largest component: {connectivity_stats['largest_component']:,}")
            
    def analyze_centrality(self):
        """Calculate centrality measures for important nodes."""
        logger.info("Analyzing node centrality...")
        
        # For very large graphs, work with the largest connected component
        if nx.is_directed(self.graph):
            largest_cc = max(nx.weakly_connected_components(self.graph), key=len)
        else:
            largest_cc = max(nx.connected_components(self.graph), key=len)
            
        # Create subgraph of largest component
        if len(largest_cc) < len(self.graph):
            logger.info(f"Working with largest component ({len(largest_cc):,} nodes)")
            subgraph = self.graph.subgraph(largest_cc)
        else:
            subgraph = self.graph
        
        centrality_measures = {}
        
        # Degree centrality (fast)
        logger.info("  Computing degree centrality...")
        degree_centrality = nx.degree_centrality(subgraph)
        centrality_measures['degree'] = degree_centrality
        
        # Betweenness centrality (slower, sample if too large)
        if len(subgraph) <= 10000:
            logger.info("  Computing betweenness centrality...")
            betweenness_centrality = nx.betweenness_centrality(subgraph)
            centrality_measures['betweenness'] = betweenness_centrality
        else:
            logger.info("  Computing betweenness centrality (sampled)...")
            sample_size = min(1000, len(subgraph) // 10)
            betweenness_centrality = nx.betweenness_centrality(subgraph, k=sample_size)
            centrality_measures['betweenness'] = betweenness_centrality
        
        # Closeness centrality (slower, sample if too large)
        if len(subgraph) <= 5000:
            logger.info("  Computing closeness centrality...")
            closeness_centrality = nx.closeness_centrality(subgraph)
            centrality_measures['closeness'] = closeness_centrality
        else:
            logger.info("  Skipping closeness centrality (graph too large)")
        
        # Eigenvector centrality (can be unstable, use with caution)
        try:
            if len(subgraph) <= 20000:
                logger.info("  Computing eigenvector centrality...")
                eigenvector_centrality = nx.eigenvector_centrality(subgraph, max_iter=1000)
                centrality_measures['eigenvector'] = eigenvector_centrality
        except:
            logger.info("  Eigenvector centrality computation failed")
        
        # Page rank (works well for directed graphs)
        logger.info("  Computing PageRank...")
        pagerank = nx.pagerank(subgraph)
        centrality_measures['pagerank'] = pagerank
        
        # Find top nodes for each centrality measure
        top_nodes = {}
        for measure_name, centrality_dict in centrality_measures.items():
            top_nodes[measure_name] = sorted(centrality_dict.items(), 
                                           key=lambda x: x[1], reverse=True)[:20]
        
        self.report['centrality'] = {
            'measures': centrality_measures,
            'top_nodes': top_nodes
        }
        
        # Print summary
        print("\n" + "="*60)
        print("CENTRALITY ANALYSIS")
        print("="*60)
        
        for measure_name, top_list in top_nodes.items():
            print(f"\nTop 10 nodes by {measure_name} centrality:")
            for i, (node, score) in enumerate(top_list[:10], 1):
                node_name = self.graph.nodes[node].get('name', node)
                print(f"  {i:2d}. {node} ({node_name[:40]}...): {score:.6f}")
                
    def analyze_clustering(self):
        """Analyze clustering properties of the graph."""
        logger.info("Analyzing clustering properties...")
        
        # Calculate clustering coefficients
        if nx.is_directed(self.graph):
            # For directed graphs, use the undirected version
            undirected_graph = self.graph.to_undirected()
            clustering_coeffs = nx.clustering(undirected_graph)
        else:
            clustering_coeffs = nx.clustering(self.graph)
        
        # Calculate average clustering coefficient
        avg_clustering = nx.average_clustering(self.graph.to_undirected() if nx.is_directed(self.graph) else self.graph)
        
        # Transitivity (global clustering coefficient)
        transitivity = nx.transitivity(self.graph.to_undirected() if nx.is_directed(self.graph) else self.graph)
        
        # Find nodes with high clustering coefficients
        high_clustering_nodes = sorted(clustering_coeffs.items(), key=lambda x: x[1], reverse=True)[:20]
        
        clustering_stats = {
            'average_clustering': avg_clustering,
            'transitivity': transitivity,
            'clustering_coeffs': clustering_coeffs,
            'high_clustering_nodes': high_clustering_nodes
        }
        
        self.report['clustering'] = clustering_stats
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Clustering coefficient distribution
        clustering_values = list(clustering_coeffs.values())
        axes[0].hist(clustering_values, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_title('Clustering Coefficient Distribution')
        axes[0].set_xlabel('Clustering Coefficient')
        axes[0].set_ylabel('Frequency')
        
        # Scatter plot: degree vs clustering coefficient
        degrees = dict(self.graph.degree())
        degree_values = [degrees[node] for node in clustering_coeffs.keys()]
        clustering_values = [clustering_coeffs[node] for node in clustering_coeffs.keys()]
        
        axes[1].scatter(degree_values, clustering_values, alpha=0.5)
        axes[1].set_title('Degree vs Clustering Coefficient')
        axes[1].set_xlabel('Degree')
        axes[1].set_ylabel('Clustering Coefficient')
        axes[1].set_xscale('log')
        
        plt.tight_layout()
        plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        self.figures.append('clustering_analysis.png')
        
        # Print summary
        print("\n" + "="*60)
        print("CLUSTERING ANALYSIS")
        print("="*60)
        print(f"Average clustering coefficient: {avg_clustering:.6f}")
        print(f"Transitivity (global clustering): {transitivity:.6f}")
        
        print(f"\nTop 10 nodes with highest clustering coefficient:")
        for node, coeff in high_clustering_nodes[:10]:
            node_name = self.graph.nodes[node].get('name', node)
            print(f"  {node} ({node_name[:40]}...): {coeff:.6f}")
            
    def community_detection(self):
        """Perform community detection analysis."""
        if not HAS_COMMUNITY:
            logger.info("Skipping community detection (python-louvain not installed)")
            return
            
        logger.info("Performing community detection...")
        
        # Convert to undirected if necessary
        if nx.is_directed(self.graph):
            undirected_graph = self.graph.to_undirected()
        else:
            undirected_graph = self.graph
            
        # For very large graphs, work with a sample or the largest component
        if len(undirected_graph) > 100000:
            logger.info("Graph is very large, working with largest connected component...")
            largest_cc = max(nx.connected_components(undirected_graph), key=len)
            undirected_graph = undirected_graph.subgraph(largest_cc)
        
        # Perform community detection using Louvain algorithm
        try:
            communities = community_louvain.best_partition(undirected_graph)
            
            # Analyze communities
            community_sizes = Counter(communities.values())
            num_communities = len(community_sizes)
            
            # Calculate modularity
            modularity = community_louvain.modularity(communities, undirected_graph)
            
            community_stats = {
                'num_communities': num_communities,
                'modularity': modularity,
                'community_sizes': dict(community_sizes),
                'communities': communities
            }
            
            self.report['communities'] = community_stats
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Community size distribution
            sizes = list(community_sizes.values())
            axes[0].hist(sizes, bins=min(50, len(sizes)), alpha=0.7, edgecolor='black')
            axes[0].set_title('Community Size Distribution')
            axes[0].set_xlabel('Community Size')
            axes[0].set_ylabel('Frequency')
            axes[0].set_yscale('log')
            
            # Top communities by size
            top_communities = community_sizes.most_common(20)
            comm_ids, comm_sizes = zip(*top_communities)
            
            axes[1].bar(range(len(comm_ids)), comm_sizes)
            axes[1].set_title('Top 20 Communities by Size')
            axes[1].set_xlabel('Community Rank')
            axes[1].set_ylabel('Community Size')
            
            plt.tight_layout()
            plt.savefig('community_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            self.figures.append('community_analysis.png')
            
            # Print summary
            print("\n" + "="*60)
            print("COMMUNITY DETECTION ANALYSIS")
            print("="*60)
            print(f"Number of communities: {num_communities:,}")
            print(f"Modularity: {modularity:.6f}")
            print(f"Largest community size: {max(sizes):,}")
            print(f"Smallest community size: {min(sizes):,}")
            print(f"Average community size: {np.mean(sizes):.2f}")
            
            print(f"\nTop 10 largest communities:")
            for i, (comm_id, size) in enumerate(top_communities[:10], 1):
                print(f"  {i:2d}. Community {comm_id}: {size:,} nodes")
                
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            
    def analyze_paths(self):
        """Analyze shortest paths and graph diameter."""
        logger.info("Analyzing path properties...")
        
        # For large graphs, work with the largest connected component
        if nx.is_directed(self.graph):
            if not nx.is_weakly_connected(self.graph):
                largest_cc = max(nx.weakly_connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc)
            else:
                subgraph = self.graph
        else:
            if not nx.is_connected(self.graph):
                largest_cc = max(nx.connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc)
            else:
                subgraph = self.graph
        
        # Sample nodes for path analysis if graph is too large
        if len(subgraph) > 10000:
            logger.info("Graph is large, sampling nodes for path analysis...")
            sample_nodes = np.random.choice(list(subgraph.nodes()), size=1000, replace=False)
            sample_graph = subgraph.subgraph(sample_nodes)
        else:
            sample_graph = subgraph
            
        try:
            # Calculate shortest path lengths
            if nx.is_directed(sample_graph):
                path_lengths = []
                for source in sample_graph.nodes():
                    lengths = nx.single_source_shortest_path_length(sample_graph, source)
                    path_lengths.extend(lengths.values())
            else:
                path_lengths = []
                for source in sample_graph.nodes():
                    lengths = nx.single_source_shortest_path_length(sample_graph, source)
                    path_lengths.extend(lengths.values())
            
            # Remove zero-length paths (self-loops)
            path_lengths = [length for length in path_lengths if length > 0]
            
            if path_lengths:
                path_stats = {
                    'average_path_length': np.mean(path_lengths),
                    'median_path_length': np.median(path_lengths),
                    'max_path_length': max(path_lengths),
                    'min_path_length': min(path_lengths),
                    'path_length_distribution': Counter(path_lengths)
                }
                
                self.report['path_analysis'] = path_stats
                
                # Create visualization
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Path length distribution
                lengths = list(path_stats['path_length_distribution'].keys())
                counts = list(path_stats['path_length_distribution'].values())
                
                axes[0].bar(lengths, counts)
                axes[0].set_title('Shortest Path Length Distribution')
                axes[0].set_xlabel('Path Length')
                axes[0].set_ylabel('Frequency')
                
                # Cumulative distribution
                sorted_lengths = sorted(path_lengths)
                cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
                
                axes[1].plot(sorted_lengths, cumulative)
                axes[1].set_title('Cumulative Distribution of Path Lengths')
                axes[1].set_xlabel('Path Length')
                axes[1].set_ylabel('Cumulative Probability')
                
                plt.tight_layout()
                plt.savefig('path_analysis.png', dpi=300, bbox_inches='tight')
                plt.show()
                self.figures.append('path_analysis.png')
                
                # Print summary
                print("\n" + "="*60)
                print("PATH ANALYSIS")
                print("="*60)
                print(f"Average path length: {path_stats['average_path_length']:.2f}")
                print(f"Median path length: {path_stats['median_path_length']:.2f}")
                print(f"Maximum path length (diameter): {path_stats['max_path_length']}")
                print(f"Minimum path length: {path_stats['min_path_length']}")
                
                print(f"\nPath length distribution:")
                for length, count in sorted(path_stats['path_length_distribution'].items()):
                    percentage = (count / len(path_lengths)) * 100
                    print(f"  Length {length}: {count:,} ({percentage:.1f}%)")
                    
        except Exception as e:
            logger.error(f"Path analysis failed: {e}")
            
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        logger.info("Generating summary report...")
        
        # Create summary
        summary = {
            'graph_file': self.graph_path,
            'file_size_mb': round(os.path.getsize(self.graph_path) / (1024*1024), 2),
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'total_figures_generated': len(self.figures),
            'figures': self.figures
        }
        
        # Add all analysis results
        summary.update(self.report)
        
        # Save detailed report as JSON
        import json
        with open('kg_eda_report.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(summary), f, indent=2)
        
        # Print final summary
        print("\n" + "="*80)
        print("KNOWLEDGE GRAPH EDA SUMMARY REPORT")
        print("="*80)
        print(f"Graph file: {self.graph_path}")
        print(f"File size: {summary['file_size_mb']:,} MB")
        print(f"Analysis completed: {summary['analysis_timestamp']}")
        print(f"Figures generated: {len(self.figures)}")
        
        if 'basic_stats' in self.report:
            stats = self.report['basic_stats']
            print(f"\nGraph Structure:")
            print(f"  • Nodes: {stats['num_nodes']:,}")
            print(f"  • Edges: {stats['num_edges']:,}")
            print(f"  • Density: {stats['density']:.6f}")
            print(f"  • Components: {stats['num_components']:,}")
            print(f"  • Largest component: {stats['largest_component_size']:,} ({stats['largest_component_fraction']:.1%})")
            
        if 'semantic_types' in self.report:
            sem_stats = self.report['semantic_types']
            print(f"\nSemantic Types:")
            print(f"  • Nodes with semantic types: {sem_stats['total_with_semantic_type']:,}")
            print(f"  • Unique semantic types: {sem_stats['unique_semantic_types']:,}")
            
        if 'relationship_types' in self.report:
            rel_stats = self.report['relationship_types']
            print(f"\nRelationship Types:")
            print(f"  • Total relationships: {rel_stats['total_relationships']:,}")
            print(f"  • Unique relationship types: {rel_stats['unique_rel_types']:,}")
            
        if 'communities' in self.report:
            comm_stats = self.report['communities']
            print(f"\nCommunity Structure:")
            print(f"  • Number of communities: {comm_stats['num_communities']:,}")
            print(f"  • Modularity: {comm_stats['modularity']:.6f}")
            
        print(f"\nOutput Files:")
        print(f"  • Detailed report: kg_eda_report.json")
        for figure in self.figures:
            print(f"  • Figure: {figure}")
            
        print("\n" + "="*80)
        print("EDA COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    def run_full_analysis(self):
        """Run the complete EDA analysis pipeline."""
        logger.info("Starting comprehensive EDA analysis...")
        
        # Load the graph
        self.load_graph()
        
        # Run all analyses
        self.basic_statistics()
        self.analyze_node_attributes()
        self.analyze_semantic_types()
        self.analyze_edge_attributes()
        self.analyze_degree_distribution()
        self.analyze_connectivity()
        self.analyze_centrality()
        self.analyze_clustering()
        self.community_detection()
        self.analyze_paths()
        
        # Generate final report
        self.generate_summary_report()


def main():
    """Main function to run the EDA."""
    parser = argparse.ArgumentParser(
        description="Perform comprehensive EDA on a knowledge graph",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "graph_file",
        nargs='?',
        default="test_kg.graphml",
        help="Path to the GraphML file (default: test_kg.graphml)"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.graph_file):
        print(f"❌ Error: Graph file not found: {args.graph_file}")
        sys.exit(1)
    
    # Run EDA
    try:
        eda = KnowledgeGraphEDA(args.graph_file)
        eda.run_full_analysis()
    except Exception as e:
        logger.error(f"❌ EDA failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 