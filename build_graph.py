#!/usr/bin/env python3
"""
Dermatology Knowledge Graph Builder - Main Script

This script builds a knowledge graph from UMLS data focused on dermatology concepts.
It uses a modular architecture with separate modules for configuration, validation,
data loading, and graph construction.

Usage:
    python build_graph.py /path/to/umls/directory [options]

Examples:
    # Use the built-in data directory
    python build_graph.py build/data/ --output build/data/output/my_kg.graphml
    
    # Use custom UMLS directory
    python build_graph.py /data/umls2023 --output my_derm_kg.graphml --sources ICD10CM MSH
    
    # Validate files first
    python build_graph.py build/data/ --validate-only
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import List

# Import our custom modules from the build package
from build.config import (
    DEFAULT_SOURCES, DEFAULT_OUTPUT_FILE, LOG_FORMAT, LOG_LEVELS
)
from build.validators import validate_umls_files, validate_output_path, validate_sources
from build.data_loaders import (
    load_seed_cuis_and_names, load_relevant_cuis_by_semantic_type,
    load_relationships, get_statistics, load_additional_names
)
from build.graph_builder import (
    build_graph, add_node_metadata, add_edge_metadata,
    calculate_graph_statistics, save_graph, add_graph_metadata,
    validate_graph
)

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    numeric_level = getattr(logging, log_level.upper())
    logging.getLogger().setLevel(numeric_level)
    
    # Set levels for our custom modules
    for module_name in ['validators', 'data_loaders', 'graph_builder']:
        logging.getLogger(module_name).setLevel(numeric_level)


def create_pipeline_metadata(sources: List[str], umls_dir: str) -> dict:
    """
    Create metadata about the pipeline execution.
    
    Args:
        sources: List of source vocabularies used
        umls_dir: Path to UMLS directory
        
    Returns:
        Dictionary with pipeline metadata
    """
    return {
        'creation_date': datetime.now().isoformat(),
        'umls_directory': umls_dir,
        'sources_used': sources,
        'script_version': '2.0.0',
        'python_version': sys.version.split()[0],
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Build a Dermatology Knowledge Graph from UMLS RRF files.",
        formatter_class=argparse.RawTextHelpFormatter,
                 epilog="""
Examples:
  %(prog)s build/data/
  %(prog)s build/data/ --output build/data/output/my_kg.graphml --sources ICD10CM MSH
  %(prog)s /custom/umls/path/ --log-level DEBUG --validate-only
        """
    )
    
    parser.add_argument(
        "umls_dir", 
        type=str,
        help="Path to the directory containing UMLS RRF files (MRCONSO.RRF, MRREL.RRF, etc.)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=DEFAULT_OUTPUT_FILE,
        help=f"Path to save the final graph file (default: {DEFAULT_OUTPUT_FILE})"
    )
    
    parser.add_argument(
        "--sources", 
        nargs='+', 
        default=DEFAULT_SOURCES,
        help=f"Source vocabularies to search for dermatology concepts (default: {DEFAULT_SOURCES})"
    )
    
    parser.add_argument(
        "--log-level", 
        choices=LOG_LEVELS, 
        default='INFO',
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate files without building the graph"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        # Create pipeline metadata
        pipeline_metadata = create_pipeline_metadata(args.sources, args.umls_dir)
        
        logger.info("="*60)
        logger.info("DERMATOLOGY KNOWLEDGE GRAPH BUILDER")
        logger.info("="*60)
        logger.info(f"UMLS Directory: {args.umls_dir}")
        logger.info(f"Output File: {args.output}")
        logger.info(f"Sources: {args.sources}")
        logger.info(f"Log Level: {args.log_level}")
        logger.info("="*60)
        
        # Step 1: Validate inputs
        logger.info("STEP 1: Validating inputs")
        logger.info("-" * 30)
        
        # Validate UMLS files
        mrconso_path, mrrel_path, mrsty_path = validate_umls_files(args.umls_dir)
        
        # Validate output path
        if not validate_output_path(args.output):
            raise ValueError(f"Invalid output path: {args.output}")
        
        # Validate sources
        if not validate_sources(args.sources):
            raise ValueError(f"Invalid sources: {args.sources}")
        
        logger.info("✓ All inputs validated successfully")
        
        # Exit if validation-only mode
        if args.validate_only:
            logger.info("Validation-only mode: Exiting after successful validation")
            return
        
        # Step 2: Load seed CUIs and names
        logger.info("\nSTEP 2: Loading seed dermatology concepts")
        logger.info("-" * 30)
        
        seed_cuis, cui_to_name = load_seed_cuis_and_names(mrconso_path, args.sources)
        
        # Step 3: Load CUIs with allowed semantic types
        logger.info("\nSTEP 3: Loading concepts with allowed semantic types")
        logger.info("-" * 30)
        
        allowed_cuis = load_relevant_cuis_by_semantic_type(mrsty_path)
        
        # Step 4: Load and filter relationships
        logger.info("\nSTEP 4: Loading and filtering relationships")
        logger.info("-" * 30)
        
        # Combine seed and allowed CUIs
        all_relevant_cuis = seed_cuis.union(allowed_cuis)
        logger.info(f"Total relevant CUIs: {len(all_relevant_cuis)}")
        
        relationships = load_relationships(mrrel_path, seed_cuis, all_relevant_cuis)
        
        # Step 5: Load additional names if needed
        logger.info("\nSTEP 5: Loading additional concept names")
        logger.info("-" * 30)
        
        # Get CUIs that appear in relationships
        cuis_in_relationships = set()
        for cui1, _, cui2 in relationships:
            cuis_in_relationships.add(cui1)
            cuis_in_relationships.add(cui2)
        
        # Load additional names for CUIs that don't have them
        cui_to_name = load_additional_names(mrconso_path, cuis_in_relationships, cui_to_name)
        
        # Step 6: Build the graph
        logger.info("\nSTEP 6: Building the knowledge graph")
        logger.info("-" * 30)
        
        # Build basic graph
        graph = build_graph(relationships, cui_to_name)
        
        # Add node metadata
        graph = add_node_metadata(graph, seed_cuis, allowed_cuis)
        
        # Add edge metadata
        graph = add_edge_metadata(graph)
        
        # Calculate statistics
        data_stats = get_statistics(seed_cuis, allowed_cuis, relationships)
        graph_stats = calculate_graph_statistics(graph)
        
        # Add comprehensive metadata
        graph = add_graph_metadata(
            graph, 
            args.sources, 
            {**data_stats, **graph_stats}, 
            pipeline_metadata
        )
        
        # Step 7: Validate the graph
        logger.info("\nSTEP 7: Validating the graph")
        logger.info("-" * 30)
        
        validation_issues = validate_graph(graph)
        if validation_issues:
            logger.warning(f"Graph validation found {len(validation_issues)} issues")
            for issue in validation_issues[:5]:  # Show first 5 issues
                logger.warning(f"  - {issue}")
            if len(validation_issues) > 5:
                logger.warning(f"  ... and {len(validation_issues) - 5} more issues")
        else:
            logger.info("✓ Graph validation passed")
        
        # Step 8: Save the graph
        logger.info("\nSTEP 8: Saving the graph")
        logger.info("-" * 30)
        
        success = save_graph(graph, args.output)
        if not success:
            raise RuntimeError("Failed to save graph")
        
        # Step 9: Final summary
        logger.info("\nSTEP 9: Final summary")
        logger.info("-" * 30)
        
        logger.info("✓ Knowledge graph construction completed successfully!")
        logger.info(f"✓ Graph saved to: {args.output}")
        logger.info("\nFinal Statistics:")
        logger.info(f"  • Seed CUIs: {data_stats['seed_cuis']:,}")
        logger.info(f"  • Allowed CUIs: {data_stats['allowed_cuis']:,}")
        logger.info(f"  • Total relevant CUIs: {data_stats['total_relevant_cuis']:,}")
        logger.info(f"  • Final graph nodes: {graph_stats['nodes']:,}")
        logger.info(f"  • Final graph edges: {graph_stats['edges']:,}")
        logger.info(f"  • Unique relationship types: {graph_stats['unique_edge_types']:,}")
        logger.info(f"  • Graph density: {graph_stats['density']:.6f}")
        logger.info(f"  • Connected components: {graph_stats['weakly_connected_components']:,}")
        if 'largest_component_size' in graph_stats:
            logger.info(f"  • Largest component size: {graph_stats['largest_component_size']:,}")
        
        logger.info("\n" + "="*60)
        logger.info("SUCCESS: Knowledge graph construction completed!")
        logger.info("="*60)
        
        print(f"\n🎉 Success! Your dermatology knowledge graph has been built.")
        print(f"📁 Output file: {args.output}")
        print(f"📊 Nodes: {graph_stats['nodes']:,} | Edges: {graph_stats['edges']:,}")
        print(f"🔗 You can now load and analyze this graph using:")
        print(f"   • NetworkX in Python")
        print(f"   • Gephi (graph visualization)")
        print(f"   • Cytoscape (network analysis)")
        print(f"   • Neo4j (graph database)")
        
    except KeyboardInterrupt:
        logger.info("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()