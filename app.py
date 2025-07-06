import os
import gc
import streamlit as st
from neo4j import GraphDatabase
from streamlit_agraph import agraph, Node, Edge, Config
import random
import warnings
import json
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Deployment configuration
IS_STREAMLIT_CLOUD = os.getenv('STREAMLIT_SHARING_MODE', 'false').lower() == 'true'
ENABLE_LLM = os.getenv('ENABLE_LLM', 'false').lower() == 'true' and not IS_STREAMLIT_CLOUD

# Import ML libraries only if LLM is enabled
if ENABLE_LLM:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        ENABLE_LLM = False
        st.warning("ML libraries not available. Using rule-based responses only.")


# --- Neo4j Connection Configuration ---
# Use Streamlit secrets for production deployment
try:
    NEO4J_URI = st.secrets["neo4j"]["uri"]
    NEO4J_USER = st.secrets["neo4j"]["user"]
    NEO4J_PASSWORD = st.secrets["neo4j"]["password"]
except KeyError:
    st.error("⚠️ Neo4j credentials not found in secrets. Please configure your secrets in Streamlit Community Cloud.")
    st.info("Required secrets: neo4j.uri, neo4j.user, neo4j.password")
    st.stop()
except Exception as e:
    st.error(f"Error loading Neo4j credentials: {e}")
    st.stop()

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="DermKG Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LLM Configuration ---
@st.cache_resource
def load_llm():
    """Load a lightweight LLM for generating graph descriptions."""
    if not ENABLE_LLM:
        return None, None
    
    try:
        # Clear memory before loading
        gc.collect()
        
        # Try only the smallest model for Streamlit Cloud
        model_name = "distilgpt2"
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            model_max_length=128  # Reduced further
        )
        
        # Load with absolute minimal memory footprint
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=None
        )
        
        # Add padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer, model
        
    except Exception as e:
        gc.collect()
        return None, None

def generate_graph_description(nodes, edges, selected_concept=None):
    """Generate a description of the current graph using the LLM or rules."""
    # Always provide a rule-based description first
    node_count = len(nodes)
    edge_count = len(edges)
    edge_types = list(set([edge.label for edge in edges[:10]]))
    
    base_description = f"This graph shows {node_count} dermatology concepts with {edge_count} relationships"
    if selected_concept:
        base_description = f"The graph centered on '{selected_concept}' shows {node_count} connected concepts"
    
    if edge_types:
        base_description += f", including relationships like: {', '.join(edge_types[:3])}"
    
    # Try LLM enhancement only if available
    if ENABLE_LLM:
        tokenizer, model = load_llm()
        if tokenizer and model:
            try:
                # Very simple prompt to avoid memory issues
                prompt = f"Medical graph: {base_description}. Insight:"
                
                inputs = tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    max_length=50, 
                    truncation=True, 
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=30,  # Very limited generation
                        temperature=0.7,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=True
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = response[len(prompt):].strip()
                
                if generated_text and len(generated_text) > 10:
                    return base_description + " " + generated_text
            except:
                pass
    
    return base_description

def generate_enhanced_path_description(source, destination, intermediate_concepts, relationship_types):
    """Generate a detailed description of a path between two medical concepts."""
    # Create a comprehensive medical analysis regardless of LLM availability
    analysis_parts = []
    
    # Basic path analysis
    if not intermediate_concepts:
        analysis_parts.append(f"This shows a direct relationship between {source} and {destination}.")
    else:
        analysis_parts.append(f"This path connects {source} to {destination} through {len(intermediate_concepts)} intermediate concept(s): {', '.join(intermediate_concepts)}.")
    
    # Relationship analysis
    if relationship_types:
        unique_relationships = list(set(relationship_types))
        analysis_parts.append(f"The connection involves {len(unique_relationships)} types of medical relationships: {', '.join(unique_relationships)}.")
    
    # Medical context based on common dermatology patterns
    medical_insights = generate_medical_insights(source, destination, intermediate_concepts, relationship_types)
    if medical_insights:
        analysis_parts.extend(medical_insights)
    
    # Try LLM enhancement if available
    tokenizer, model = load_llm()
    if tokenizer and model:
        try:
            # Simpler prompt for better results
            context = " ".join(analysis_parts)
            prompt = f"Medical context: {context}\n\nProvide additional clinical insights about this dermatology connection:"
            
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=300, 
                truncation=True, 
                padding=True
            )
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=inputs.input_ids.shape[1] + 60,
                    num_return_sequences=1,
                    temperature=0.8,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    top_p=0.9
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            if generated_text and len(generated_text) > 10:
                analysis_parts.append(f"Additional insight: {generated_text}")
        except Exception:
            pass  # Continue with rule-based analysis
    
    return " ".join(analysis_parts)

def generate_medical_insights(source, destination, intermediate_concepts, relationship_types):
    """Generate medical insights based on dermatology knowledge patterns."""
    insights = []
    
    # Drug and treatment analysis
    drug_keywords = ['drug', 'treatment', 'therapy', 'medication', 'antibiotic', 'steroid', 'cream', 'ointment']
    condition_keywords = ['acne', 'eczema', 'psoriasis', 'dermatitis', 'infection', 'rash', 'lesion']
    
    source_lower = source.lower()
    dest_lower = destination.lower()
    
    # Treatment pathway analysis
    if any(word in dest_lower for word in drug_keywords) and any(word in source_lower for word in condition_keywords):
        insights.append(f"This represents a treatment pathway where {destination} is used to manage {source}.")
    
    # Causal relationship analysis
    if 'caused_by' in relationship_types or 'causes' in relationship_types:
        insights.append("This connection suggests a causal relationship in the disease process.")
    
    # Treatment relationship analysis  
    if 'treats' in relationship_types or 'treatment' in relationship_types:
        insights.append("This indicates a therapeutic relationship between these concepts.")
    
    # Symptom analysis
    if intermediate_concepts:
        symptom_keywords = ['symptom', 'sign', 'manifestation', 'finding']
        for concept in intermediate_concepts:
            if any(word in concept.lower() for word in symptom_keywords):
                insights.append(f"The intermediate concept '{concept}' represents a clinical manifestation linking these conditions.")
    
    return insights

def generate_chat_response(user_input, graph_description):
    """Generate a chat response based on user input and current graph context."""
    # First, try to provide rule-based responses for common questions
    rule_based_response = get_rule_based_response(user_input, graph_description)
    if rule_based_response:
        return rule_based_response
    
    # Then try LLM enhancement
    tokenizer, model = load_llm()
    if tokenizer and model:
        try:
            # Create context-aware prompt with enhanced context
            context_parts = []
            if graph_description:
                context_parts.append(f"Graph context: {graph_description}")
            
            # Add path-specific context if available
            if st.session_state.current_path_context:
                path_info = st.session_state.current_path_context
                context_parts.append(f"Current path: {path_info['source']} → {path_info['destination']}")
                if path_info['intermediate_concepts']:
                    context_parts.append(f"Via: {', '.join(path_info['intermediate_concepts'])}")
                context_parts.append(f"Relationships: {', '.join(path_info['relationship_types'])}")
            
            context = " | ".join(context_parts)
            prompt = f"{context}\nQ: {user_input}\nA:"
            
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=300, 
                truncation=True, 
                padding=True
            )
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=inputs.input_ids.shape[1] + 50,
                    num_return_sequences=1,
                    temperature=0.8,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    top_p=0.9
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            if generated_text and len(generated_text) > 10:
                return generated_text
        except Exception:
            pass
    
    # Fallback to contextual response
    return get_contextual_fallback_response(user_input)

def get_rule_based_response(user_input, graph_description):
    """Generate rule-based responses for common dermatology questions."""
    user_lower = user_input.lower()
    
    # Path analysis questions
    if st.session_state.current_path_context:
        path_info = st.session_state.current_path_context
        
        if any(phrase in user_lower for phrase in ['explain', 'connection', 'relate', 'link']):
            response_parts = []
            response_parts.append(f"This path shows how {path_info['source']} connects to {path_info['destination']}.")
            
            if path_info['intermediate_concepts']:
                response_parts.append(f"The connection goes through {len(path_info['intermediate_concepts'])} intermediate step(s): {', '.join(path_info['intermediate_concepts'])}.")
            else:
                response_parts.append("This is a direct connection between the two concepts.")
                
            if 'treats' in path_info['relationship_types']:
                response_parts.append("This suggests a treatment relationship.")
            if 'caused_by' in path_info['relationship_types']:
                response_parts.append("This indicates a causal relationship.")
                
            return " ".join(response_parts)
        
        if any(phrase in user_lower for phrase in ['intermediate', 'between', 'steps']):
            if path_info['intermediate_concepts']:
                return f"The intermediate concepts connecting {path_info['source']} to {path_info['destination']} are: {', '.join(path_info['intermediate_concepts'])}. These represent the medical pathway linking these conditions."
            else:
                return f"There are no intermediate concepts - {path_info['source']} is directly connected to {path_info['destination']}."
        
        if any(phrase in user_lower for phrase in ['treatment', 'therapy', 'clinical']):
            treatment_insights = []
            if any(word in path_info['destination'].lower() for word in ['drug', 'medication', 'treatment', 'therapy']):
                treatment_insights.append(f"{path_info['destination']} appears to be a treatment option for {path_info['source']}.")
            if 'treats' in path_info['relationship_types']:
                treatment_insights.append("This path represents a therapeutic relationship.")
            
            if treatment_insights:
                return " ".join(treatment_insights) + " Consider consulting medical literature for specific treatment protocols."
    
    # General concept questions
    if any(phrase in user_lower for phrase in ['what is', 'define', 'definition']):
        return "This is a concept in the dermatology knowledge graph. For detailed medical definitions, please consult medical references or speak with a healthcare professional."
    
    if any(phrase in user_lower for phrase in ['treatment', 'treat', 'therapy']):
        return "For treatment information, this knowledge graph shows relationships between conditions and therapies. Always consult healthcare professionals for specific treatment advice."
    
    if any(phrase in user_lower for phrase in ['symptom', 'sign', 'manifestation']):
        return "The graph shows connections between conditions and their clinical manifestations. For symptom evaluation, consult medical professionals."
    
    return None

def get_contextual_fallback_response(user_input):
    """Provide helpful fallback responses based on context."""
    if st.session_state.current_path_context:
        path_info = st.session_state.current_path_context
        return f"I'm analyzing the connection between {path_info['source']} and {path_info['destination']}. You can ask about their medical relationship, intermediate concepts, or clinical implications."
    else:
        return "I can help explain medical concepts and their relationships in the dermatology knowledge graph. Try selecting a concept or finding a path between two concepts, then ask specific questions about their connections."

# Optimize Neo4j queries for deployment
def safe_neo4j_query(driver, query, params=None, limit=30):
    """Execute Neo4j query with safety limits for deployment."""
    # Add default limit if not present
    if "LIMIT" not in query.upper():
        query += f" LIMIT {limit}"
    
    try:
        with driver.session(database="neo4j") as session:
            result = session.run(query, parameters=params or {})
            return list(result)
    except Exception as e:
        st.error(f"Query error: {str(e)[:100]}")
        return []

# Add memory monitoring for deployment
def check_memory_usage():
    """Check memory usage and warn if high."""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > 800:  # 800MB threshold for 1GB limit
            st.warning("⚠️ High memory usage detected. Some features may be limited.")
            gc.collect()
            
        return memory_mb
    except:
        return 0

# Add a cleanup function for chat history
def manage_chat_history():
    """Keep chat history within limits to prevent memory issues."""
    if len(st.session_state.chat_history) > st.session_state.chat_history_limit:
        st.session_state.chat_history = st.session_state.chat_history[-st.session_state.chat_history_limit:]

# --- Helper Functions ---
def get_pretty_name(long_name: str) -> str:
    """
    Cleans up long, complex node names for better display.
    e.g., '(Acne vulgaris) or (blackhead) or (comedo)' -> 'Acne vulgaris'
    """
    if not long_name or not isinstance(long_name, str):
        return "Unknown"
    name = long_name.strip()
    # If the name is in the format `(A) or (B)`, take the content of the first parenthesis.
    if name.startswith('(') and ')' in name:
        return name[1:name.find(')')].strip()
    return name

def get_node_color(node_name: str) -> str:
    """Assigns a color based on a simple keyword search in the node name."""
    name_lower = str(node_name).lower() if node_name else ""
    if any(keyword in name_lower for keyword in ["drug", "pharmaceutical", "methotrexate", "steroid", "permethrin"]):
        return "#FF6B6B"  # Red
    if any(keyword in name_lower for keyword in ["finding", "symptom", "pruritus", "comedone", "blister", "nevus"]):
        return "#FFD166"  # Yellow
    return "#4D9DE0"      # Blue (default for diseases)

# --- Input Validation ---
def validate_node_name(name: str) -> bool:
    """Validate node name to prevent issues."""
    if not name or not isinstance(name, str):
        return False
    if len(name) > 500:
        return False
    return True

# --- Caching Neo4j Driver and Node Data ---
@st.cache_resource
def get_driver():
    """Establishes a connection to the Neo4j database."""
    try:
        # Determine if we need to explicitly set encryption based on URI scheme
        uri_scheme = NEO4J_URI.split('://')[0].lower()
        encrypted_schemes = ['neo4j+s', 'neo4j+ssc', 'bolt+s', 'bolt+ssc']
        
        # Only set encrypted=True for schemes that don't already include encryption
        driver_config = {
            'auth': (NEO4J_USER, NEO4J_PASSWORD),
            'max_connection_lifetime': 30 * 60,
            'max_connection_pool_size': 10 if IS_STREAMLIT_CLOUD else 25,
            'connection_acquisition_timeout': 60,
        }
        
        # Add encryption setting only if URI scheme doesn't already specify it
        if uri_scheme not in encrypted_schemes:
            driver_config['encrypted'] = True
        
        driver = GraphDatabase.driver(NEO4J_URI, **driver_config)
        
        # Test the connection
        with driver.session() as session:
            session.run("RETURN 1")
            
        st.success("✅ Connected to Neo4j database")
        return driver
        
    except Exception as e:
        st.error(f"❌ Failed to connect to Neo4j: {str(e)}")
        st.info("""
        Please check:
        1. Your Neo4j instance is running
        2. Credentials are correctly set in Streamlit secrets
        3. The database allows remote connections
        """)
        return None

@st.cache_data
def get_node_data(_driver):
    """
    Fetches all node names and creates a mapping from a "pretty name" to the full name.
    """
    if not _driver:
        return [], {}
    
    with _driver.session(database="neo4j") as session:
        result = session.run("MATCH (c:Concept) RETURN c.name AS name ORDER BY name")
        full_names = [record["name"] for record in result if record["name"]]
    
    # Create the mapping. Handle potential duplicate pretty names by letting the last one win.
    pretty_to_full_map = {get_pretty_name(name): name for name in full_names}
    # Create a sorted list of unique pretty names for the dropdowns
    pretty_names = sorted(list(pretty_to_full_map.keys()))
    
    return pretty_names, pretty_to_full_map

# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_history_limit" not in st.session_state:
    st.session_state.chat_history_limit = 20 if IS_STREAMLIT_CLOUD else 50

if "current_graph_description" not in st.session_state:
    st.session_state.current_graph_description = None

# --- Initialize path context ---
if "current_path_context" not in st.session_state:
    st.session_state.current_path_context = None

# --- Initialize chat input state ---
if "chat_input_value" not in st.session_state:
    st.session_state.chat_input_value = ""

# --- Main App ---
st.title("🔬 DermKG: The Dermatology Knowledge Graph Explorer")
st.markdown("An interactive tool to explore connections between diseases, drugs, and symptoms.")
st.markdown("---")

driver = get_driver()
if driver:
    pretty_node_names, pretty_to_full_map = get_node_data(driver)
else:
    pretty_node_names, pretty_to_full_map = [], {}
    st.stop()

# --- Sidebar Controls ---
st.sidebar.title("Controls")
st.sidebar.markdown("Use the options below to explore the graph.")

if st.sidebar.button("Show Random Concept", use_container_width=True):
    if pretty_node_names:
        st.session_state.selected_node_key = random.choice(pretty_node_names)

# Add deployment information in sidebar
if IS_STREAMLIT_CLOUD:
    st.sidebar.info("🌐 Running on Streamlit Cloud (Limited Resources)")
    if st.sidebar.button("Clear Memory"):
        gc.collect()
        st.cache_resource.clear()
        st.success("Memory cleared!")

# Show memory usage if psutil is available
try:
    memory_mb = check_memory_usage()
    if memory_mb > 0:
        st.sidebar.metric("Memory Usage", f"{memory_mb:.0f} MB / 1000 MB")
except:
    pass

# --- Tabs for Different Functionalities ---
tab1, tab2 = st.tabs(["🔍 Explore a Concept", "↔️ Find a Path"])

# --- TAB 1: Explore a Concept ---
with tab1:
    st.header("Explore a Concept and its Neighbors")
    
    # Create two columns for graph and chat
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Graph Visualization")
        
        default_index = None
        if 'selected_node_key' in st.session_state and st.session_state.get('selected_node_key') in pretty_node_names:
            default_index = pretty_node_names.index(st.session_state.get('selected_node_key'))

        selected_pretty_name = st.selectbox(
            "Search for a concept or show a random one",
            options=pretty_node_names,
            index=default_index,
            placeholder="Type to search...",
            key='selected_node_key'
        )

        if selected_pretty_name and validate_node_name(selected_pretty_name):
            # Clear path context when exploring individual concepts
            st.session_state.current_path_context = None
            
            # Translate the selected "pretty name" back to the full name for the query
            selected_full_name = pretty_to_full_map.get(selected_pretty_name)
            if not selected_full_name:
                st.error("Selected concept not found in database.")
                st.stop()
            st.success(f"Displaying neighbors for: **{selected_pretty_name}**")
            
            query = "MATCH (p:Concept {name: $node_name})-[r]-(neighbor:Concept) RETURN p, r, neighbor LIMIT 30"
            nodes = []
            edges = []
            node_ids = set()

            with st.spinner("Loading graph data..."):
                with driver.session(database="neo4j") as session:
                    result = session.run(query, node_name=selected_full_name)
                    for record in result:
                        source_node, rel, target_node = record["p"], record["r"], record["neighbor"]

                    # Use full names for IDs and pretty names for labels
                    if source_node.element_id not in node_ids:
                        nodes.append(Node(id=source_node['name'], label=get_pretty_name(source_node['name']), size=40, color="#06D6A0", shape="star"))
                        node_ids.add(source_node.element_id)
                    if target_node.element_id not in node_ids:
                        nodes.append(Node(id=target_node['name'], label=get_pretty_name(target_node['name']), size=20, color=get_node_color(target_node['name'])))
                        node_ids.add(target_node.element_id)
                    
                    edges.append(Edge(source=source_node['name'], target=target_node['name'], label=rel['type']))
            
            config = Config(width=800, height=600, directed=True, physics={"solver": "forceAtlas2Based", "forceAtlas2Based": {"gravitationalConstant": -45, "centralGravity": 0.005, "springLength": 120}}, nodeHighlightBehavior=True, highlightColor="#F7A7A6", node={'labelProperty':'label'})

            if nodes:
                agraph(nodes=nodes, edges=edges, config=config)
                
                # Store concept exploration context
                neighbor_concepts = [node.label for node in nodes if node.label != selected_pretty_name]
                relationship_types = list(set([edge.label for edge in edges]))
                
                # Generate description for the current graph
                description = generate_graph_description(nodes, edges, selected_pretty_name)
                st.session_state.current_graph_description = description
                
                # Display AI-generated description
                st.info(f"**AI Analysis:** {description}")
                
                # Display concept summary
                st.success(f"**Selected Concept:** {selected_pretty_name}")
                st.write(f"**Connected concepts:** {len(neighbor_concepts)} neighbors")
                st.write(f"**Relationship types:** {', '.join(relationship_types)}")
                
                # Show sample neighbors
                if neighbor_concepts:
                    sample_neighbors = neighbor_concepts[:5]
                    st.write(f"**Sample neighbors:** {', '.join(sample_neighbors)}")
                    if len(neighbor_concepts) > 5:
                        st.write(f"... and {len(neighbor_concepts) - 5} more")
            else:
                st.warning("No relationships found for this concept in the graph.")
    
    with col2:
        st.subheader("💬 Chat Assistant")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (user_msg, ai_msg) in enumerate(st.session_state.chat_history):
                st.write(f"**You:** {user_msg}")
                st.write(f"**AI:** {ai_msg}")
                st.write("---")
        
        # Chat input with context-aware placeholder
        if st.session_state.current_path_context:
            path_info = st.session_state.current_path_context
            placeholder = f"Ask about the path from {path_info['source']} to {path_info['destination']}..."
        else:
            placeholder = "Ask about the graph or dermatology concepts..."
        
        # Chat input with clear functionality
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask about the graph or dermatology concepts:", 
                value=st.session_state.chat_input_value,
                placeholder=placeholder, 
                key="chat_input"
            )
        
        with col2:
            if st.button("Clear", key="clear_chat_tab1"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Show helpful prompts based on context
        if st.session_state.current_path_context:
            st.caption("💡 Try asking: 'Explain the medical connection', 'What are the intermediate concepts?', 'Clinical implications?'")
        else:
            st.caption("💡 Try asking: 'What is this concept?', 'Related treatments?', 'Connected symptoms?'")
        
        if st.button("Send", key="send_chat"):
            if user_input:
                ai_response = generate_chat_response(user_input, st.session_state.current_graph_description)
                st.session_state.chat_history.append((user_input, ai_response))
                
                # Limit chat history
                manage_chat_history()
                
                # Clear input
                st.session_state.chat_input_value = ""
                st.rerun()

# --- TAB 2: Find a Path ---
with tab2:
    st.header("Find the Shortest Path Between Two Concepts")
    
    # Create two columns for graph and chat in path tab too
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Path Visualization")
        
        path_col1, path_col2 = st.columns(2)
        with path_col1:
            source_pretty_name = st.selectbox("Select a Source Concept", options=pretty_node_names, index=None, key="source_path")
        with path_col2:
            target_pretty_name = st.selectbox("Select a Target Concept", options=pretty_node_names, index=None, key="target_path")

        if source_pretty_name and target_pretty_name and validate_node_name(source_pretty_name) and validate_node_name(target_pretty_name):
            if source_pretty_name == target_pretty_name:
                st.warning("Source and Target concepts cannot be the same.")
            else:
                # Translate pretty names to full names for the query
                source_full_name = pretty_to_full_map.get(source_pretty_name)
                target_full_name = pretty_to_full_map.get(target_pretty_name)
                
                if not source_full_name or not target_full_name:
                    st.error("One or both selected concepts not found in database.")
                    st.stop()
                
                path_query = "MATCH path = shortestPath((source:Concept {name: $source_name})-[*..5]->(target:Concept {name: $target_name})) RETURN path"
                path_nodes = []
                path_edges = []
                path_node_ids = set()

                with st.spinner("Finding shortest path..."):
                    with driver.session(database="neo4j") as session:
                        result = session.run(path_query, source_name=source_full_name, target_name=target_full_name).single()
                if result:
                    path = result["path"]
                    
                    # Extract detailed path information
                    path_node_names = []
                    relationship_types = []
                    
                    for node in path.nodes:
                        if node.element_id not in path_node_ids:
                            node_pretty_name = get_pretty_name(node['name'])
                            path_nodes.append(Node(id=node['name'], label=node_pretty_name, size=25, color=get_node_color(node['name'])))
                            path_node_ids.add(node.element_id)
                            path_node_names.append(node_pretty_name)
                    
                    for rel in path.relationships:
                        path_edges.append(Edge(source=rel.start_node['name'], target=rel.end_node['name'], label=rel['type']))
                        relationship_types.append(rel['type'])
                    
                    # Store detailed path context for AI assistant
                    intermediate_concepts = path_node_names[1:-1] if len(path_node_names) > 2 else []
                    st.session_state.current_path_context = {
                        'source': source_pretty_name,
                        'destination': target_pretty_name,
                        'path_length': len(path.nodes),
                        'intermediate_concepts': intermediate_concepts,
                        'relationship_types': list(set(relationship_types)),
                        'all_concepts': path_node_names
                    }
                    
                    config_path = Config(width=800, height=500, directed=True, physics=True)
                    agraph(nodes=path_nodes, edges=path_edges, config=config_path)
                    
                    # Generate enhanced description for the path
                    path_description = generate_enhanced_path_description(
                        source_pretty_name, 
                        target_pretty_name, 
                        intermediate_concepts, 
                        relationship_types
                    )
                    st.session_state.current_graph_description = path_description
                    
                    st.info(f"**AI Analysis:** {path_description}")
                    
                    # Display path summary
                    st.success(f"**Path Found:** {source_pretty_name} → {target_pretty_name}")
                    if intermediate_concepts:
                        st.write(f"**Intermediate concepts:** {' → '.join(intermediate_concepts)}")
                    st.write(f"**Relationship types:** {', '.join(set(relationship_types))}")
                    st.write(f"**Path length:** {len(path.nodes)} concepts")
                else:
                    st.error(f"No path found between '{source_pretty_name}' and '{target_pretty_name}'.")
                    st.session_state.current_path_context = None
    
    with col2:
        st.subheader("💬 Chat Assistant")
        
        # Display chat history (same as tab 1)
        chat_container = st.container()
        with chat_container:
            for i, (user_msg, ai_msg) in enumerate(st.session_state.chat_history):
                st.write(f"**You:** {user_msg}")
                st.write(f"**AI:** {ai_msg}")
                st.write("---")
        
        # Chat input with context-aware placeholder
        if st.session_state.current_path_context:
            path_info = st.session_state.current_path_context
            placeholder = f"Ask about the path from {path_info['source']} to {path_info['destination']}..."
        else:
            placeholder = "Ask about the path or relationships..."
        
        # Chat input with clear functionality
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask about the path or relationships:", 
                value=st.session_state.chat_input_value,
                placeholder=placeholder, 
                key="chat_input_path"
            )
        
        with col2:
            if st.button("Clear", key="clear_chat_tab2"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Show helpful prompts based on context
        if st.session_state.current_path_context:
            st.caption("💡 Try asking: 'Explain this medical connection', 'Why these intermediate steps?', 'Treatment implications?'")
        else:
            st.caption("💡 Select concepts above to analyze their connection path")
        
        if st.button("Send", key="send_chat_path"):
            if user_input:
                ai_response = generate_chat_response(user_input, st.session_state.current_graph_description)
                st.session_state.chat_history.append((user_input, ai_response))
                
                # Limit chat history
                manage_chat_history()
                
                # Clear input
                st.session_state.chat_input_value = ""
                st.rerun()