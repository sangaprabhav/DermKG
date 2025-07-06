# 🔬 DermKG: Dermatology Knowledge Graph Explorer

An interactive Streamlit application for exploring connections between dermatology diseases, drugs, and symptoms using a Neo4j knowledge graph.

## 🚀 Features

- **Interactive Graph Visualization**: Explore concepts and their relationships
- **Shortest Path Finding**: Discover connections between any two medical concepts
- **AI-Powered Insights**: Get intelligent descriptions of medical relationships
- **Chat Assistant**: Ask questions about the knowledge graph
- **Responsive Design**: Works on desktop and mobile devices

## 🌐 Live Demo

[Visit the deployed app on Streamlit Community Cloud](https://your-app-url.streamlit.app)

## 🛠️ Local Development

### Prerequisites

- Python 3.8+
- Neo4j database (local or cloud instance)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DermKG.git
cd DermKG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Neo4j credentials in `.streamlit/secrets.toml`:
```toml
[neo4j]
uri = "neo4j+s://your-neo4j-instance.databases.neo4j.io"
user = "neo4j"
password = "your-neo4j-password"
```

4. Run the application:
```bash
streamlit run app.py
```

## ☁️ Deployment on Streamlit Community Cloud

### Step 1: Prepare Your Repository

1. Fork this repository to your GitHub account
2. Ensure all files are committed and pushed to your repository

### Step 2: Create a Streamlit Community Cloud Account

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign up with your GitHub account
3. Connect your GitHub repository

### Step 3: Configure Secrets

1. In your Streamlit Community Cloud dashboard, go to your app settings
2. Navigate to the "Secrets" section
3. Add the following secrets:

```toml
[neo4j]
uri = "neo4j+s://your-neo4j-instance.databases.neo4j.io"
user = "neo4j"
password = "your-neo4j-password"
```

### Step 4: Deploy

1. Click "Deploy" in your Streamlit Community Cloud dashboard
2. Select your repository and the `app.py` file
3. Wait for the deployment to complete

### Step 5: Test

1. Visit your deployed app URL
2. Verify that the Neo4j connection works
3. Test the graph visualization and chat features

## 🔧 Configuration

### Neo4j Setup

This application requires a Neo4j database with dermatology concepts. The database should have:

- **Nodes**: Concepts with a `name` property
- **Relationships**: Various types connecting the concepts
- **Labels**: All nodes should have the `Concept` label

### Environment Variables

The app uses Streamlit secrets for configuration:

- `neo4j.uri`: Neo4j connection URI
- `neo4j.user`: Neo4j username
- `neo4j.password`: Neo4j password

### Model Configuration

The app uses lightweight language models optimized for cloud deployment:

- Primary: `microsoft/DialoGPT-small`
- Fallback: `gpt2`
- Lightest: `distilgpt2`

If models fail to load, the app falls back to rule-based responses.

## 📋 Dependencies

- `streamlit>=1.28.0` - Web application framework
- `neo4j>=5.12.0` - Neo4j database driver
- `streamlit-agraph>=0.0.45` - Graph visualization
- `transformers>=4.21.0` - AI model loading
- `torch>=2.0.0` - Machine learning backend
- `accelerate>=0.20.0` - Model optimization
- `sentencepiece>=0.1.99` - Tokenization
- `protobuf>=3.20.0` - Protocol buffers

## 🔒 Security

- Database credentials are stored in Streamlit secrets
- No sensitive information is hardcoded
- The app includes error handling for missing credentials

## 🐛 Troubleshooting

### Common Issues

1. **"Neo4j credentials not found"**
   - Ensure secrets are properly configured in Streamlit Community Cloud
   - Check that the secret keys match exactly: `neo4j.uri`, `neo4j.user`, `neo4j.password`

2. **"Failed to connect to Neo4j"**
   - Verify your Neo4j instance is running and accessible
   - Check that the URI, username, and password are correct
   - Ensure your Neo4j instance allows remote connections

3. **"LLM models not available"**
   - This is normal in resource-constrained environments
   - The app will still function with rule-based responses

4. **Slow loading**
   - First-time model loading may take a few minutes
   - Subsequent loads should be faster due to caching

### Performance Tips

- The app uses caching for database connections and model loading
- Graph visualizations are limited to 50 nodes for performance
- Models are optimized for low memory usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Neo4j for the graph database
- Streamlit for the web framework
- Hugging Face for the language models
- The dermatology community for domain knowledge

## 📧 Support

For questions or issues:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the deployment documentation

---

**Note**: This application is for educational and research purposes. Always consult medical professionals for health-related decisions.
