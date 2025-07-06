# 🚀 Deployment Checklist for Streamlit Community Cloud

## Pre-Deployment Checklist

### ✅ Repository Setup
- [ ] Fork or clone the repository to your GitHub account
- [ ] Ensure all files are committed and pushed to your repository
- [ ] Verify that `.gitignore` is properly configured
- [ ] Check that `requirements.txt` contains all necessary dependencies

### ✅ Neo4j Database Setup
- [ ] Neo4j database instance is running and accessible
- [ ] Database contains dermatology concepts with proper labels
- [ ] Connection credentials are ready (URI, username, password)
- [ ] Database allows remote connections
- [ ] Test connection from your local environment

### ✅ Streamlit Community Cloud Account
- [ ] Create account at [share.streamlit.io](https://share.streamlit.io)
- [ ] Link your GitHub account
- [ ] Verify repository access permissions

### ✅ Secrets Configuration
- [ ] Prepare your secrets in the correct format:
  ```toml
  [neo4j]
  uri = "neo4j+s://your-instance.databases.neo4j.io"
  user = "neo4j"
  password = "your-password"
  ```
- [ ] Test secrets format locally (optional)

## Deployment Steps

### 1. Create New App
- [ ] Go to Streamlit Community Cloud dashboard
- [ ] Click "New app"
- [ ] Select your repository
- [ ] Choose `app.py` as the main file
- [ ] Select the correct branch (usually `main`)

### 2. Configure Secrets
- [ ] In app settings, navigate to "Secrets" tab
- [ ] Paste your secrets configuration
- [ ] Save the secrets
- [ ] Verify the format is correct

### 3. Deploy
- [ ] Click "Deploy" button
- [ ] Wait for deployment to complete
- [ ] Monitor deployment logs for any errors

### 4. Test Deployment
- [ ] Access your app URL
- [ ] Verify Neo4j connection works
- [ ] Test graph visualization
- [ ] Try concept exploration
- [ ] Test path finding feature
- [ ] Verify chat assistant functionality

## Post-Deployment Verification

### ✅ Functionality Tests
- [ ] **Database Connection**: Green checkmark appears for Neo4j connection
- [ ] **Graph Visualization**: Concepts display correctly in interactive graph
- [ ] **Search**: Concept search dropdown works
- [ ] **Random Concept**: Random concept button functions
- [ ] **Path Finding**: Shortest path feature works between concepts
- [ ] **Chat Assistant**: AI responses are generated (or rule-based fallback works)
- [ ] **Responsive Design**: App works on mobile and desktop

### ✅ Performance Tests
- [ ] **Loading Speed**: App loads within 30 seconds
- [ ] **Graph Rendering**: Interactive graphs render smoothly
- [ ] **Memory Usage**: No memory overflow errors
- [ ] **Model Loading**: LLM models load successfully (or fallback works)

### ✅ Error Handling
- [ ] **Missing Secrets**: Appropriate error messages for missing credentials
- [ ] **Database Errors**: Clear error messages for connection issues
- [ ] **Model Errors**: Graceful fallback to rule-based responses
- [ ] **Network Errors**: Proper handling of network timeouts

## Common Issues and Solutions

### Issue: "Neo4j credentials not found"
- **Solution**: Verify secrets are configured correctly in app settings
- **Check**: Ensure keys are exactly `neo4j.uri`, `neo4j.user`, `neo4j.password`

### Issue: "Failed to connect to Neo4j"
- **Solution**: Check database is running and accessible
- **Check**: Verify URI format includes protocol (e.g., `neo4j+s://`)
- **Check**: Ensure credentials are correct

### Issue: "LLM models not available"
- **Solution**: This is normal on resource-constrained environments
- **Expected**: App should still work with rule-based responses

### Issue: App is slow or times out
- **Solution**: Check model loading configuration
- **Check**: Verify database queries are optimized
- **Consider**: Reducing model complexity or disabling LLM features

### Issue: Memory errors
- **Solution**: Optimize model loading settings
- **Check**: Reduce batch sizes or model complexity
- **Consider**: Using lighter models or disabling AI features

## Resource Limits

### Streamlit Community Cloud Limits
- **Memory**: 1GB RAM limit
- **CPU**: Shared CPU resources
- **Storage**: Limited disk space
- **Bandwidth**: Fair usage policy

### Optimization Tips
- **Models**: Use lightweight models (DialoGPT-small, distilgpt2)
- **Caching**: Leverage Streamlit caching for database connections
- **Queries**: Limit Neo4j query results (currently set to 50 nodes)
- **Memory**: Enable low memory usage options for models

## Support Resources

### Documentation
- [Streamlit Community Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Neo4j Documentation](https://neo4j.com/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Community Support
- [Streamlit Community Forum](https://discuss.streamlit.io/)
- [Neo4j Community](https://community.neo4j.com/)
- [GitHub Issues](https://github.com/yourusername/DermKG/issues)

## Final Verification

After successful deployment, your app should:
- ✅ Load within 30 seconds
- ✅ Display dermatology concepts in an interactive graph
- ✅ Allow exploration of concept relationships
- ✅ Enable path finding between concepts
- ✅ Provide AI-powered or rule-based chat assistance
- ✅ Work responsively on different screen sizes

## Troubleshooting Commands

If you need to debug locally:
```bash
# Test local deployment
streamlit run app.py

# Check requirements
pip install -r requirements.txt

# Verify Neo4j connection
python -c "from neo4j import GraphDatabase; print('Neo4j driver imported successfully')"

# Check model loading
python -c "from transformers import AutoTokenizer; print('Transformers imported successfully')"
```

## Security Notes

- 🔒 Never commit secrets to your repository
- 🔒 Use Streamlit secrets for all sensitive configuration
- 🔒 Regularly rotate your Neo4j credentials
- 🔒 Monitor app usage for any unusual activity

---

**Need Help?** Check the troubleshooting section in the main README or open an issue on GitHub. 