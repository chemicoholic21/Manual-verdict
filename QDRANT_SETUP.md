# Qdrant Setup

Qdrant vector database client has been configured for your project.

## Configuration

**Cluster Details:**
- **Cluster ID**: `df856a23-961c-45c7-a091-6b755c9343ef`
- **URL**: `https://df856a23-961c-45c7-a091-6b755c9343ef.us-east4-0.gcp.cloud.qdrant.io:6333`
- **Endpoint**: `https://df856a23-961c-45c7-a091-6b755c9343ef.us-east4-0.gcp.cloud.qdrant.io`
- **API Key**: Stored in `.env` file

## Installation

Install the Qdrant client:

```bash
pip install qdrant-client
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Test Connection

```bash
python3 qdrant_setup.py
```

### Use in Your Code

```python
from qdrant_setup import get_qdrant_client

# Get client
client = get_qdrant_client()

# List collections
collections = client.get_collections()
print(collections)

# Use the client for vector operations
# Example: Create a collection, add vectors, search, etc.
```

## Files Created

1. **`qdrant_setup.py`** - Qdrant client utility module
   - `get_qdrant_client()` - Returns configured Qdrant client
   - `test_connection()` - Tests connection and lists collections

2. **`.env`** - Environment variables (credentials stored here)
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
   - `QDRANT_CLUSTER_ID`
   - `QDRANT_ENDPOINT`

## Security Note

The API key is stored in the `.env` file. Make sure `.env` is in your `.gitignore` to avoid committing credentials to version control.

## Next Steps

You can now use Qdrant for:
- Storing vector embeddings from your verdict classifier
- Semantic search on job descriptions and resumes
- RAG (Retrieval-Augmented Generation) workflows
- Similarity search for candidate matching

Example integration with verdict classifier:
- Store job description + resume embeddings
- Search for similar candidates
- Use similarity scores to improve verdict predictions

