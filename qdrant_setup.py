"""
Qdrant client setup and connection test.
"""

try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠️  qdrant-client not installed. Install with: pip install qdrant-client")

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Qdrant configuration - can be overridden by environment variables
QDRANT_URL = os.getenv("QDRANT_URL", "https://df856a23-961c-45c7-a091-6b755c9343ef.us-east4-0.gcp.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.mB-rfMITSHSC3VJfEtHqSTkQ1dM7UXxKkrSi30foT60")
QDRANT_CLUSTER_ID = os.getenv("QDRANT_CLUSTER_ID", "df856a23-961c-45c7-a091-6b755c9343ef")
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT", "https://df856a23-961c-45c7-a091-6b755c9343ef.us-east4-0.gcp.cloud.qdrant.io")

def get_qdrant_client():
    """
    Create and return a Qdrant client instance.
    """
    if not QDRANT_AVAILABLE:
        raise ImportError("qdrant-client package is not installed. Install with: pip install qdrant-client")
    
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        return client
    except Exception as e:
        print(f"❌ Error creating Qdrant client: {e}")
        raise

def test_connection():
    """
    Test the Qdrant connection and list collections.
    """
    print("🔌 Testing Qdrant connection...")
    print(f"   URL: {QDRANT_URL}")
    print(f"   Cluster ID: {QDRANT_CLUSTER_ID}")
    
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        
        print(f"\n✅ Connection successful!")
        print(f"   Collections found: {len(collections.collections)}")
        
        if collections.collections:
            print("\n   Collection names:")
            for collection in collections.collections:
                print(f"      - {collection.name}")
        else:
            print("   No collections found (this is normal for a new cluster)")
        
        return client
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        return None

if __name__ == "__main__":
    test_connection()

