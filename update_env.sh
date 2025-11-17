#!/bin/bash
# Update .env file with the API key
cat > .env << 'ENVEOF'
# Google Generative AI API Key
GEMINI_API_KEY=AIzaSyDS_WwRmh_wmmsftGpHu_QPjuwTRWS7w0I
ENVEOF
echo "✅ .env file updated successfully!"
