#!/bin/sh

# Wait for a moment to ensure volumes are mounted
sleep 5

# Clean up duplicate n8n-nodes-base if it exists
if [ -d "/home/node/.n8n/nodes/node_modules/n8n-nodes-base" ]; then
    rm -rf /home/node/.n8n/nodes/node_modules/n8n-nodes-base
fi

# Ensure custom directory exists and is initialized
mkdir -p /home/node/.n8n/custom
cd /home/node/.n8n/custom
if [ ! -f "package.json" ]; then
    npm init -y
fi

# Go to the node package directory and create the link
cd /backup/n8n-nodes-mcp-server
npm link

# Go to n8n custom directory and link to the package
cd /home/node/.n8n/custom
npm link n8n-nodes-mcp-server

# Start n8n
exec n8n 