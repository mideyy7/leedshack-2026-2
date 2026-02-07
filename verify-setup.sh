#!/bin/bash

echo "==================================="
echo "Chain-Reaction Setup Verification"
echo "==================================="
echo ""

# Check backend
echo "Checking backend..."
if [ -f "backend/main.py" ] && [ -f "backend/requirements.txt" ]; then
    echo "✓ Backend files found"
else
    echo "✗ Backend files missing"
fi

# Check frontend
echo "Checking frontend..."
if [ -f "frontend/package.json" ] && [ -d "frontend/src" ]; then
    echo "✓ Frontend files found"
else
    echo "✗ Frontend files missing"
fi

# Check data directory
echo "Checking data directory..."
if [ -d "data" ]; then
    echo "✓ Data directory exists"
    if [ -f "data/shipments.csv" ]; then
        echo "  ✓ shipments.csv found"
    else
        echo "  ✗ shipments.csv not found (add this file to proceed with Phase 2)"
    fi
    if [ -f "data/weather.csv" ]; then
        echo "  ✓ weather.csv found"
    else
        echo "  ✗ weather.csv not found (add this file to proceed with Phase 3)"
    fi
else
    echo "✗ Data directory missing"
fi

echo ""
echo "==================================="
echo "Next Steps:"
echo "==================================="
echo "1. Add your CSV files to the data/ directory"
echo "2. Start backend: cd backend && uvicorn main:app --reload"
echo "3. Start frontend: cd frontend && npm start"
echo ""
