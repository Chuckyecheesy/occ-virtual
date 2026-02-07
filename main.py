"""
main.py - Main Integration Point for Off-Campus Housing Recommendation System

Orchestrates the integration of:
1. Affordability Model (affordability_model.py) - ML-powered rent prediction
2. Flask API (fake_api.py) - RESTful API for apartment filtering
3. Streamlit Dashboard (dashboard.py) - Interactive UI for recommendations

This module provides:
- A unified entry point to run different components
- CLI interface for direct affordability predictions
- Model initialization and caching
- Complete integration of all system modules
"""

import sys
import argparse
import subprocess
from typing import Dict, Any
import pandas as pd

from affordability_model import (
    load_historical_data,
    train_model,
    load_sublets,
    predict_safe_rent,
    recommend_apartments
)


# ========== Global Model Cache ==========
_model = None
_sublets_df = None


def initialize_system():
    """
    Initialize the affordability model and load apartment data.
    Called once at system startup.
    """
    global _model, _sublets_df
    
    print("🔧 Initializing Off-Campus Housing Recommendation System...")
    
    # Load historical data and train model
    print("  → Loading historical student data...")
    df_train = load_historical_data()
    print(f"    ✓ Loaded {len(df_train)} historical records")
    
    print("  → Training affordability model...")
    _model = train_model(df_train)
    print("    ✓ Model trained successfully")
    
    # Load apartment listings
    print("  → Loading apartment listings...")
    _sublets_df = load_sublets()
    print(f"    ✓ Loaded {len(_sublets_df)} apartments")
    
    print("✅ System initialized!\n")
    
    return _model, _sublets_df


def get_model():
    """Get the cached affordability model, initialize if needed."""
    global _model
    if _model is None:
        initialize_system()
    return _model


def get_sublets():
    """Get the cached sublets dataframe, initialize if needed."""
    global _sublets_df
    if _sublets_df is None:
        initialize_system()
    return _sublets_df


# ========== CLI Mode: Interactive Affordability Calculator ==========
def run_cli_mode():
    """
    Run interactive CLI mode for affordable housing prediction.
    Users input their financial information and receive apartment recommendations.
    """
    print("=" * 60)
    print("💰 INTERACTIVE AFFORDABILITY CALCULATOR")
    print("=" * 60)
    print("\nEnter your financial information (or press Enter for defaults):\n")
    
    # Collect user input
    def get_input(prompt: str, default: float) -> float:
        try:
            value = input(f"{prompt} (default: ${default}): ").strip()
            return float(value) if value else default
        except ValueError:
            print(f"  Invalid input. Using default: ${default}")
            return default
    
    user_input = {
        'tuition': get_input("Annual Tuition", 8000),
        'bank_balance': get_input("Current Bank Balance", 1500),
        'part_time_income': get_input("Monthly Part-Time Income", 1000),
        'internship_income': get_input("Monthly Internship Income", 0),
        'scholarships': get_input("Total Scholarships", 2000),
        'loans': get_input("Total Loans", 1500),
        'months': get_input("Months of Housing Needed", 8),
    }
    
    print("\n" + "=" * 60)
    print("📊 PROCESSING YOUR INFORMATION...")
    print("=" * 60 + "\n")
    
    # Get model and predict safe rent
    model = get_model()
    safe_rent = predict_safe_rent(model, user_input)
    
    # Display financial summary
    total_income = user_input['part_time_income'] + user_input['internship_income']
    total_support = user_input['scholarships'] + user_input['loans']
    
    print("💵 YOUR FINANCIAL SUMMARY:")
    print(f"  Income (Monthly):          ${total_income:>10.2f}")
    print(f"  Financial Support:         ${total_support:>10.2f}")
    print(f"  Bank Balance:              ${user_input['bank_balance']:>10.2f}")
    print(f"  Tuition (Annual):          ${user_input['tuition']:>10.2f}")
    
    print("\n🏠 AI RECOMMENDATION:")
    print(f"  Safe Monthly Rent:         ${safe_rent:>10.2f}")
    print(f"  (Based on your financial situation for {int(user_input['months'])} months)")
    
    # Get recommendations
    sublets_df = get_sublets()
    recommendations = recommend_apartments(safe_rent, sublets_df)
    
    print(f"\n🎯 APARTMENT RECOMMENDATIONS:")
    print(f"  Found {len(recommendations)} apartments under ${safe_rent:.2f}/month\n")
    
    if len(recommendations) > 0:
        # Sort by rent
        recommendations = recommendations.sort_values('monthly_rent')
        
        print(f"  {'Address':<40} {'Rent':>10}")
        print("  " + "-" * 52)
        for idx, row in recommendations.iterrows():
            address = row['address'][:38] if len(str(row['address'])) > 38 else row['address']
            rent = row['monthly_rent']
            savings = safe_rent - rent
            print(f"  {address:<40} ${rent:>9.2f}")
            if savings > 0:
                print(f"    └─ Save ${savings:.2f}/month")
    else:
        if len(sublets_df) > 0:
            print(f"  ⚠️  No apartments under ${safe_rent:.2f}/month")
            print(f"  Available range: ${sublets_df['monthly_rent'].min():.2f} - ${sublets_df['monthly_rent'].max():.2f}")
        else:
            print("  ⚠️  No apartments available in database")
    
    print("\n" + "=" * 60)


# ========== API Mode: Start Flask Server ==========
def run_api_server(host: str = '127.0.0.1', port: int = 5000):
    """
    Start the Flask API server for housing recommendations.
    
    Args:
        host: Server host address (default: 127.0.0.1)
        port: Server port (default: 5000)
    """
    print("=" * 60)
    print("🚀 STARTING FLASK API SERVER")
    print("=" * 60)
    print(f"\n📡 Server running on http://{host}:{port}")
    print("\nAPI Endpoints:")
    print(f"  GET /api/affordable/<budget>")
    print("    Returns apartments with monthly_rent <= budget\n")
    print("Example:")
    print(f"  curl http://{host}:{port}/api/affordable/800\n")
    print("Press Ctrl+C to stop the server\n")
    print("=" * 60 + "\n")
    
    # Import and run Flask app
    from fake_api import app
    app.run(host=host, port=port, debug=True)


# ========== Dashboard Mode: Start Streamlit App ==========
def run_dashboard():
    """
    Start the Streamlit dashboard for interactive UI.
    Uses subprocess to launch streamlit from command line.
    """
    print("=" * 60)
    print("📊 STARTING STREAMLIT DASHBOARD")
    print("=" * 60)
    print("\n🌐 Opening dashboard in your browser...")
    print("   (If browser doesn't open, visit http://localhost:8501)\n")
    print("Press Ctrl+C to stop the dashboard\n")
    print("=" * 60 + "\n")
    
    # Use subprocess to run streamlit
    try:
        subprocess.run(
            ["streamlit", "run", "dashboard.py"],
            cwd="/Applications/occvirtual"
        )
    except FileNotFoundError:
        print("❌ Error: Streamlit not installed")
        print("   Install with: pip install streamlit")
        sys.exit(1)


# ========== Complete Pipeline: Affordability → API → Dashboard ==========
def run_complete_pipeline():
    """
    Run the complete integrated pipeline:
    1. Initialize affordability model
    2. Start Flask API
    3. Launch Streamlit dashboard
    
    Note: This would require multiple terminals/processes.
    Currently implemented as sequential (API first, then dashboard).
    """
    print("=" * 60)
    print("🔗 COMPLETE INTEGRATION PIPELINE")
    print("=" * 60)
    print("\nStarting all components in sequence...")
    print("\nPipeline: Affordability → API → Dashboard\n")
    
    # Step 1: Initialize system
    print("Step 1: Initialize Affordability Model")
    initialize_system()
    
    # Step 2: Prompt user to start API
    print("\nStep 2: Flask API Server")
    response = input("Start Flask API server? (y/n): ").strip().lower()
    if response == 'y':
        run_api_server()
    else:
        print("Skipping API server")
    
    # Step 3: Prompt user to start dashboard
    print("\nStep 3: Streamlit Dashboard")
    response = input("Start Streamlit dashboard? (y/n): ").strip().lower()
    if response == 'y':
        run_dashboard()
    else:
        print("Skipping dashboard")
    
    print("\n✅ Pipeline complete!")


# ========== Main Entry Point ==========
def main():
    """
    Main entry point with command-line argument parsing.
    Provides options to run different components or the complete system.
    """
    parser = argparse.ArgumentParser(
        description="Off-Campus Housing Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --cli              Run interactive CLI calculator
  python main.py --api              Start Flask API server (port 5000)
  python main.py --api --port 8080  Start Flask API on custom port
  python main.py --dashboard        Start Streamlit dashboard
  python main.py --full             Run complete integrated pipeline
  python main.py                    Show this help message
        """
    )
    
    parser.add_argument(
        '--cli',
        action='store_true',
        help='Run interactive CLI affordability calculator'
    )
    
    parser.add_argument(
        '--api',
        action='store_true',
        help='Start Flask API server'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port for Flask API server (default: 5000)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host for Flask API server (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Start Streamlit dashboard'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run complete integrated pipeline (affordability → API → dashboard)'
    )
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Execute based on arguments
    if args.cli:
        run_cli_mode()
    elif args.api:
        run_api_server(host=args.host, port=args.port)
    elif args.dashboard:
        run_dashboard()
    elif args.full:
        run_complete_pipeline()


if __name__ == '__main__':
    main()
