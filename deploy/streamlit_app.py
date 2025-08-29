#!/usr/bin/env python3
"""
Streamlit Cloud-compatible main app
Combines all dashboards into a single deployable app
"""

import streamlit as st
import sys
import os

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

# Page config
st.set_page_config(
    page_title="IRCTC Review Analysis Dashboard",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("🚂 IRCTC Analytics")
st.sidebar.markdown("---")

dashboard_choice = st.sidebar.radio(
    "Select Dashboard:",
    ["Professional Overview", "Department Analysis", "Root Cause Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Live Analytics Dashboard")
st.sidebar.markdown("Real-time insights from 90,000+ IRCTC app reviews")
st.sidebar.markdown("Last updated: Auto-refresh enabled")

# Load the selected dashboard
if dashboard_choice == "Professional Overview":
    st.markdown("# 🚂 Professional Overview")
    try:
        # Import and run professional dashboard
        sys.path.append(os.path.join(parent_dir, 'src', 'dashboard'))
        from professional_app import main as prof_main
        prof_main()
    except Exception as e:
        st.error(f"Error loading Professional Dashboard: {e}")
        st.info("Please ensure all required data files are present.")

elif dashboard_choice == "Department Analysis":
    st.markdown("# 📊 Department Analysis")
    try:
        from segregated_dashboard import main as seg_main
        seg_main()
    except Exception as e:
        st.error(f"Error loading Department Dashboard: {e}")
        st.info("Please ensure all required data files are present.")

elif dashboard_choice == "Root Cause Analysis":
    st.markdown("# 🔍 Root Cause Analysis")
    try:
        from root_cause_dashboard import main as root_main
        root_main()
    except Exception as e:
        st.error(f"Error loading Root Cause Dashboard: {e}")
        st.info("Please ensure all required data files are present.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")
st.sidebar.markdown("Data Mining Project 2025")