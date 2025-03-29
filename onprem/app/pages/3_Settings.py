import os
import sys
import yaml
import streamlit as st

# Add parent directory to path to allow importing when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import from parent modules
from webapp import DEFAULT_YAML_FPATH, read_config, DEFAULT_YAML
from utils import hide_webapp_sidebar_item

def main():
    """
    Settings page for configuring the application
    """
    # Hide webapp sidebar item
    hide_webapp_sidebar_item()
    
    st.header("Settings")
    st.subheader("Configuration File")
    
    cfg, _ = read_config()
    
    # Display and edit current configuration
    st.info(f"Configuration file location: {DEFAULT_YAML_FPATH}")
    
    # Edit YAML directly
    with open(DEFAULT_YAML_FPATH, "r") as f:
        yaml_content = f.read()
    
    edited_yaml = st.text_area("Edit Configuration", yaml_content, height=400)
    
    if st.button("Save Configuration"):
        try:
            # Validate YAML format by parsing it
            yaml.safe_load(edited_yaml)
            
            # Save the edited YAML
            with open(DEFAULT_YAML_FPATH, "w") as f:
                f.write(edited_yaml)
            
            st.success("Configuration saved successfully. Please restart the app for changes to take effect.")
        except Exception as e:
            st.error(f"Error saving configuration: {str(e)}")
    
    # Reset to default
    if st.button("Reset to Default Configuration"):
        try:
            from onprem import utils as U
            yaml_content = DEFAULT_YAML.format(datadir=U.get_datadir()).strip()
            yaml_content = yaml_content.replace('PROMPT_VARIABLE', '{prompt}')
            
            with open(DEFAULT_YAML_FPATH, "w") as f:
                f.write(yaml_content)
            
            st.success("Configuration reset to default. Please restart the app for changes to take effect.")
        except Exception as e:
            st.error(f"Error resetting configuration: {str(e)}")


if __name__ == "__main__":
    main()