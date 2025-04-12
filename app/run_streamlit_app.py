import os
import sys
import streamlit as st
import importlib

def main():
    """
    Launch the Streamlit app with error handling
    """
    st.set_page_config(
        page_title="Fashion Recommender",
        page_icon="👔",
        layout="wide"
    )
    
    # Check for dependencies
    dependencies = {
        "streamlit": "streamlit",
        "numpy": "numpy",
        "pandas": "pandas",
        "PIL": "pillow",
        "sentence_transformers": "sentence-transformers",
        "transformers": "transformers",
        "dotenv": "python-dotenv"
    }
    
    missing_deps = []
    for module, package in dependencies.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing_deps.append(package)
    
    if missing_deps:
        st.error("⚠️ Missing Dependencies")
        st.write("The following packages are required but not installed:")
        for dep in missing_deps:
            st.write(f"- {dep}")
        
        st.write("\n### Installation Instructions")
        st.code("pip install -r requirements.txt", language="bash")
        return
    
    # Try to import FAISS but handle gracefully if not available
    try:
        import faiss
        st.success("✅ FAISS library is available for fast vector search")
    except ImportError:
        st.warning("⚠️ FAISS library is not available. Using fallback numpy-based search.")
        st.write("For better performance, install FAISS using the instructions below:")
        
        # Show platform-specific installation instructions
        if sys.platform == "darwin":
            st.write("**Mac Installation Options:**")
            st.write("1. Using Conda (recommended for Mac):")
            st.code("conda install -c conda-forge faiss-cpu", language="bash")
            st.write("2. Or try pip if conda doesn't work:")
            st.code("pip install faiss-cpu", language="bash")
            st.write("Note: FAISS can be difficult to install on Mac. The app will work without it using a numpy fallback.")
        else:
            st.code("pip install faiss-cpu", language="bash")
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ OpenAI API key not found")
        st.write("""
        For enhanced features like AI-generated descriptions, set your OpenAI API key:
        1. Create a `.env` file in the project root
        2. Add `OPENAI_API_KEY=your_api_key_here` to the file
        """)
        
        # Option to set API key in session
        api_key = st.text_input("Or enter your OpenAI API key here (not stored permanently):", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("✅ API key set for this session")
    
    # Try to import and run the main app
    try:
        from app import main as app_main
        app_main()
    except Exception as e:
        st.error(f"❌ Error starting the application: {str(e)}")
        st.write("Please make sure all dependencies are correctly installed.")
        st.write("Error details:")
        st.code(str(e))

if __name__ == "__main__":
    main() 