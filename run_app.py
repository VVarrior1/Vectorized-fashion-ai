import os
import subprocess
import sys

def run_app():
    """Run the Streamlit app with proper directory handling"""
    # Change to the script's directory to ensure correct relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check if the app directory exists
    if not os.path.exists('app'):
        print("Error: 'app' directory not found. Make sure you're in the project root directory.")
        return
    
    # Check if app files exist
    if not os.path.exists('app/app.py'):
        print("Error: 'app/app.py' not found.")
        return
    
    if not os.path.exists('app/run_streamlit_app.py'):
        print("Error: 'app/run_streamlit_app.py' not found.")
        return
    
    print("Starting AI Product Recommendation System...")
    print("Press Ctrl+C to stop the application.")
    
    # Run streamlit app with the launcher
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/run_streamlit_app.py"])
    except KeyboardInterrupt:
        print("\nApplication stopped.")
    except Exception as e:
        print(f"\nError running Streamlit app: {str(e)}")
        print("\nTrying alternative launch method...")
        try:
            # Alternative method - run directly with python
            subprocess.run([sys.executable, "app/run_streamlit_app.py"])
        except Exception as e2:
            print(f"\nError with alternative launch method: {str(e2)}")
            print("Make sure you have installed all requirements from requirements.txt")

if __name__ == "__main__":
    run_app() 