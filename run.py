import os
import webbrowser
from app import app

if __name__ == "__main__":
    # Get the port from environment variable or use default
    port = int(os.environ.get("PORT", 5000))
    
    # Print startup message
    print("\n" + "="*50)
    print("Multi-Agent Chatbot System")
    print("="*50)
    print("\nStarting web server...")
    print(f"Access the web interface at: http://localhost:{port}")
    print("\nPress Ctrl+C to stop the server")
    print("="*50 + "\n")
    
    # Open the browser automatically
    webbrowser.open(f"http://localhost:{port}")
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=port, debug=True) 