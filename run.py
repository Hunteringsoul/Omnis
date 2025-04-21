import os
from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*50)
    print("Multi-Agent Chatbot System")
    print("="*50)
    print(f"PORT environment variable: {os.environ.get('PORT', 'Not set')}")
    print(f"Using port: {port}")
    print("\nStarting web server...")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=port)
