"""
Main entry point for Crop Yield Prediction System
"""

import os
import sys
import subprocess
import webbrowser
import time
import argparse
from pathlib import Path


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = {
        'sklearn': 'scikit-learn',
        'tensorflow': 'tensorflow',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'pydantic': 'pydantic',
        'joblib': 'joblib',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'shap': 'shap',
        'keras_tuner': 'keras-tuner'
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            if module_name == 'sklearn':
                __import__('sklearn')
            elif module_name == 'keras_tuner':
                __import__('keras_tuner')
            else:
                __import__(module_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print("❌ Missing dependencies:", ', '.join(missing))
        print("\nInstall them using:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("✅ All dependencies are installed!")
    return True


def train_model(args):
    """Train the model"""
    print("\n" + "="*60)
    print("🚀 Starting Model Training")
    print("="*60)
    
    # First, ensure we can import the required modules
    try:
        from src.train import train_model as train
        train(model_type=args.model_type, epochs=args.epochs)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTrying to run training script directly...")
        cmd = [
            sys.executable, '-m', 'src.train',
            '--model', args.model_type,
            '--epochs', str(args.epochs)
        ]
        
        if args.attention:
            cmd.extend(['--attention', args.attention])
        
        subprocess.run(cmd)


def tune_model(args):
    """Run hyperparameter tuning"""
    print("\n" + "="*60)
    print("🔧 Running Hyperparameter Tuning")
    print("="*60)
    
    cmd = [
        sys.executable, '-m', 'src.tune',
        '--model', args.model_type,
        '--trials', str(args.trials),
        '--epochs', str(args.epochs)
    ]
    
    subprocess.run(cmd)


def start_api(args):
    """Start the FastAPI server"""
    print("\n" + "="*60)
    print("🌐 Starting API Server")
    print("="*60)
    
    # Check if API files exist
    if not os.path.exists('api/app.py'):
        print("❌ API files not found. Creating API directory...")
        os.makedirs('api', exist_ok=True)
        print("Please ensure api/app.py and api/schemas.py exist")
        return
    
    # Open browser after a short delay
    if args.open_browser:
        def open_browser():
            time.sleep(2)
            webbrowser.open(f'http://{args.host}:{args.port}/docs')
        
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
    
    cmd = [
        sys.executable, '-m', 'uvicorn',
        'api.app:app',
        '--host', args.host,
        '--port', str(args.port)
    ]
    
    if args.reload:
        cmd.append('--reload')
    
    subprocess.run(cmd)


def start_webapp(args):
    """Start the web application"""
    print("\n" + "="*60)
    print("🌾 Starting Web Application")
    print("="*60)
    
    # First ensure API is running
    api_process = None
    if args.start_api:
        print("Starting API server in background...")
        api_cmd = [
            sys.executable, '-m', 'uvicorn',
            'api.app:app',
            '--host', 'localhost',
            '--port', '8000'
        ]
        api_process = subprocess.Popen(api_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        print("✅ API server started")
    
    # Open the web app
    frontend_path = Path(__file__).parent / 'frontend' / 'index.html'
    if frontend_path.exists():
        webbrowser.open(f'file://{frontend_path.absolute()}')
        print(f"✅ Web app opened at: file://{frontend_path.absolute()}")
        print("⚠️ Make sure the API server is running on http://localhost:8000")
    else:
        print("❌ Frontend file not found. Creating frontend directory...")
        os.makedirs('frontend', exist_ok=True)
        print("Please ensure frontend/index.html exists")
    
    if api_process:
        print("\nPress Ctrl+C to stop the API server...")
        try:
            api_process.wait()
        except KeyboardInterrupt:
            api_process.terminate()
            print("\n✅ API server stopped")


def explain_model(args):
    """Run model explainability"""
    print("\n" + "="*60)
    print("🔍 Running Model Explainability")
    print("="*60)
    
    cmd = [sys.executable, '-m', 'src.explain']
    subprocess.run(cmd)


def predict_sample(args):
    """Run sample prediction"""
    print("\n" + "="*60)
    print("📊 Running Sample Prediction")
    print("="*60)
    
    cmd = [sys.executable, '-m', 'src.predict']
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description='Crop Yield Prediction System')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--model-type', default='advanced', choices=['advanced', 'bidirectional', 'weather'])
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--attention', default='multihead', choices=['none', 'single', 'multihead', 'temporal', 'feature'])
    
    # Tune command
    tune_parser = subparsers.add_parser('tune', help='Hyperparameter tuning')
    tune_parser.add_argument('--model-type', default='advanced', choices=['advanced', 'bidirectional', 'weather'])
    tune_parser.add_argument('--trials', type=int, default=10)
    tune_parser.add_argument('--epochs', type=int, default=20)
    
    # API command
    api_parser = subparsers.add_parser('api', help='Start API server')
    api_parser.add_argument('--host', default='localhost')
    api_parser.add_argument('--port', type=int, default=8000)
    api_parser.add_argument('--reload', action='store_true')
    api_parser.add_argument('--open-browser', action='store_true')
    
    # Web app command
    web_parser = subparsers.add_parser('web', help='Start web application')
    web_parser.add_argument('--start-api', action='store_true', help='Start API server automatically')
    
    # Explain command
    subparsers.add_parser('explain', help='Run model explainability')
    
    # Predict command
    subparsers.add_parser('predict', help='Run sample prediction')
    
    args = parser.parse_args()
    
    # Skip dependency check for certain commands
    skip_check_commands = ['api', 'web']
    
    if args.command not in skip_check_commands:
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Execute command
    if args.command == 'train':
        train_model(args)
    elif args.command == 'tune':
        tune_model(args)
    elif args.command == 'api':
        start_api(args)
    elif args.command == 'web':
        start_webapp(args)
    elif args.command == 'explain':
        explain_model(args)
    elif args.command == 'predict':
        predict_sample(args)
    else:
        # Show help
        parser.print_help()
        print("\n" + "="*60)
        print("Quick Start:")
        print("="*60)
        print("1. Train a model:      python main.py train")
        print("2. Start API server:   python main.py api --open-browser")
        print("3. Start web app:      python main.py web")
        print("4. Explain model:      python main.py explain")
        print("5. Run prediction:     python main.py predict")


if __name__ == "__main__":
    main()