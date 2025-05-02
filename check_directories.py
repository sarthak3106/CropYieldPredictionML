import os
import sys

def check_project_structure():
    """
    Check if all required directories and files exist for the project
    """
    print("Checking project structure...")
    
    # Check for archive directory (where model files are stored)
    if not os.path.exists('archive'):
        print("WARNING: 'archive' directory not found")
        os.makedirs('archive', exist_ok=True)
        print("Created 'archive' directory")
    else:
        print("'archive' directory exists")
    
    # Check for preprocessor.pkl and model.pkl
    preprocessor_path = os.path.join('archive', 'preprocessor.pkl')
    model_path = os.path.join('archive', 'model.pkl')
    
    if not os.path.exists(preprocessor_path):
        print(f"ERROR: Preprocessor file not found at {preprocessor_path}")
        print("Please run the training pipeline to create this file")
    else:
        print(f"Found preprocessor at {preprocessor_path}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        print("Please run the training pipeline to create this file")
    else:
        print(f"Found model at {model_path}")
    
    # Check for templates directory
    if not os.path.exists('templates'):
        print("WARNING: 'templates' directory not found")
        os.makedirs('templates', exist_ok=True)
        print("Created 'templates' directory")
    else:
        print("'templates' directory exists")
        # Check for required templates
        for template in ['index.html', 'form.html']:
            if not os.path.exists(os.path.join('templates', template)):
                print(f"WARNING: '{template}' not found in 'templates' directory")
            else:
                print(f"Found template: {template}")
    
    # Check for source directory structure
    src_dirs = ['src', 'src/components', 'src/pipeline', 'src/utils']
    for dir_path in src_dirs:
        if not os.path.exists(dir_path):
            print(f"WARNING: '{dir_path}' directory not found")
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created '{dir_path}' directory")
        else:
            print(f"'{dir_path}' directory exists")
    
    # Check if prediction_pipeline.py exists
    if not os.path.exists('src/pipeline/prediction_pipeline.py'):
        print("WARNING: 'src/pipeline/prediction_pipeline.py' not found")
    else:
        print("'src/pipeline/prediction_pipeline.py' exists")
    
    print("\nSummary:")
    if not (os.path.exists(preprocessor_path) and os.path.exists(model_path)):
        print("Missing model files. Run train_pipeline.py to create them.")
    
    if not (os.path.exists('templates/index.html') and os.path.exists('templates/form.html')):
        print("Missing template files. Make sure you have index.html and form.html in the templates directory.")
    
    if not os.path.exists('src/pipeline/prediction_pipeline.py'):
        print("Missing prediction_pipeline.py. Make sure to create this file.")

if __name__ == "__main__":
    check_project_structure()