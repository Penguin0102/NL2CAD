import pickle
import os
import sys

def inspect_pkl(file_path):
    print(f"Inspecting {file_path}...")
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size / (1024**3):.2f} GB")
    
    try:
        with open(file_path, 'rb') as f:
            # Try to load just the header if it's a pandas pickle
            # or use a more memory-efficient way if possible
            # But standard pickle doesn't support this easily.
            # Let's try to see if it's a sequence of objects.
            count = 0
            while count < 5:
                try:
                    obj = pickle.load(f)
                    print(f"Object {count} type: {type(obj)}")
                    if hasattr(obj, 'head'):
                        print(f"Object {count} head:\n{obj.head()}")
                    count += 1
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error loading object {count}: {e}")
                    break
    except Exception as e:
        print(f"Failed to open/read file: {e}")

if __name__ == "__main__":
    inspect_pkl(r"G:\text2cad-data\train_data.pkl")
