import os

file_path = r'c:\Users\86185\Desktop\NL2CAD\App\app.py'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Shrink button
    if 'height: 52px !important;' in line:
        line = line.replace('height: 52px !important;', 'height: 40px !important;')
    if 'font-size: 1rem !important;' in line and '.synthesize-btn' in ''.join(new_lines[-10:]): # Context check
        line = line.replace('font-size: 1rem !important;', 'font-size: 0.9rem !important;')
    
    # Change success message
    if 'return stl_path, f"✅ Generated: {text}", zip_path' in line:
        line = line.replace('return stl_path, f"✅ Generated: {text}", zip_path', '        return stl_path, "Generated Success", zip_path')
    elif 'f"✅ Generated: {text}"' in line: # Fallback for slightly different matching
        line = line.replace('f"✅ Generated: {text}"', '"Generated Success"')
        
    new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Successfully patched app.py")
