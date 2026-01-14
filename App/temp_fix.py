import os
import re

file_path = r'c:\Users\86185\Desktop\NL2CAD\App\app.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the indentation of the return statement
new_content = re.sub(r'\n\s+return stl_path, "Generated Success", zip_path', '\n        return stl_path, "Generated Success", zip_path', content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Successfully fixed indentation in app.py")
