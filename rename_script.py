import os

files_to_update = [
    'Cad_VLM/train.py',
    'Cad_VLM/test.py', 
    'Cad_VLM/test_user_input.py',
    'Cad_VLM/dataprep/t2c_dataset.py',
    'Cad_VLM/models/loss.py',
    'App/app.py',
    'CadSeqProc/merge_vlm_minimal.py',
    'CadSeqProc/json2vec.py',
    'README.md'
]

for f in files_to_update:
    if os.path.exists(f):
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        content = content.replace('Text2CAD', 'NL2CAD').replace('text2cad', 'nl2cad')
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f'Updated: {f}')

print('Done!')
