import os

with open('App/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Rename Text2CAD to NL2CAD
content = content.replace('from Cad_VLM.models.text2cad import Text2CAD', 'from Cad_VLM.models.nl2cad import NL2CAD')
content = content.replace('text2cad = Text2CAD', 'nl2cad = NL2CAD')
content = content.replace('text2cad.load_state_dict', 'nl2cad.load_state_dict')
content = content.replace('text2cad.eval()', 'nl2cad.eval()')
content = content.replace('return text2cad', 'return nl2cad')
content = content.replace('Text2CAD:', 'NL2CAD:')
content = content.replace('text2cad-', 'nl2cad-')
content = content.replace('text2cad-header', 'nl2cad-header')
content = content.replace('text2cad-card', 'nl2cad-card')
content = content.replace('text2cad-label', 'nl2cad-label')
content = content.replace('text2cad-footer', 'nl2cad-footer')
content = content.replace('text2cad-generate-btn', 'nl2cad-generate-btn')
content = content.replace('text2cad-model3d', 'nl2cad-model3d')

with open('App/app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done! Applied renaming to restored app.py.')
