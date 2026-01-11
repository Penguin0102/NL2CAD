from PIL import Image
import os

print("Testing Pillow save...")
try:
    img = Image.new('RGB', (100, 100), color = (73, 109, 137))
    img.save('test_pillow.png')
    print("Successfully saved to test_pillow.png")
except Exception as e:
    print("Error saving with Pillow:", e)
print("Done.")
