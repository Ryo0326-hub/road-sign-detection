from pathlib import Path
from PIL import Image
ps = sorted(Path("reports/viz").glob("*.jpg"))
cols, cell = 4, 640
rows = (len(ps) + cols - 1) // cols
sheet = Image.new("RGB", (cols * cell, rows * cell), (20, 20, 20))
for i, p in enumerate(ps):
    im = Image.open(p); im.thumbnail((cell, cell))
    sheet.paste(im, ((i % cols) * cell, (i // cols) * cell))
sheet.save("reports/contact_sheet.jpg", quality=90)
print(f"{len(ps)} images -> reports/contact_sheet.jpg")
