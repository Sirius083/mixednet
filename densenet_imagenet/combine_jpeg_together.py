# combine imagenet pictures together
import os
from PIL import Image
files = [filenames[i] for i in range(9)]
result = Image.new("RGB", (224*3, 224*3))

for index, file in enumerate(files):
  path = os.path.expanduser(file)
  img = Image.open(path)
  # resize inplace; parameter are input dimension
  img.resize((224, 224), Image.ANTIALIAS)
  # img.thumbnail((224, 224), Image.ANTIALIAS)
  x = index // 3 * 224
  y = index % 3 * 224
  w, h = img.size
  print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
  result.paste(img, (x, y, x + w, y + h))

save_path =r'E:\denseNet\densenet_imagenet\visualization\image.jpg'
result.save(os.path.expanduser(save_path))
