from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

img = Image.open("demo_road_gps.jpg")
exif = img._getexif()
readable = {TAGS.get(k, k): v for k, v in exif.items()}
gps = readable.get('GPSInfo')
print({GPSTAGS.get(k, k): v for k, v in gps.items()})