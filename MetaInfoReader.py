import piexif
from PIL import Image, ExifTags
exif_dict = piexif.load(r"F:\diffusers-test\imgs\1.7397833969172887.jpg")
print(exif_dict)