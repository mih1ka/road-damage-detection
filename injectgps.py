import piexif
from PIL import Image

def inject_gps(input_path, output_path, lat, lon):
    img = Image.open(input_path)

    def to_dms(value):
        d = int(value)
        m = int((value - d) * 60)
        s = round(((value - d) * 60 - m) * 60 * 10000)
        return ((d, 1), (m, 1), (s, 10000))

    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef:  b'N',
        piexif.GPSIFD.GPSLatitude:     to_dms(abs(lat)),
        piexif.GPSIFD.GPSLongitudeRef: b'E',
        piexif.GPSIFD.GPSLongitude:    to_dms(abs(lon)),
    }

    exif_bytes = piexif.dump({"GPS": gps_ifd})
    img.save(output_path, exif=exif_bytes)
    print(f"Done! Saved to: {output_path}")
    print(f"   Coordinates: {lat}, {lon}")

# ── EDIT THESE THREE LINES ────────────────────────────────────────
INPUT_IMAGE  = "/Users/mihika/Documents/Road-damage-detection/testroadimages/1000316761-road.jpg"   
OUTPUT_IMAGE = "demo_road_gps.jpg"     # what to call the output
LAT, LON     = 12.9692, 79.1559        # coordinates (see options below)
# ─────────────────────────────────────────────────────────────────

inject_gps(INPUT_IMAGE, OUTPUT_IMAGE, LAT, LON)