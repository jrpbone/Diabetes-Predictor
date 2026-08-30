import struct
import unittest
from pathlib import Path


ASSET_DIRECTORY = Path(__file__).resolve().parents[1] / "assets"


class AppIconTests(unittest.TestCase):
    def test_windows_icon_contains_title_bar_and_large_sizes(self):
        icon_data = (ASSET_DIRECTORY / "diabetes-predictor.ico").read_bytes()
        reserved, icon_type, image_count = struct.unpack("<HHH", icon_data[:6])
        self.assertEqual((0, 1), (reserved, icon_type))

        sizes = []
        for index in range(image_count):
            offset = 6 + index * 16
            width, height = struct.unpack("<BB", icon_data[offset : offset + 2])
            sizes.append((width or 256, height or 256))

        self.assertGreaterEqual(image_count, 7)
        self.assertTrue({16, 24, 32, 48, 64, 128, 256}.issubset({w for w, _h in sizes}))


if __name__ == "__main__":
    unittest.main()
