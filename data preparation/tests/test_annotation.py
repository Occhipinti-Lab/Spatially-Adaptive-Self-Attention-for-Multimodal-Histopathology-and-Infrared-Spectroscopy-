from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from spectral_pathology.annotation import label_spectral_csv


class AnnotationTests(unittest.TestCase):
    def test_label_spectral_csv_matches_annotation_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            spectra_csv = workspace / "sample_spectra.csv"
            annotation_image = workspace / "sample.png"
            labeled_csv = workspace / "sample_labeled.csv"

            pd.DataFrame(
                {
                    "Pixel_Y": [0, 1, 2],
                    "Pixel_X": [0, 2, 2],
                    "Wavenumber_1000.00": [0.1, 0.2, 0.3],
                }
            ).to_csv(spectra_csv, index=False)

            image = Image.new("RGB", (3, 3), (0, 0, 0))
            image.putpixel((0, 0), (255, 0, 0))
            image.putpixel((2, 1), (103, 193, 66))
            image.save(annotation_image)

            summary = label_spectral_csv(spectra_csv, annotation_image, labeled_csv)
            labeled = pd.read_csv(labeled_csv)

            self.assertEqual(summary.input_rows, 3)
            self.assertEqual(summary.labeled_rows, 2)
            self.assertEqual(
                labeled["Label"].tolist(),
                ["malignant_epithelium", "non_malignant_epithelium"],
            )


if __name__ == "__main__":
    unittest.main()
