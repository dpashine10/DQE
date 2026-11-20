import unittest

from backend.csv_io import MAX_BYTES, MAX_ROWS, preview, read_csv
from unittest.mock import patch


class CsvIoTests(unittest.TestCase):
    def test_upload_limit_is_512_mib(self):
        self.assertEqual(MAX_BYTES, 512 * 1024 * 1024)

    def test_rejects_files_over_the_byte_limit(self):
        with patch("backend.csv_io.MAX_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "512 MiB"):
                read_csv(b"x\n" + b"a" * 10)

    def test_row_limit_is_enforced(self):
        accepted = ("value\n" + "x\n" * 100_000).encode()
        self.assertEqual(len(read_csv(accepted)), 100_000)
        with patch("backend.csv_io.MAX_ROWS", 100_000):
            rejected = ("value\n" + "x\n" * 100_001).encode()
            with self.assertRaisesRegex(ValueError, "100,000 records"):
                read_csv(rejected)

    def test_cell_budget_and_late_binary_data_are_rejected(self):
        with patch("backend.csv_io.MAX_CELLS", 3):
            with self.assertRaisesRegex(ValueError, "data cells"):
                read_csv(b"a,b\n1,2\n3,4\n")
        with self.assertRaisesRegex(ValueError, "binary data"):
            read_csv(b"a\n" + b"x\n" * 40000 + b"\x00\n")

    def test_keeps_literal_values_and_detects_semicolon_csv(self):
        frame = read_csv(b"id;code;value\n001;?;10\n002;0007;20\n")
        self.assertEqual(frame.loc[0, "id"], "001")
        self.assertEqual(frame.loc[1, "code"], "0007")
        self.assertEqual(frame.loc[0, "value"], "10")

    def test_rejects_duplicate_headers_and_ragged_rows(self):
        with self.assertRaisesRegex(ValueError, "duplicate column"):
            read_csv(b"Name,name\nA,B\n")
        with self.assertRaisesRegex(ValueError, "expected 2 cells"):
            read_csv(b"name,value\nA\n")

    def test_rejects_header_only_files(self):
        with self.assertRaisesRegex(ValueError, "at least one data row"):
            read_csv(b"name,value\n")

    def test_preview_defaults_to_five_rows(self):
        frame = read_csv(("value\n" + "x\n" * 7).encode())
        self.assertEqual(len(preview(frame)["rows"]), 5)


if __name__ == "__main__":
    unittest.main()
