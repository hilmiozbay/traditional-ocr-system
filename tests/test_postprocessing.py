"""
tests/test_postprocessing.py
=============================
postprocessing.py modülü için birim testler (pytest).
"""

import json
import sys
import pathlib
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from postprocessing import (
    fix_unicode,
    remove_control_chars,
    normalize_whitespace,
    fix_punctuation,
    apply_turkish_corrections,
    normalize_text,
    export_txt,
    export_json,
    export_csv,
    export_all,
)
from ocr import OCRResult, WordResult


# ─── Yardımcı ──────────────────────────────────

def make_ocr_result(text: str = "Merhaba dünya", word_count: int = 2) -> OCRResult:
    words = []
    for i, w in enumerate(text.split()[:word_count]):
        words.append(WordResult(
            text=w, confidence=85.0,
            bbox=(i * 80, 10, 70, 25),
            line_num=1, block_num=1, page_num=1,
        ))
    return OCRResult(
        full_text=text,
        words=words,
        avg_confidence=85.0,
        language="tur+eng",
        psm=3,
        image_shape=(400, 600),
    )


# ─── Metin normalleştirme ──────────────────────

class TestFixUnicode:
    def test_nfc_normalization(self):
        # NFD (ayrık) → NFC (birleşik)
        nfd = "o\u0308"   # ö = o + birleştirme umlaut
        assert fix_unicode(nfd) == "ö"

    def test_plain_text_unchanged(self):
        assert fix_unicode("Türkçe metin") == "Türkçe metin"


class TestRemoveControlChars:
    def test_removes_null_byte(self):
        assert "\x00" not in remove_control_chars("abc\x00def")

    def test_keeps_newline(self):
        assert "\n" in remove_control_chars("satır1\nsatır2")

    def test_keeps_tab(self):
        assert "\t" in remove_control_chars("sütun1\tsütun2")


class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        result = normalize_whitespace("çok   fazla   boşluk")
        assert "  " not in result

    def test_trims_line_edges(self):
        result = normalize_whitespace("  kenar boşluk  ")
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_limits_blank_lines(self):
        result = normalize_whitespace("satır1\n\n\n\n\nsatır2")
        assert result.count("\n") <= 3


class TestFixPunctuation:
    def test_removes_space_before_comma(self):
        result = fix_punctuation("merhaba , dünya")
        assert "merhaba," in result

    def test_removes_space_before_period(self):
        result = fix_punctuation("cümle sonu .")
        assert "sonu." in result


class TestApplyTurkishCorrections:
    def test_no_crash_on_normal_text(self):
        text = "Bu normal bir Türkçe cümledir."
        result = apply_turkish_corrections(text)
        assert isinstance(result, str)


class TestNormalizeText:
    def test_full_pipeline(self):
        dirty = "  Merhaba\x00,  dünya !\n\n\n"
        result = normalize_text(dirty)
        assert "\x00" not in result
        assert "  " not in result
        assert not result.startswith(" ")

    def test_empty_string(self):
        assert normalize_text("") == ""


# ─── Dışa aktarma fonksiyonları ───────────────

class TestExportTXT:
    def test_creates_file(self, tmp_path):
        result = make_ocr_result()
        out = str(tmp_path / "test.txt")
        export_txt(result, out)
        assert Path(out).exists()

    def test_file_contains_text(self, tmp_path):
        result = make_ocr_result("Python OCR testi")
        out = str(tmp_path / "out.txt")
        export_txt(result, out, include_metadata=False)
        content = Path(out).read_text(encoding="utf-8")
        assert "Python" in content

    def test_metadata_included(self, tmp_path):
        result = make_ocr_result()
        out = str(tmp_path / "meta.txt")
        export_txt(result, out, include_metadata=True)
        content = Path(out).read_text(encoding="utf-8")
        assert "Dil" in content or "OCR" in content


class TestExportJSON:
    def test_creates_valid_json(self, tmp_path):
        result = make_ocr_result()
        out = str(tmp_path / "out.json")
        export_json(result, out)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert "full_text" in data
        assert "metadata" in data

    def test_words_included(self, tmp_path):
        result = make_ocr_result()
        out = str(tmp_path / "words.json")
        export_json(result, out, include_words=True)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert "words" in data
        assert len(data["words"]) > 0

    def test_bbox_in_words(self, tmp_path):
        result = make_ocr_result()
        out = str(tmp_path / "bbox.json")
        export_json(result, out, include_words=True, include_bboxes=True)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert "bbox" in data["words"][0]


class TestExportCSV:
    def test_creates_file(self, tmp_path):
        result = make_ocr_result()
        out = str(tmp_path / "out.csv")
        export_csv(result, out)
        assert Path(out).exists()

    def test_has_header_and_rows(self, tmp_path):
        result = make_ocr_result("bir iki üç", word_count=3)
        out = str(tmp_path / "data.csv")
        export_csv(result, out)
        lines = Path(out).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 2  # başlık + en az 1 satır
        assert "text" in lines[0]


class TestExportAll:
    def test_all_formats(self, tmp_path):
        result = make_ocr_result()
        cfg = {
            "formats": ["txt", "json", "csv"],
            "include_confidence": True,
            "include_bboxes": True,
            "include_metadata": True,
            "encoding": "utf-8",
        }
        paths = export_all(result, str(tmp_path / "belge"), cfg)
        assert "txt" in paths
        assert "json" in paths
        assert "csv" in paths
        for p in paths.values():
            assert Path(p).exists()

    def test_unknown_format_skipped(self, tmp_path):
        result = make_ocr_result()
        cfg = {"formats": ["xyz"], "encoding": "utf-8"}
        paths = export_all(result, str(tmp_path / "out"), cfg)
        assert "xyz" not in paths
