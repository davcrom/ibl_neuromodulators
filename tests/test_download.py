"""Tests for scripts/download.py — the catalog phase and its CLI."""
import pytest

import scripts.download as download


class TestParseArgs:
    def test_unknown_product_is_rejected(self):
        """Both product flags validate at parse time, not silently at build time."""
        with pytest.raises(SystemExit):
            download.parse_args(['--skip', 'photometry/nonsense'])
        with pytest.raises(SystemExit):
            download.parse_args(['--rebuild', 'photometry/nonsense'])

    def test_products_and_defaults(self):
        args = download.parse_args(['--skip', 'video/pose', '--workers', '4'])

        assert args.skip == ['video/pose']
        assert args.rebuild == []
        assert args.workers == 4
        assert args.retry_failed is False
