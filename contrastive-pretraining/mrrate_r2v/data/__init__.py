"""Data layer: manifest -> Dataset -> preprocessed [1, X, Y, Z] volume + report text.

See `mrrate_r2v/data/README.md` for how to use it. Layered by concern:

    storage.py    read bytes out of un-extracted archives          (stdlib only)
    manifest.py   which (study, series) pairs exist, and where     (stdlib only)
    reports.py    where the conditioning text comes from           (stdlib only)
    geometry.py   what voxel grid each series is resampled onto     (stdlib only)
    dataset.py    puts it together and returns tensors             (torch)

**Re-exports here are lazy** (PEP 562). `from mrrate_r2v.data import MRReportToVolumeDataset`
works as usual, but `import mrrate_r2v.data.manifest` does *not* drag in torch -- which is what
lets an interpreter with pyarrow and no torch build a shards manifest, and is why there is no
duplicate standalone builder script. `test_data_dataset.py` asserts this stays true; don't convert
these into plain top-level imports.
"""

import importlib

# public name -> submodule that defines it
_EXPORTS = {
    # dataset.py (torch)
    "ARCHIVE_ACCESS_MODES": "dataset",
    "BUCKET_ORDERS": "dataset",
    "SERIES_SELECTION_MODES": "dataset",
    "GeometryBucketBatchSampler": "dataset",
    "MRReportToVolumeDataset": "dataset",
    "R2VDatasetConfig": "dataset",
    "collate_fn_r2v": "dataset",
    "compute_modality_balance_weights": "dataset",
    "get_modality_balanced_sampler": "dataset",
    # geometry.py
    "FALLBACK_GEOMETRY_KEY": "geometry",
    "FIXED_GEOMETRY_KEY": "geometry",
    "GEOMETRY_MODES": "geometry",
    "NV_BRAIN_FOV_MM": "geometry",
    "GeometryPolicy": "geometry",
    "GeometrySpec": "geometry",
    "build_geometry_table": "geometry",
    "dhw_to_xyz": "geometry",
    "xyz_to_dhw": "geometry",
    # manifest.py
    "DEFAULT_EXCLUDED_MODALITIES": "manifest",
    "MANIFEST_FIELDS": "manifest",
    "ManifestRow": "manifest",
    "MetadataStore": "manifest",
    "SeriesMeta": "manifest",
    "build_manifest_rows": "manifest",
    "build_manifest_rows_from_data_path_zips": "manifest",
    "build_manifest_rows_from_shards_parquet": "manifest",
    "build_shard_report_index": "manifest",
    "is_eligible": "manifest",
    "read_manifest_csv": "manifest",
    "series_id_from_path": "manifest",
    "verify_archive_locators_sample": "manifest",
    "write_manifest_csv": "manifest",
    "write_report_index_csv": "manifest",
    # reports.py
    "REPORT_SECTION_NAMES": "reports",
    "ReportRecord": "reports",
    "SentenceJSONLReportStore": "reports",
    "ShardReportStore": "reports",
    "StructuredReportStore": "reports",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return __all__
