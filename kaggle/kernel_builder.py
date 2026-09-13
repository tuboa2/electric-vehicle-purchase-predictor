import json
from pathlib import Path
from typing import Any


class KernelBuilder:
    """
    Builds self-contained Kaggle submission kernels and metadata bundles.
    Enforces offline execution rules (enable_internet=False) for code competitions.
    """

    def __init__(self, templates_dir: str | Path = "templates"):
        self.templates_dir = Path(templates_dir)

    def generate_kernel_metadata(
        self,
        kernel_slug: str,
        title: str,
        code_file: str,
        competition_slug: str,
        dataset_sources: list[str],
        enable_gpu: bool = True,
        enable_internet: bool = False,
        kernel_type: str = "script",
    ) -> dict[str, Any]:
        """
        Generates Kaggle kernel-metadata.json dictionary adhering strictly
        to offline code competition submission requirements.
        """
        metadata = {
            "id": kernel_slug,
            "title": title,
            "code_file": code_file,
            "language": "python",
            "kernel_type": kernel_type,
            "is_private": "true",
            "enable_gpu": "true" if enable_gpu else "false",
            "enable_internet": "true" if enable_internet else "false",
            "dataset_sources": dataset_sources,
            "competition_sources": [competition_slug],
            "kernel_sources": [],
        }
        return metadata

    def build_submission_bundle(
        self,
        output_dir: str | Path,
        kernel_slug: str,
        title: str,
        inference_code: str,
        competition_slug: str,
        dataset_sources: list[str],
        enable_gpu: bool = True,
    ) -> Path:
        bundle_path = Path(output_dir)
        bundle_path.mkdir(parents=True, exist_ok=True)

        code_filename = "inference.py"
        code_path = bundle_path / code_filename
        code_path.write_text(inference_code, encoding="utf-8")

        metadata = self.generate_kernel_metadata(
            kernel_slug=kernel_slug,
            title=title,
            code_file=code_filename,
            competition_slug=competition_slug,
            dataset_sources=dataset_sources,
            enable_gpu=enable_gpu,
            enable_internet=False, # Mandatory offline inference
        )

        metadata_file = bundle_path / "kernel-metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return bundle_path
