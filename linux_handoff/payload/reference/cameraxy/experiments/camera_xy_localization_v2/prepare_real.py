"""Build an image-bound real OV2735 manifest from an operator's optical-center survey."""
import argparse
import json
from pathlib import Path

from PIL import Image

from contract import digest
from infer import survey_label, survey_source


def create_manifest(survey_path, output):
    survey = json.loads(survey_path.read_text(encoding="utf8"))
    data_root = survey_path.parent.resolve()
    if output.parent.resolve() != data_root:
        raise ValueError("Keep the manifest alongside the survey for portable image paths")
    camera = survey["camera_metadata"]
    source = survey_source(camera)
    records, sample_ids, positions, image_hashes = [], set(), set(), set()
    coordinates = set()
    for position in survey["positions"]:
        fields = {"survey_position_id", "images", "camera_xy_target", "measurement"}
        if set(position) != fields:
            raise ValueError("Unexpected survey position fields")
        position_id = position["survey_position_id"]
        if not isinstance(position_id, str) or not position_id.strip() or position_id in positions:
            raise ValueError("Each physical survey position needs a unique identifier")
        positions.add(position_id)
        row = dict(survey_position_id=position_id, camera_metadata=camera,
                   camera_xy_target=position["camera_xy_target"], measurement=position["measurement"])
        pair, _ = survey_label(row)
        if pair in coordinates:
            raise ValueError("Use the same survey position ID for repeated captures at one XY")
        coordinates.add(pair)
        if not position["images"]:
            raise ValueError(f"No clean image frame for surveyed position {position_id}")
        for frame in position["images"]:
            if set(frame) != {"sample_id", "image"}:
                raise ValueError("Each surveyed image needs sample_id and relative image")
            sample = frame["sample_id"]
            relative_path = Path(frame["image"])
            path = (data_root / relative_path).resolve()
            if (not isinstance(sample, str) or not sample.strip() or sample in sample_ids
                    or relative_path.is_absolute() or not path.is_relative_to(data_root)):
                raise ValueError("Image path escapes survey, or sample ID is duplicated")
            sample_ids.add(sample)
            image_hash = digest(path)
            if image_hash in image_hashes:
                raise ValueError("Duplicate survey image bytes")
            image_hashes.add(image_hash)
            with Image.open(path) as im:
                im.load()
                if im.width*9 != im.height*16:
                    raise ValueError("Real input must use the measured 16:9 profile")
            records.append(dict(row, sample_id=sample, image=path.relative_to(data_root).as_posix(),
                                image_sha256=image_hash))
    if not records:
        raise ValueError("The survey contains no labeled frames")
    with output.open("x", encoding="utf8") as stream:
        for row in records:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    return dict(survey_sha256=digest(survey_path), manifest_sha256=digest(output),
                surveyed_positions=len(positions), frames=len(records), capture_source=source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("survey", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(create_manifest(args.survey, args.manifest), indent=2))


if __name__ == "__main__":
    main()
