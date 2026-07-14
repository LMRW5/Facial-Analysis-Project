#This scirpt organizes the data parsed in webscraper.py and organizes it into a csv for easier lookup

import csv
from pathlib import Path


def parse_web_data(input_path: Path) -> dict:
    with input_path.open("r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f]

    rows = []
    i = 0

    while i < len(lines):
        while i < len(lines) and lines[i] == "":
            i += 1

        if i >= len(lines):
            break

        name = lines[i]
        if name in ("Phenotype Search", "About this page"):
            break

        i += 1
        group = ""
        description = []
        physical_traits = []
        literature = []
        similar_types = []

        while i < len(lines):
            line = lines[i]

            if line == "Group:":
                i += 1
                while i < len(lines) and lines[i] == "":
                    i += 1
                if i < len(lines):
                    group = lines[i]

            elif line == "Description:":
                i += 1
                while i < len(lines) and lines[i] != "Physical Traits:":
                    if lines[i]:
                        description.append(lines[i])
                    i += 1
                continue

            elif line == "Physical Traits:":
                i += 1
                while i < len(lines) and lines[i] != "Literature:":
                    if lines[i]:
                        physical_traits.append(lines[i])
                    i += 1
                continue

            elif line == "Literature:":
                i += 1
                while i < len(lines) and lines[i] != "Similar types:":
                    if lines[i]:
                        literature.append(lines[i])
                    i += 1
                continue

            elif line == "Similar types:":
                i += 1
                while i < len(lines) and lines[i] != "Phenotype Search":
                    if lines[i]:
                        similar_types.append(lines[i])
                    i += 1
                break

            i += 1

        rows.append(
            {
                "Name": name,
                "Group": group,
                "Description": " ".join(description),
                "Physical Traits": " ".join(physical_traits),
                "Literature": " ".join(literature),
                "Similar Types": ", ".join(similar_types),
            }
        )

    return rows


def process_dataset(dataset_dir: Path) -> None:
    input_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    processed = 0

    for folder in input_dirs:
        input_file = folder / "webData.txt"
        if not input_file.exists():
            continue

        rows = parse_web_data(input_file)
        output_file = folder / "info.csv"

        with output_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Name",
                    "Group",
                    "Description",
                    "Physical Traits",
                    "Literature",
                    "Similar Types",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        processed += 1
        print(f"Processed {folder.name}: {len(rows)} rows")

    print(f"Done! Wrote info.csv files for {processed} folders.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "dataset"
    process_dataset(dataset_dir)
