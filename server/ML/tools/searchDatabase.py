from server.ML.tools.objects.faceSearcher import FaceSearcher

from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from server.ML.tools.processing.faceCrop import crop_face

# -------------------------
# SETTINGS
# -------------------------
crop_face("tools/processing/pictures/Ishowspeed.png")
INPUT_IMAGE = (
    "tools\\processing\\stored-faces\\face_color.jpg"
)

TOP_PEOPLE = 5

DATASET_DIR = Path("Open Famous People Faces")


# -------------------------
# SEARCH
# -------------------------

searcher = FaceSearcher()


results = searcher.search(
    INPUT_IMAGE,
    top_k=200
)


if not results:

    print("No results found.")
    exit()


# -------------------------
# GET BEST IMAGE PER PERSON
# -------------------------

best_per_person = {}


for result in results:

    person = result["person"]

    if person not in best_per_person:

        best_per_person[person] = result

    else:

        if result["score"] > best_per_person[person]["score"]:

            best_per_person[person] = result


# Sort people by their best image score

people = sorted(
    best_per_person.values(),
    key=lambda x: x["score"],
    reverse=True
)


people = people[:TOP_PEOPLE]


# Helper: choose a representative image file for a person (first matching image)
def _choose_representative_file(person_name: str) -> str | None:
    person_dir = DATASET_DIR / person_name
    if not person_dir.exists() or not person_dir.is_dir():
        return None

    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        files = sorted(person_dir.glob(pattern))
        if files:
            return files[0].name

    return None


# Attach a representative filename to each result so downstream code can use it
for result in people:
    rep = _choose_representative_file(result["person"])
    if rep:
        result["file"] = rep
    else:
        result["file"] = ""


# -------------------------
# PRINT RESULTS
# -------------------------

print()

for i, result in enumerate(
    people,
    start=1
):

    print(
        f"{i}. "
        f"{result['person']} "
        f"- {result['score']:.4f}"
    )

    print(
        f"   {result['file']}"
    )


# -------------------------
# OPEN IMAGES
# -------------------------

images = []

titles = []


# Original image

original = Image.open(
    INPUT_IMAGE
)

images.append(
    original
)

titles.append(
    "INPUT IMAGE"
)


# Closest image for each person

for result in people:

    image_path = (
        DATASET_DIR
        / result["person"]
        / result["file"]
    )


    if not image_path.exists():

        print(
            "Image not found:",
            image_path
        )

        continue


    image = Image.open(
        image_path
    )


    images.append(
        image
    )

    titles.append(
        f"{result['person']}\n"
        f"{result['score']:.4f}"
    )


# -------------------------
# DISPLAY
# -------------------------

fig, axes = plt.subplots(
    1,
    len(images),
    figsize=(4 * len(images), 5)
)


# Handle case where there's only one image

if len(images) == 1:

    axes = [axes]


for ax, image, title in zip(
    axes,
    images,
    titles
):

    ax.imshow(image)

    ax.set_title(
        title
    )

    ax.axis(
        "off"
    )


plt.tight_layout()

plt.show()
