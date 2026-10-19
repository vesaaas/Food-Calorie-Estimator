# Dataset Layer

## Selection and source

Food-101 was selected because it is a standard real-world food image classification
dataset supported directly by torchvision.datasets.Food101. It contains 101 food
categories and 101,000 JPEG images collected for dish recognition. The official
source is the Food-101 dataset page:
https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

Food-101 is a classification dataset. It does not provide calories, portion sizes,
weights, protein, carbohydrates, fat, or other nutrition measurements. It can support
food recognition in the Food Calorie Estimator, but calorie estimation requires a
separate nutrition data layer and serving-size assumptions.

## Original organization

Torchvision expects the following root:

~~~text
ml/data/raw/
├── food-101.tar.gz
└── food-101/
    ├── images/<class>/<image>.jpg
    └── meta/
        ├── classes.txt
        ├── labels.txt
        ├── train.json
        ├── test.json
        ├── train.txt
        └── test.txt
~~~

The official organization contains all 101 classes, with 750 training images and
250 test images per class. The verified totals are 75,750 training images and
25,250 test images.

The downloader stores the archive and extraction in ml/data/raw/, reuses a valid
local extraction, and supports explicit checksum-based repair. Metadata and every
referenced image path are validated before the pipeline is built.

## Development classes and labels

The current development subset contains exactly these ten Food-101 labels:

| Application label | Food-101 class |
| ---: | --- |
| 0 | pizza |
| 1 | hamburger |
| 2 | sushi |
| 3 | steak |
| 4 | ice_cream |
| 5 | donuts |
| 6 | omelette |
| 7 | french_fries |
| 8 | chicken_wings |
| 9 | spaghetti_bolognese |

The requested names are checked against classes.txt. Unknown names raise an error;
no unrelated classes are substituted. Food101Subset remaps the original Food-101
indices into this contiguous application label order.

## Split strategy

The official Food-101 test split is preserved and is never used to create validation
data. Selected official training samples are divided into 80% training and 20%
validation per class using the deterministic default seed 42.

Verified counts:

| Split | Total | Per class |
| --- | ---: | ---: |
| Training | 6,000 | 600 |
| Validation | 1,500 | 150 |
| Official test | 2,500 | 250 |

Paths are ordered deterministically before the class-specific seeded shuffle. The
split code also hashes selected training image bytes and keeps identical-content
groups together. This prevents the duplicate hamburger images found in Food-101
from crossing train and validation. Verification hashes all selected split files
and rejects identical bytes across split boundaries.

The implementation verifies that:

- every selected training sample belongs to exactly train or validation;
- every selected official test sample remains in test;
- each split contains every selected class;
- class-to-label mappings agree across all splits;
- file paths exist, are nonempty, and are under the expected class directory;
- official train/test paths do not overlap;
- selected split paths and image bytes do not cross split boundaries.

## Git and storage policy

Raw images and generated processed outputs are excluded by .gitignore:

~~~text
ml/data/raw/
ml/data/processed/
*.pt
*.pth
*.ckpt
~~~

The dataset is several gigabytes and is reproducible through the downloader, so
committing image data or generated reports would make the source repository large
and environment-specific. The code, configuration, and verification behavior are
kept in Git; local data is re-created when needed.

## Limitations and implications

- Training labels intentionally contain some noise.
- Food-101 class recognition does not determine portion size or recipe details.
- A single predicted class may correspond to many nutritional profiles.
- Exact calories cannot be inferred from Food-101 alone.
- The current leakage check catches path overlap and identical image bytes, but not
  perceptual near-duplicates.
- The current subset is a development choice and does not represent all 101 classes.

Nutrition mapping is implemented separately in Phase 4. A representative USDA
record and default serving do not remove the class and portion ambiguities above.
