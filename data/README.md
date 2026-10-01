# Data

Course data (Piazza posts, lecture slides, assignment handouts) is **not included** in this repository for student-privacy and course-copyright reasons.

To reproduce, collect your own course materials with the extractors in `scripts/` (`extract_slides*.py`, `extract_assignments*.py`, `piazza_python_extractor.py`), then build splits with `scripts/combine_all_data.py` and `scripts/create_splits_from_new_data.py`. Expected record format: see `src/data_preparation.py`.
