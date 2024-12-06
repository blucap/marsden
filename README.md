# Marsden Fund Analysis

This repository contains tools and scripts for analyzing data related to the Marsden Fund. The goal is to process and visualize funding distributions, topic modeling, and institutional involvement using the data provided in supplementary files from various years.

## Features

- **Text Cleaning**: Automated cleaning of summaries for consistent analysis.
- **Topic Modeling**: Using TF-IDF and SVD to extract dominant research topics.
- **Visualizations**: Generate bar charts, line plots, and word clouds for funding and topics.
- **Institutional Analysis**: Compare funding across institutions over time.

## Prerequisites

- Python 3.8 or later
- Required libraries:
  - `pandas`
  - `matplotlib`
  - `wordcloud`
  - `skimpy`
  - `sklearn`
  - `numpy`

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/marsden-fund-analysis.git
   cd marsden-fund-analysis
   ```

2. Download the supplementary files listed below into the project directory:
   - [2024-Marsden-Fund-announcement-supplement_v10.xlsx](https://www.royalsociety.org.nz/assets/2024-Marsden-Fund-announcement-supplement_v10.xlsx)
   - [2023_Marsden_Announcements_New.xlsx](https://www.royalsociety.org.nz/assets/2023_Marsden_Announcements_New.xlsx)
   - [2022_Marsden_Fund_announcement_supplement-v14.xlsx](https://www.royalsociety.org.nz/assets/2022_Marsden_Fund_announcement_supplement-v14.xlsx)
   - [2021-Marsden-Fund-announcement-supplement-with-websheet-v3.xlsx](https://www.royalsociety.org.nz/assets/2021-Marsden-Fund-announcement-supplement-with-websheet-v3.xlsx)
   - [2020-Marsden-Fund-announcement-supplement_corrected.xlsx](https://www.royalsociety.org.nz/assets/2020-Marsden-Fund-announcement-supplement_corrected.xlsx)
   - [2019-Marsden-Fund-awarded-supplement_with-websheet-v2.xlsx](https://www.royalsociety.org.nz/assets/2019-Marsden-Fund-awarded-supplement_with-websheet-v2.xlsx)
   - [Marsden-2018-annoucement-supplement.xlsx](https://www.royalsociety.org.nz/assets/Uploads/Marsden-2018-annoucement-supplement.xlsx)
   - [Marsden-2017-annoucement-supplement-v2.xlsx](sandbox:/mnt/data/Marsden-2017-annoucement-supplement-v2.xlsx)
   - [Marsden-2016-annoucement-supplement.xlsx](https://www.royalsociety.org.nz/assets/documents/Marsden-2016-annoucement-supplement.xlsx)
   - [Marsden-Announcements-2008-2017.xlsx](https://www.royalsociety.org.nz/assets/Uploads/Marsden-Announcements-2008-2017.xlsx)

3. Make sure you have this file in your project directory:

    - **files_in_folder.csv**


4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook marsden.ipynb
   ```

5. Make sure to to set the project directory in the lines that mention `os.chdir`.

5. Run the notebook cells sequentially to process the data and generate visualizations.

## File Description

| File Name                                        | Description                                       |
|-------------------------------------------------|-------------------------------------------------|
| `marsden.ipynb`                                 | Main notebook for data analysis and visualization |
| `2019-Marsden-Fund-awarded-supplement_with-websheet-v2.xlsx` | Marsden Fund data for 2019                      |
| `2020-Marsden-Fund-announcement-supplement_corrected.xlsx` | Marsden Fund data for 2020                      |
| `2021-Marsden-Fund-announcement-supplement-with-websheet-v3.xlsx` | Marsden Fund data for 2021                      |
| `2022_Marsden_Fund_announcement_supplement-v14.xlsx` | Marsden Fund data for 2022                      |
| `2023_Marsden_Announcements_New.xlsx`           | Marsden Fund data for 2023                      |
| `2024-Marsden-Fund-announcement-supplement_v10.xlsx` | Marsden Fund data for 2024                      |
| `Marsden-2016-annoucement-supplement.xlsx`      | Marsden Fund data for 2016                      |
| `Marsden-2017-annoucement-supplement-v2.xlsx`   | Marsden Fund data for 2017                      |
| `Marsden-Announcements-2008-2017.xlsx`   | Marsden Fund data for 2008 to 2017                      |



## Outputs

- **Bar Charts**: Funding by institution, panel, and more.
- **Line Charts**: Normalized funding trends by panel.
- **Word Clouds**: Top terms from research summaries.
- **Data Files**: `marsden_projects.csv`, `marsden_teams.csv`

## Contributions

Feel free to open issues, submit pull requests, or suggest improvements to the repository.
# marsden
