# Skin-Cancer-Image-Classification

## How to use it?
* First open anaconda
* Then click on environments
* Make a new environment
* Then open anaconda prompt terminal
* On anaconda prompt change your environment from base to whatever new environment you made
* download tensorflow, keras, streamlit, pandas, nupmy, matplotlib, scikit-learn
* Then type `streamlit run prediction.py` on the anaconda prompt


#### TO RUN THE JUPYTER NOTEBOOK ####
The python code is located in "CNN_Notebook.ipynb"

Download all of the images from the kaggle dataset: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
Save to a Folder in the same directory as the notebook called "Data", with the HAM10000_metadata.csv, HAM10000_images_part_1, and HAM10000_images_part_2

In addition to the images, please download the libraries that are listed in the first cell of the "CNN_Notebook.ipynb" in order for the Notebook to run.

#### UPDATES FROM REPORT ####
Since the creation of report, we have implemented callback functions that stop the model before it overfits. Removing these callback funtions can result in the testing accuracy to fluctuate. There are two separate pdfs that show the Model with callback functions and without callback functions. These pdfs are for model evaluation/comparison and accuracy visualization purposes.
