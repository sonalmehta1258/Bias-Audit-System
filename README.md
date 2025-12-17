LexiGuard: Ethical AI & Legal Bias Detection

Project Overview

LexiGuard is a final-year Computer Science project designed to audit and mitigate bias in AI systems used for judicial decision-making. It integrates NLP model training, fairness metric calculation (Demographic Parity), and societal trend analysis into a single unified dashboard.

Dataset Structure

The project assumes three CSV files are present in the root directory:

data1.csv (Legal Bias Dataset): Contains case facts, verdicts, and bias labels.

data2.csv (Country Time Freq): Contains historical frequency of stereotype words.

data3.csv (Type 1 Anti-Stereotype): Used for validation unit tests.

Installation & Setup

Install Dependencies:
Make sure you have Python installed, then run:

pip install -r requirements.txt


Run the Application:
Execute the Streamlit app:

streamlit run lexiguard_app.py


Using the Dashboard:

Module 1: Click "Train Model" to simulate the learning process on the legal dataset.

Module 2: Check the "Demographic Parity Ratio" card. If it is below 0.8, the model is flagged as biased. Use the slider to simulate threshold adjustment.

Module 3: Select words like "fat" or "sexy" to see how their usage has changed over the years in different countries.

Module 4: Review the anti-stereotype unit tests.

Key Technologies

Python 3.8+

Streamlit: For the web interface.

Scikit-Learn: For the Random Forest Classifier and TF-IDF Vectorizer.

Matplotlib/Seaborn: For data visualization.
