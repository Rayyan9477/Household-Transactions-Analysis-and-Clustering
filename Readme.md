# Household Transactions Analysis and Clustering

This project involves analyzing household transaction data to gain insights into spending patterns and behaviors. The analysis includes data cleaning, exploratory data analysis (EDA), clustering using K-Means, and visualization of customer segments. The project aims to provide a comprehensive understanding of household transactions and identify distinct customer segments based on their spending habits.


## Project Structure

```
.
├── data
│   └── Daily Household Transactions.csv  # Dataset file
├── notebooks
│   └── DEP_Task1.ipynb                   # Jupyter Notebook with analysis
├── src
│   └── analysis.py                       # Python script for analysis
├── README.md                             # Project documentation
└── requirements.txt                      # Dependencies
```

## Dependencies

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes the following libraries:

```pip-requirements
pandas
matplotlib
seaborn
scikit-learn
plotly
```

## Usage

1. **Data Cleaning and Preprocessing:**
   - Handle missing values and outliers.
   - Feature engineering to create new features from the `Date` column.

2. **Exploratory Data Analysis (EDA):**
   - Analyze the distribution of transaction amounts.
   - Analyze the number of transactions over time.
   - Analyze the breakdown of transactions by category and subcategory.
   - Correlation analysis between numerical features.

3. **Clustering:**
   - Use K-Means clustering to segment customers based on their spending habits.
   - Determine the optimal number of clusters using the Elbow Method.
   - Visualize customer segments using PCA.

4. **Visualization:**
   - Create interactive visualizations using Plotly.

## Conclusion

By following the steps outlined in this project, you can gain valuable insights into household transaction data and identify distinct customer segments based on their spending habits. This project demonstrates the use of data cleaning, exploratory data analysis, clustering, and visualization techniques to analyze and understand transaction data.


## Contact

- **LinkedIn:** [Rayyan Ahmed](https://www.linkedin.com/in/rayyan-ahmed9477/)
- **Email:** rayyanahmed265@yahoo.com

## Video Demonstration

[Watch the video demonstration here](#)
