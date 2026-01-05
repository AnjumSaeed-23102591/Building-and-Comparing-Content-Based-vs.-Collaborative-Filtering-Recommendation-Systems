# Building-and-Comparing-Content-Based-vs.-Collaborative-Filtering-Recommendation-SystemsHere’s a comprehensive **README** file template for your project based on the provided code:

---

# Movie Recommendation System

## Overview

This project implements a **Movie Recommendation System** using three different approaches:

1. **Content-Based Filtering**
2. **Collaborative Filtering**
3. **Hybrid Model** (Combination of both Content-Based and Collaborative Filtering)

The system provides movie recommendations based on the **MovieLens 32M** dataset, focusing on user preferences and movie features such as genres. The models are evaluated using **precision**, **recall**, **F1-score**, **diversity**, and **novelty** metrics to assess their effectiveness in providing relevant, diverse, and novel recommendations.

---

## Key Features

* **Content-Based Filtering**: Recommends movies based on the similarity of genres using **TF-IDF** and **Cosine Similarity**.
* **Collaborative Filtering**: Recommends movies based on user-item interactions (ratings) using **Cosine Similarity** between users.
* **Hybrid Model**: Combines both Content-Based and Collaborative Filtering to generate recommendations by blending the strengths of both models.
* **Evaluation**: Performance is evaluated using precision, recall, F1-score, diversity, and novelty. Plots are generated for visual comparison of model performance.

---

## Dataset

This project uses the **MovieLens 32M** dataset, which contains:

* **Ratings**: Over 32 million ratings provided by users for movies.
* **Movies**: Information about movies, including **movieId**, **title**, and **genres**.

The data is processed and sampled for performance, with the option to switch to mock data if the dataset is unavailable.

---

## Requirements

* **Python 3.10**
* **Libraries**:

  * `pandas`: For data manipulation.
  * `numpy`: For numerical operations.
  * `scikit-learn`: For machine learning algorithms and metrics.
  * `scipy`: For sparse matrix operations.
  * `matplotlib`: For data visualization.
  * `seaborn`: For enhanced visualization and plotting.

To install the required libraries, use the following command:

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

---

## Data Loading & Sampling

The dataset is loaded from CSV files and preprocessed for training and testing. If the data is not available or sparse, mock data is generated for demonstration.

```python
# Example function to load and sample data
ratings_sample, movies_sample = load_and_sample('/content/ml-32m/ml-32m/')
```

---

## Model Implementation

### Content-Based Filtering

* **TF-IDF Vectorization** is applied to the movie genres to create a matrix of genre similarities.
* **Cosine Similarity** is used to compute the similarity between movies based on their genres.

### Collaborative Filtering

* A **user-item matrix** is created, and **Cosine Similarity** is calculated between movies based on user ratings.

### Hybrid Model

* The **Hybrid Model** combines the top recommendations from both Content-Based and Collaborative Filtering models, giving the user the best of both worlds.

---

## Functions

### `get_recommendations_logic(user_id, model_type, k)`

This function generates movie recommendations for a given user and model type:

* `model_type` can be `'Content-Based'`, `'Collaborative Filtering'`, or `'Hybrid'`.
* `k` specifies the number of recommendations to generate.

### `evaluate_and_plot()`

This function evaluates the models using multiple metrics:

* **Precision**
* **Recall**
* **F1-Score**
* **Diversity**
* **Novelty**

It also generates the following plots:

1. **Bar chart** comparing model performance (Precision, Recall, F1-score, Diversity, Novelty).
2. **Line chart** comparing precision and recall at different values of `K`.
3. **Heatmap** of the collaborative similarity matrix.

---

## Usage

1. **Load Data**: Load and preprocess the data by running the data loading functions.
2. **Generate Recommendations**: Use the `get_recommendations_logic()` function to get movie recommendations for a given user.
3. **Evaluate Models**: Use the `evaluate_and_plot()` function to evaluate and visualize the performance of the recommendation models.

```python
# Example: Get recommendations for user ID 1 using the Hybrid model
user_id = 1
recommended_movies = get_recommendations_logic(user_id, 'Hybrid', 10)
print(f"Recommended Movies: {recommended_movies}")

# Example: Evaluate model performance
metrics = evaluate_and_plot()
print(metrics)
```

---

## Evaluation Metrics

The models are evaluated using the following metrics:

* **Precision**: Measures the proportion of recommended movies that are relevant to the user.
* **Recall**: Measures the proportion of relevant movies that are recommended.
* **F1-Score**: The harmonic mean of precision and recall.
* **Diversity**: Measures the dissimilarity of recommended movies, ensuring the recommendations are not too similar.
* **Novelty**: Measures how surprising the recommendations are, ensuring that users are introduced to less popular items.

---

## Visualizations

The evaluation generates the following visual outputs:

* **Bar Charts** comparing the precision, recall, F1-score, diversity, and novelty for models.
* **Line Charts** comparing the precision and recall at different values of `K`.
* **Heatmap** visualizing the similarity matrix for collaborative filtering.

---

## Example Outputs

* **Recommendation Example**: For a given user, the model will recommend 10 movies based on their ratings and genre preferences.
* **Performance Metrics**: The system will display metrics for each model, such as precision, recall, and F1-score.

Example of printed output for recommendations:

```text
Recommended Movies: ['Movie 12', 'Movie 45', 'Movie 78', ...]
```

---

## Future Work

* **Deep Learning Models**: Investigating Neural Collaborative Filtering (NCF) and other deep learning models to improve accuracy.
* **Temporal Modeling**: Incorporating time-awareness into the recommendation models to account for evolving user preferences.
* **Explainable AI (XAI)**: Providing transparent explanations for the recommendations made by the system.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Conclusion

This project provides a robust and comprehensive approach to movie recommendations using both traditional machine learning models and hybrid techniques. The models are evaluated using multiple performance metrics and visualized for further analysis.
