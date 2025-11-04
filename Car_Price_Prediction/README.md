# Car Price Prediction with Machine Learning

### Author  
**Rupali Pragati**



### Overview  
The project predicts car prices based on technical and brand features using **machine learning regression models**.  
It demonstrates how various specifications like engine power, fuel efficiency, and brand affect the market price of cars.



### Objectives  
- Analyze and visualize car data.  
- Build regression models to predict prices.  
- Compare model accuracy and select the best-performing algorithm.  



### Dataset  
Dataset Source: [Car Price Prediction Dataset on Kaggle](https://www.kaggle.com/)  
Automatically downloaded via KaggleHub in the script.  



### Project Workflow  
1. **Data Collection:** Import dataset from Kaggle.  
2. **Preprocessing:** Handle missing values, encode categorical columns, and normalize data.  
3. **EDA:** Understand relationships between car features and price.  
4. **Model Building:** Train Linear Regression and Random Forest Regressor models.  
5. **Evaluation:** Compare models using MAE, RMSE, and R² metrics.  
6. **Visualization:** Display feature importance and prediction comparison.  



### Results  
- Random Forest achieved higher prediction accuracy compared to Linear Regression.  
- Engine size, brand, and horsepower are the top influencing factors.  



### Technologies Used  
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- KaggleHub  


### Run the Project  
```bash
# Install dependencies
pip install -r requirements.txt

# Run the project
python car_price_prediction.py
