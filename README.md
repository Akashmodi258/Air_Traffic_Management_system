    
# ✈️Air Management System

A smart Air Traffic Management System that dynamically distributes aircraft landing and takeoff among multiple airports in the same city using an AI/ML model. We utilize LSTM (Long Short-Term Memory) networks to optimize aircraft allocation, reducing delays and improving efficiency.

## Overview
The system aims to optimize air traffic flow by ensuring that aircraft are dynamically routed to different airports in a city based on traffic conditions and predicted congestion. This improves flight scheduling, reduces waiting times, and enhances overall airport efficiency.    
## Features

- AI-Powered Optimization -  Uses an LSTM-based model to predict and allocate aircraft to the optimal airport.
- Dynamic Traffic Distribution -  Balances air traffic across multiple airports in a city.
- Efficient Takeoff & Landing -  Reduces congestion and enhances operational efficiency.
- Real-Time Traffic Management -  Continuously updates aircraft allocation based on real-time data.
- User-Friendly Interface -  Intuitive frontend built with HTML, CSS, JavaScript, and Bootstrap.
- Secure Backend -  Flask-powered backend with SQLAlchemy (SQLite3) for data management.


## 🛠️Tech Stack

**Frontend -** HTML, CSS, JavaScript, Bootstrap

**Backend -** Django

**Database -** SQLite3

**Machine Learning Model -** LSTM (Long Short-Term Memory)


## Workflow
1. **Data Collection -** Collect real-time and historical air traffic data.

2. **Traffic Prediction -** se an LSTM model to analyze trends and predict congestion.

3. **Airport Selection -** Dynamically allocate aircraft to the least congested airport.

4. **Traffic Update -** Continuously update the system with new data for real-time optimization.

5. **User Interface -** Display traffic status, recommended airport allocations, and flight details.
## Deployment

To deploy this project run
1.  Clone the repository

```bash
    git clone https - //github.com/Akashmodi258/Air_Traffic_Management_system.git
    cd air-traffic-management 
```
2. Create a virtual environment
```bash
    python -m venv venv
    venv\Scripts\Activate  
```

2.  Install dependencies
```bash
    pip install -r requirements.txt  
```
3. Apply migrations
```bash
     python manage.py migrate
```
4.  Run the server
```bash
    python mannage.py runserver
```



## 🚀 About Me
A passionate and results-driven AI/ML Developer with expertise in designing, developing, and deploying machine learning models and AI-driven solutions. Skilled in MLOps practices for scalable and efficient model deployment and lifecycle management. Proficient in Python and popular frameworks like TensorFlow, PyTorch, and Scikit-learn. Experienced in data preprocessing, model evaluation, and visualization tools such as Pandas, Seaborn, and Matplotlib. Dedicated to leveraging AI/ML to solve real-world challenges and deliver impactful results through innovative and robust solutions.


## Author

- [@Akash Modi](https://www.github.com/Akashmodi258)

## License

[MIT](https://choosealicense.com/licenses/mit/)


## Contributers
- Sunil
- Ishitha
    
