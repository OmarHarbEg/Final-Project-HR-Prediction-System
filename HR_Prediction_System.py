
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np

# Load the datasets
eda_dataset = pd.read_csv('7-Ready_Emp_Dataset.csv')
# modeling_dataset = pd.read_csv('10-Ready_Data_Processed.csv')

# Define pages
pages = {
    "Home Page": "home",
    "Statistics": "statistics",
    "Prediction": "prediction",
    # Add other page names here if needed
}

# Sidebar for page navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))

# Page 1: Home Page
if selected_page == "Home Page":

    # Define image URLs or upload paths
    top_right_image_path = os.path.join('epsilon-AI-logo-White2.jpg')
    bottom_left_image_path = os.path.join('pic1.jpg')
    bottom_image_path = os.path.join('machine-learning-process-flow.jpg')

    # Display the top-right image (local image)
    st.image(top_right_image_path, width=300, caption="Final Project AUG 2024")
    
    st.title("Home Page")
    st.write("Welcome to the Home Page of the Employee Attrition & Promotion Prediction App.")
    st.write("Use the sidebar to navigate through different sections.")
    
    # Add a custom CSS to position the top-right image and style the page
    st.markdown("""
    <style>
    .top-right {
        position: absolute;
        top: 10px;
        right: 50px;
        width: 300px;
        height: auto;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #fc8a15;
        color: #141010;
        text-align: center;
        padding: 0px;  /* Reduced padding to shorten the footer */
        margin-bottom: 0;
        font-size: 14px;
        font-weight: bold;  /* Makes text bold */
        font-style: italic;  /* Makes text italic */
    }
    .content {
        padding-top: 10px;
    }
    .half-half {
        display: flex;
        justify-content: space-between;
    }
    .half {
        width: 48%;
    }
    </style>
    """, unsafe_allow_html=True)

    # Content for the middle section (Two vertical halves)
    st.markdown("<div class='content'>", unsafe_allow_html=True)
    
    # Create two columns for text and image in the bottom half
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <h3>About the System</h3>
            <p>
            This system allows you to predict whether an employee is likely to leave the company or be promoted based on various input factors.
            </p>
            <ul>
                <li>Enter employee details like department, position, salary, etc.</li>
                <li>Get the probability of attrition or promotion based on machine learning models.</li>
            </ul>
            <p>
            Our predictive system uses state-of-the-art algorithms to help organizations better manage their workforce and make informed decisions.
            </p>
            <p>
            With an intuitive interface, managers can easily input data and receive real-time predictions on employee behavior. 
            It is designed to integrate seamlessly into existing HR workflows, providing key insights without disrupting daily operations.
            <br>
            <br>
            The system's ML models are continuously updated to reflect the latest workforce trends, ensuring that predictions 
            remain relevant and adaptive to changing business environments.
            </p>     
        """, unsafe_allow_html=True)
        # Display the image in the right half (local image)
        st.image(bottom_image_path, width=300, caption="Machine Learing Process", use_column_width=True)
    
    with col2:
        # Display the image in the right half (local image)
        st.image(bottom_left_image_path, width=300, caption="Predictive System Overview", use_column_width=True)

        st.markdown("""
            <h5>Objective of the project</h5>
            <p>
            The main goal of this project is to derive insights from the employee dataset that can help the company improve its workforce management.
            </p>
            <p>
            Attrition Prediction: What factors are driving employee attrition? Can we predict whether an employee will leave the organization based on 
            features like salary, years of experience, or KPIs?
            <br>
            <br>
            Promotion Factors: What are the key factors contributing to employee promotion? How do training, education, 
            and performance metrics affect promotion rates?
            </p>
            Prepared by:
            OMAR HARB
            <br>
            Business Development Manager
            <br>
            Egypt, Cairo, Heliopolis
            <br>
            All rights reserved © Omar ElFarouk Ahmed
            </p>
        """, unsafe_allow_html=True)
        
    # Close content section
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer section
    st.markdown("""
        <div class="footer">
            <p><strong><em>© 2024 Employee Prediction System | Powered by OMAR HARB AI SOLUTIONS</em></strong></p>
        </div>
        """, unsafe_allow_html=True)
    

# Page 2: Statistics
elif selected_page == "Statistics":
    st.title("Statistics")
    st.write("Explore various distributions and statistical insights in this section.")
    st.write(eda_dataset.head())
    
    # Age Distribution
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(eda_dataset['Age'], kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    st.pyplot(plt)
    
    # Salary Distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Salary', data=eda_dataset)
    plt.title('Salary Distribution')
    plt.xlabel('Salary')
    st.pyplot(plt)
    
    # Correlation Matrix for Numeric Features
    numeric_columns = eda_dataset.select_dtypes(include=[np.number])
    correlation_matrix = numeric_columns.corr()
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix for Numeric Features')
    st.pyplot(plt)
    
    # Salary Distribution Across Departments
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Department', y='Salary', data=eda_dataset)
    plt.title('Salary Distribution Across Departments')
    plt.xlabel('Department')
    plt.ylabel('Salary')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    
    # Salary Distribution Across Regions
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Region', y='Salary', data=eda_dataset)
    plt.title('Salary Distribution Across Regions')
    plt.xlabel('Region')
    plt.ylabel('Salary')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    
    # Salary and Gender Distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Gender', y='Salary', data=eda_dataset)
    plt.title('Distribution between Salary and Gender')
    st.pyplot(plt)
    
    # Salary Distribution Across Positions
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Position', y='Salary', data=eda_dataset)
    plt.xticks(rotation=45)
    plt.title('Distribution between Salary and Position')
    st.pyplot(plt)

    # Distribution between Gender and Marital Status
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Marital_Status', hue='Gender', data=eda_dataset)
    plt.title('Distribution between Gender and Marital Status')
    st.pyplot(plt)
    
    # Average Salary by Position Group
    def categorize_position(position):
        group_1 = ['Director', 'Regional', 'Head', 'Manager', 'Executive']
        group_2 = ['Associate', 'Assistant', 'Engineer']
        group_3 = ['Developer', 'Accountant', 'Lawyer']
        group_4 = ['Representative', 'Paralegal', 'Analyst']
        group_5 = ['Intern', 'Specialist', 'Junior']
        
        if any(keyword in position for keyword in group_1):
            return 'Group 1 (Director, Regional, Head, Manager, Executive)'
        elif any(keyword in position for keyword in group_2):
            return 'Group 2 (Associate, Assistant, Engineer)'
        elif any(keyword in position for keyword in group_3):
            return 'Group 3 (Developer, Accountant, Lawyer)'
        elif any(keyword in position for keyword in group_4):
            return 'Group 4 (Representative, Paralegal, Analyst)'
        elif any(keyword in position for keyword in group_5):
            return 'Group 5 (Intern, Specialist, Junior)'
        else:
            return 'Other'
    
    eda_dataset['Position_Group'] = eda_dataset['Position'].apply(categorize_position)
    filtered_eda_dataset = eda_dataset[eda_dataset['Position_Group'] != 'Other']
    
    avg_salary = filtered_eda_dataset.groupby('Position_Group')['Salary'].mean().reset_index()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Position_Group', y='Salary', data=avg_salary, palette='Set2')
    plt.title('Average Salary by Position Group')
    plt.xlabel('Position Group')
    plt.ylabel('Average Salary')
    plt.xticks(rotation=30, ha='right')
    st.pyplot(plt)


# Page 3: Prediction
elif selected_page == "Prediction":
    # Load the saved models for Attrition and Promotion
    attrition_model = joblib.load('best_attrition_model.pkl')
    promotion_model = joblib.load('best_promotion_model.pkl')
    
    # Title & Sub-header of the page
    st.title("Employee Attrition & Promotion Prediction")
    st.write("Make a prediction for employee attrition or promotion.")
    st.subheader("Enter Employee Information to Predict Attrition and Promotion")

    # Input fields for user data
    # Department
    department = st.selectbox("Select Department", [
    'Finance', 'Analytics', 'Operations', 'IT', 'Human Resources', 'Legal', 'Marketing',
    'Sales', 'Procurement', 'Project Control', 'Quality Control/Quality Assurance'])

    # Position
    position = st.selectbox("Select Position", [
    'Account Director', 'Account Associate', 'Accountant', 'Analytics Director', 'Analytics Manager',
    'Construction Engineer', 'Data Analyst', 'Developer', 'ERP Head', 'Finance Director', 'Finance Manager',
    'HR Assistant', 'HR Director', 'HR Manager', 'HR Specialist', 'HR Executive', 'IT Director', 'IT Manager',
    'Junior Accountant', 'Junior Analyst', 'Junior Developer', 'Lawyer', 'Legal Manager', 'Legal Assistant',
    'Legal Director', 'Marketing Assistant', 'Marketing Director', 'Marketing Manager', 'Marketing Specialist',
    'MEP Engineer', 'National Account Head', 'National Marketing Manager', 'National Sales Manager', 
    'Operations Director', 'Paralegal', 'Procurement Assistant', 'Procurement Director', 'Procurement Manager',
    'Procurement Specialist', 'Project Control Assistant', 'Project Control Director', 'Project Control Manager',
    'Project Control Specialist', 'Project Manager', 'QA Director', 'QA Lead', 'QA Manager', 'QA Specialist',
    'QA Engineer II', 'QC Inspector', 'QC Manager', 'Recruitment Manager', 'Regional Account Head',
    'Regional Marketing Manager', 'Regional Sales Manager', 'Sales Director', 'Sales Executive', 'Sales Representative',
    'Senior Account Executive', 'Senior Accountant', 'Senior Developer', 'Senior Executive', 'Senior HR', 'Senior Analyst',
    'Senior Marketing Executive', 'Site Engineer', 'Software Engineer III', 'Technical Office Engineer', 'Technical Lead',
    'Account Executive', 'Account Representative', 'Accounts Intern', 'Business Development Representative', 
    'HR Associate', 'HR Representative', 'HR Intern', 'Marketing Associate', 'Marketing Representative', 
    'Marketing Intern', 'Marketing Development Representative', 'Sales Development Representative',
    'Software Engineer II', 'Software Engineer I', 'QA Engineer I'])

    # Education
    education = st.selectbox("Select Education Level", ['Bachelor', 'Masters & above'])
    
    # Region
    region = st.slider('Region', 1, 34, 1)
    
    # Recruitment Channel
    recruitment_channel = st.selectbox("Recruitment Channel", ['Sourcing', 'Referred', 'Other'])
    
    # Years of Trainings
    years_of_trainings = st.number_input('Years of Trainings', 0, 50, 1)

    # Years of Experience
    years_of_experience = st.slider('Years of Experience', 0, 50, 1)
    
    # Previous Year Rating
    previous_year_rating = st.number_input('Previous Year Rating', 1, 5, 3)
    
    # KPIs met >80%
    kpis_met = st.selectbox("KPIs Met >80%", [0, 1])
    
    # Awards Won
    awards_won = st.selectbox("Awards Won", [0, 1])
    
    # Average Training Score
    avg_training_score = st.number_input("Avg Training Score", 30, 100, 70)
    
    # Gender
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    
    # Marital Status
    marital_status = st.selectbox("Marital Status", ['Married', 'Divorced', 'Widowed', 'Single'])
    
    # Dependents
    dependents = st.number_input('Dependents', 0, 10, 0)

    # Age
    age = st.slider('Age', 21, 50, 65)
    
    # Basic Salary
    salary = st.number_input('Salary', 3000, 1000000, 50000)

    # HRA
    #HRA = st.number_input('House Rent Allowance', 3000, 1000000, 50000)

    # DA
    #DA = st.number_input('Dearness Allowance', 3000, 1000000, 50000)

    # PF
    #PF = st.number_input('Provident Fund', 3000, 1000000, 50000)
    
    # Gross Salary
    #gross_salary = st.number_input('Gross Salary', 3000, 1000000, 50000)
    
    # Insurance
    insurance = st.selectbox("Insurance Type", ['Medical', 'Life', 'Both'])
    
    # Over Time
    over_time = st.selectbox("Over Time", ['Yes', 'No'])
    
    # Business Travel
    business_travel = st.selectbox("Business Travel Frequency", ['Frequently', 'Non-Travel', 'Rarely'])

    # =====================================
    # Calculating HRA, DA, PF and Gross Salary
    # =====================================
    hra = salary * 0.25  # 25% of Salary
    da = salary * 0.2    # 20% of Salary
    pf = salary * 0.15   # 15% of Salary
    gross_salary = salary + hra + da + pf  # Total Gross Salary
    
    # Display the calculated values with size and color using markdown
    st.markdown(f"<p style='font-size:20px; color:blue;'>Calculated HRA: <strong>{hra:.2f}</strong></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:20px; color:green;'>Calculated DA: <strong>{da:.2f}</strong></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:20px; color:orange;'>Calculated PF: <strong>{pf:.2f}</strong></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:22px; color:white;'>Calculated Gross Salary: <strong>{gross_salary:.2f}</strong></p>", unsafe_allow_html=True)
        
    # Display the calculated values
    #st.write(f"### Calculated HRA: **{hra}**")
    #st.write(f"### Calculated DA: **{da}**")
    #st.write(f"### Calculated PF: **{pf}**")
    #st.write(f"### Calculated Gross Salary: **{gross_salary}**")

    # Create a dictionary of the user input
    user_data = {
    'Department': department,
    'Position': position,
    'Education': education,
    'Region': region,
    'Recruitment_channel': recruitment_channel,
    'Years_of_trainings': years_of_trainings,
    'Years_of_Experience': years_of_experience,
    'Previous_Year_Rating': previous_year_rating,
    'KPIs_met >80%': kpis_met,
    'Awards_Won': awards_won,
    'Avg_Training_Score': avg_training_score,
    'Gender': gender,
    'Marital_Status': marital_status,
    'Dependents': dependents,
    'Age': age,
    'Salary': salary,
    'HRA': hra, # Add calculated HRA
    'DA': da, # Add calculated HRA
    'PF': pf, # Add calculated HRA
    'Gross_Salary': gross_salary, # Add calculated HRA
    'Insurance': insurance,
    'Over_Time': over_time,
    'Business_Travel': business_travel
}
    
    # Convert the user input into a pandas DataFrame
    input_df = pd.DataFrame([user_data])
    
    # Display user input data
    st.write("### User Input")
    st.write(input_df)
    
    # Prediction button
    if st.button('Predict'):
        # Preprocess the input data and predict Attrition
        attrition_prob = attrition_model.predict_proba(input_df)[:, 1]  # Probability for class 1 (Attrition = Yes)
        promotion_prob = promotion_model.predict_proba(input_df)[:, 1]  # Probability for class 1 (Promotion = Yes)
    
        # Display results
        st.write(f"The **Attrition** probability is: **{attrition_prob[0] * 100:.2f}%**")
        st.write(f"The **Promotion** probability is: **{promotion_prob[0] * 100:.2f}%**")
