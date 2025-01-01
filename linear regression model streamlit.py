import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title and description
st.title("Crime Rate Analysis and Prediction")
st.write("This application allows you to analyze and predict crime rates based on provided data.")

# File upload
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    # Read Excel file
    df = pd.read_excel(uploaded_file, sheet_name="data1")
    st.write("Dataset Preview:")
    st.dataframe(df)

    # Convert to CSV and read back for processing
    df.to_csv("final_crime_rate.csv", index=False)
    df = pd.read_csv("final_crime_rate.csv")

    # Visualization
    st.subheader("Visualization")
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    sns.lineplot(x="Year", y="Average Crime Rate", hue="Province", markers=True, alpha=0.7, data=df)
    plt.title("Average Crime Rate over Years by Province")
    plt.xlabel("Year")
    plt.ylabel("Average Crime Rate")
    plt.legend(title="Provinces", loc="upper left", bbox_to_anchor=(1, 1))
    st.pyplot(plt)

    # FacetGrid visualization
    g = sns.FacetGrid(df, col="Province", height=4, aspect=1, col_wrap=2)
    g.map(sns.lineplot, "Year", "Average Crime Rate", hue="Province", marker="o", alpha=0.7, data=df)
    st.pyplot(g.fig)

    # Data preparation
    st.subheader("Data Preparation and Encoding")
    df_encoded = pd.get_dummies(df, columns=["Province"], drop_first=True).astype(int)
    st.write("Encoded Data Preview:")
    st.dataframe(df_encoded)

    # Splitting data
    x = df_encoded.drop("Average Crime Rate", axis=1)
    y = df_encoded["Average Crime Rate"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Training Linear Regression model
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Model evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Evaluation")
    st.write(f"Mean Absolute Error: {mae}")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R² Score: {r2}")

    # Prediction plot
    st.subheader("Prediction vs Actual Values")
    plt.figure(figsize=(12, 8))
    plt.scatter(x_test["Year"], y_test, color="blue", alpha=0.6, label="Actual Values")
    x_test_sorted = x_test.sort_values("Year")
    y_pred_sorted = model.predict(x_test_sorted)
    plt.plot(x_test_sorted["Year"], y_pred_sorted, color="red", linewidth=2, label="Regression Line")
    plt.title("Linear Regression: Actual vs Predicted Crime Rate")
    plt.xlabel("Year")
    plt.ylabel("Average Crime Rate")
    plt.legend(title="Legend")
    st.pyplot(plt)

    # Prediction function
    st.subheader("Crime Rate Prediction")
    input_year = st.number_input("Enter the year:", min_value=int(df["Year"].min()), max_value=int(df["Year"].max()), step=1)
    input_province = st.selectbox("Select the province:", [col.replace("Province_", "") for col in x.columns if "Province" in col])

    if st.button("Predict"):
        # Prepare input for prediction
        feature_columns = x.columns.tolist()
        input_data = {col: 0 for col in feature_columns}
        input_data["Year"] = input_year
        input_data[f"Province_{input_province}"] = 1
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.write(f"Predicted Average Crime Rate for {input_province} in {input_year}: {prediction}")
