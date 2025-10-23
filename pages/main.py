import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import plotly.express as px
from streamlit_option_menu import option_menu
import data_analysis_functions as function
import data_preprocessing_function as preprocessing_function
import home_page
import base64
import streamlit_authenticator as stauth


import streamlit as st

if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    st.switch_page("loginsystem.py")  # Redirect back if not authenticated
# # page config sets the text and icon that we see on the tab
st.set_page_config(page_icon="✨", page_title="AutoEDA")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

#uncomment the above lines of code when we want to remove made with streamlit logo and also the top right three-dots icon



# Create a Streamlit sidebar
# st.sidebar.title("AutoEDA: Automated Exploratory Data Analysis and Processing")

# Set custom CSS
custom_css = home_page.custom_css


# Create the introduction section
st.title("Welcome to AutoEDA")
# st.write('<div class="tagline">Unleash the Power of Data with AutoEDA!</div>', unsafe_allow_html=True)


selected = option_menu(
    menu_title=None,
    options=['Home', 'Data Exploration', 'Data Preprocessing','Machine Learning'],
    icons=['house-heart', 'bar-chart-fill', 'hammer'],
    orientation='horizontal'
)

if selected == 'Home':
    home_page.show_home_page()


# Create a button in the sidebar to upload CSV
uploaded_file = st.sidebar.file_uploader("Upload Your CSV File Here", type=["csv","xls","xlsx","txt"])
# use_example_data = st.sidebar.checkbox("Use Example Titanic Dataset", value=False)

# ADDING LINKS TO MY PROFILES 
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")

st.sidebar.markdown("### **Connect with Me**")
# Create columns in the sidebar for LinkedIn and GitHub icons
col1, col2 = st.sidebar.columns(2)

# Define the width and height for the icons (adjust as needed)
icon_width = 80
icon_height = 80

# Add the LinkedIn icon and link
col1.markdown(f'<a href="https://www.linkedin.com/in/umarkhattab/"><img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width={icon_width} height={icon_height}></a>', unsafe_allow_html=True)

# Add the GitHub icon and link
col2.markdown(f'<a href="https://github.com/UmarKhattab09"><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAMAAAAt85rTAAAAkFBMVEX///8XFRUAAAAUEhIJBQUQDQ0MCQkEAADt7e319fXr6+vw8PAVEhLo6Oi/v7/6+vpcW1shHx9rampXVla5uLh4d3dBQECfnp7j4+PLy8tPTk7HxsbX19cuLS1IR0eYl5eOjY04Njaop6coJiaFhIRubW0yMTGvrq58e3tjYmKSkpKJiYnc29tDQkIkIiKkpKQrDOYYAAAKMUlEQVR4nO2d53arOhBGDyMDJrjhGre4xiVOef+3u8Y+jjGgGYEKPndp/8paSRAfqEyT+PPHYrFYLBaLxWKxWCwWi8VisVgsFovF8u/Rbs7G09aF6XjWbFd9P8oIXsY/p84GMkS97c+0GVR9f1IE02VnHoupMScDq8W/2fRO03rV91mKcHqKtdXcrLSMzGg7Dau+32IE+yMDILQlVJ4fRGf9z2gM+z0AP6dTUm/ybf8vaJwdHPCKibvhgTcYV33/OO3+CqDgu3t4jwDz/ROvIKO5+Ljj4cL34jklthcFphXiNS6fcHkcRdBQIe8q0VlUrSdFK5IZenkSvX3VmhK8dtR0zkeJk8+qdf2l/QE11fJiarB9iqE424EOeTHw/QTL4kl977zjwqBiec03ba/vCgxfq9TXL2uViePBujp9B7VrA4fKummou3v+Klw1q9DXjAzpOy8YXgVL4lTn7JnGhb5pfX0jw+8XBoYttx+z+s6AUfN7bWz4VaNwUYG+s8KTKX37SvSdFY7+3/pMKZxWpu+s0MBqMTM+fyZgoN1/CjyD63uewhe9+torLc67ON5Gr5f/XuEAvAIdnfp+Ktd3VvihTx9ngmFxos9XOjbdS/IwfzrTONG4uSKY97H+OR2HoCq45gGsjtvRerHhPE9dw3CQ30HhcP11c98B+RCGC/C2fr1mJ0acBrt69PU5AxBmv3/SXn/JvUYPVot7krDJyQboCdO0Oa2584dk0LhXXqIHw9ZDm5P8DsFAR6a0y3uB6ZjQrGQgn8FkmroUz23xe+r1jXkrRI59ON5lJsCG69X8WwWJX/Pc9B8wcHIuVKBRWRq8twJ5kdm183tr7KIrWk063cH2tFyetofucTKM4GEh8POXN55ABqqTpJwJLRaY+/dh99xPLzUiXvdjP3sJ0zcUvnz2F+/RdQVtwCQ/MrjhPlbF3u8L14dwh5x/GUewOY7G1JOerbs78Hk9jrMyxa9QbUif25AD77z/CWeCC3L7kxvY5QdH4FhcBZ9Xvg2q0zREoweJ5VcexInQG0Zo8RuuKXQrPhEnQm/yh7tOOEqNbt4aX63AmrLVvol5gXoFohEuUJWS2aKtaB2DPAP/2jR3Ai9G6GFxNFiqaSUfNEfAQE3eEE9E+JqcsysfaNuK0hUOGghlkZJGOHRQ14vlm4kFmRGBJiWN8CCizJD2r8pARQp1BpvR+fuMr8BeC/AeqneWofI8DV/etUcn6hh3p69+tUdFPxRkto8+1YbGUPOBerreRLoNsom5AiFcOlTz0jFSKhuo2vFMEXBd+ptA2eAM39NV1AAB5sjEyNoZ7RX+BHXE7x5Z4gqZZDqNeICqrEEM4hFLOvZEQYxeS/sK4tRfbkHOm+mii4QaW5BiiCZ0JN1e3BY08QKpV9iQesi4LcjAzNZGfBRK+fX4KqjXFbyDV45Bi74CF8LfVOGsCPCC38VW4tK4IWpkionZYX1UKj46xK6swhkTA+2j7qq8NxMyNNxkrN4fi446zC8/1SEpCUdhWJIkzNumfr+P8uY+OomyhrktxehQkZhGUUPN/VKogACd7OCn9HXRwa0yu0OBx9bLJ/DQiIHJHTdojZzEjaBhVzOG6BU08iVhbuNj21CJeAxqb7ur0tf9Rmfn8mO7MKhAieQBunH8ad6ghFdqIrUjBO4S6hKotbzikWoEyrgpBcFzFLoEGlwH+YVkGgWa8udj8OCoJoGe9pjvHTxHKSEQWwe5VXgaQE0qiWUCzX02InPu0hdmUjGn9HXRWIhjIGz/lzDCLA53V/rCnJrwm0BDQTUqPCuRBMUD9+aMUXydl5jO8dnZV1RJRUOEZ5elL4yvr4ypk4DTw4dKeaufqLAwNssQt1E+yUwUOZk6f4G6jfIpUDwpYCy0fSIElt/1GuTvb7vBIjOHSqGRE7ksPb4QymWuhCF6qFQtEFFEYiY0igZFJd02qhjOxDzapu5BplwNTes4Zs7PoE4+kSvnJAu5NO/bP79AsphL6vJEFY6BwAy1s10i7BtDFFJJJeeECMgbWEpdnzx6RN32k3yoYkBZpy30qFNRK9wYEsM8ybgCsdQ7ejPZdfJsF+mSX9xjinH1xWbaK/rxynYgqiD1jK8tvNajG5fvP9SugriRiZ66+yOtTyKidoNcKGKFbzoUCrw/FWlmgT56bmejfDlsCp2sq2KGw72xv3iqvfuWL3KuiZLoOj2PxjA4KvR+g67Y2V9KFmHSWLo15iirXVuLHl2q5uicTPgXIPc8IgYbJf20/y16dJuiDF7aKYRDfzbeD3JOOHJh9SM5nwajnfhhLap29r09jPdf63bayz5pF6A7Ld1vwtaxyBcPvDc1+lKBi0SkqeXmjBUfou6+RCjjdX10ix0mpO7U2Idt2My5r3nt3N1hlxNk3kfjF8FXGTSniyMU/k4Mc1XpS3nVrpcIJfPq9S4i55PulnDXptvuZMP5MhOBwuxWKi7CYHufSrCKRNcDsqD6C9Bt+sjFNwrtw3RgBHZ3Ewnbw8gcsp46+C55pJ7a9GSUespuYoZ2+HcoEk+gtrFzcNXu38+EDhJHmfLjNmLutoBXlINq4zcTukjsbeUGhsQSW69lvoajYHfyI9mO5P0GJAOOaeUJlq3j5znko/TAowvZ13Q7tJEb4RedBogtkLmXVl8oF2ZjF3dvM9+lEg0KU+mVLMzREOjKPueELZgX2GC+6KW/ih7aqScxmY2RJNr5yRrJ4hlukbjPQ7t6wunZTposopqtUpNhTbzmueDZ61o6aEy2kz7MZaOzSXk9dJJ55x8nLWFTquBary9znrE7Uy71dPDVuBxF+TboF3EKqWNjUvoO9BVLkgmlZ/fuhc1mU9RN+oUoV3nEk9gQSZJJhqh5mvUCzlJD79bvzDBUkgCtu+LGmu7SlbTjpOQ8oAIC9ddwps85VHEgkLhAXUdsJ0k7uArarGeONObpM1J5NE8rlM6e1QWjFqD1eKVfgvQ3J3xZ51NQYG1l6IuSYSOlkEFHKsYstkzUzG37rs/TLmoNVut00ZP47QgJrO0MftY9jDL1+C7A12A9nTXD8OVz3Bq9TxrCN1QXCPn6BvehnAmGOdaVC0lqNXGBdDoJhoa/6Nomv67IPIUC9VQB4FCnVjLxN8g/h/2mz9gWjSQf+G0VEohaMszkbtokLfQDKAWOQcHdJc9MaXgen2mj5vG5qxEI8yq/if3O76YN8WpgxKNn1Qy/O33g7lBTIVDaCpSn3uG9RHmBZxPQoPXCZQ2cyL1wop4jEKr8SnSSy+dPlAt0oWvUOEMZRzn9VEogg+gJvkWfYJ+tvpIRCJHhDwvTBIt0dVdpgQxgYdi0FqK9aDxILCmQQWNh3rIWI1z791qlAtt+Esa2B/76eeaWHFo9+Gs5FznfuHe1FhoAvcrsTmFeT37s7UKRmPfr9T/8k+aNQqoYH3q9ZaGOFi57vcNzrQsWi8VisVgsFovFYrFYLBaLxWKxWCyWLP8Bd0OMHy9KQeAAAAAASUVORK5CYII=" width={icon_width} height={icon_height}></a>', unsafe_allow_html=True)




if uploaded_file:
    df = function.load_data(uploaded_file)
    df.drop(df.columns[df.columns.str.contains('^Unnamed')], axis=1, inplace=True)

    current_file_name = uploaded_file.name

    # Check if a different file is uploaded
    if st.session_state.get("current_file_name") != current_file_name:
        # Update stored file name
        st.session_state.current_file_name = current_file_name
        # Reset new_df with the new file
        st.session_state.new_df = df.copy()
        st.success(f"✅ New file '{current_file_name}' uploaded and data refreshed.")
    else:
        # If same file, keep using the one in session state
        df = st.session_state.get("new_df", df)

    # Make sure df always refers to the current working copy
    st.session_state.new_df = st.session_state.get("new_df", df)

    # st.write("### Current DataFrame:")
    # st.dataframe(st.session_state.new_df.head())

    

# Create a checkbox in the sidebar to choose between the example dataset and uploaded dataset

elif use_example_data:
    # Load the example dataset
    df = function.load_data(file="example_dataset/titanic.csv")

    # Set st.session_state.new_df to the example dataset for data preprocessing
    if 'new_df' not in st.session_state:
        st.session_state.new_df = df
   


# TODO: Some issue related to session_state. When we upload a new dataset, it does not reflect changes in the data preprocessing tab as we are using session state.
# and the data is defined only once. need to solve this issue.
# Temporary solution is to reload the page and upload a new dataset


# Display the dataset preview or any other content here
if uploaded_file is None and selected!='Home' and not use_example_data:
    # st.subheader("Welcome to DataExplora!")
    st.markdown("#### Use the sidebar to upload a CSV file or use the provided example dataset and explore your data.")
    
else:
    
    if selected=='Data Exploration':

        tab1, tab2 = st.tabs(['📊 Dataset Overview :clipboard', "🔎 Data Exploration and Visualization"])
        num_columns, cat_columns, bool_columns = function.categorical_numerical(st.session_state.new_df)
        
        
        with tab1: # DATASET OVERVIEW TAB
            st.subheader("1. Dataset Preview")
            st.markdown("This section provides an overview of your dataset. You can select the number of rows to display and view the dataset's structure.")
            function.display_dataset_overview(st.session_state.new_df,cat_columns,num_columns,bool_columns)


            st.subheader("3. Missing Values")
            function.display_missing_values(st.session_state.new_df)
            
            st.subheader("4. Data Statistics and Visualization")
            function.display_statistics_visualization(st.session_state.new_df,cat_columns,num_columns)

            st.subheader("5. Data Types")
            function.display_data_types(st.session_state.new_df)

            st.subheader("Search for a specific column or datatype")
            function.search_column(st.session_state.new_df)

        with tab2: 

            function.display_individual_feature_distribution(st.session_state.new_df,num_columns)

            st.subheader("Scatter Plot")
            function.display_scatter_plot_of_two_numeric_features(st.session_state.new_df,num_columns)


            if len(cat_columns)!=0:
                st.subheader("Categorical Variable Analysis")
                function.categorical_variable_analysis(st.session_state.new_df,cat_columns)
            else:
                st.info("The dataset does not have any categorical columns")


            st.subheader("Feature Exploration of Numerical Variables")
            if len(num_columns)!=0:
                function.feature_exploration_numerical_variables(st.session_state.new_df,num_columns)

            else:
                st.warning("The dataset does not contain any numerical variables")

            # Create a bar graph to get relationship between categorical variable and numerical variable
            st.subheader("Categorical and Numerical Variable Analysis")
            if len(num_columns)!=0 and len(cat_columns)!=0:
                function.categorical_numerical_variable_analysis(st.session_state.new_df,cat_columns,num_columns)
                
            else:
                st.warning("The dataset does not have any numerical variables. Hence Cannot Perform Categorical and Numerical Variable Analysis")

    
       




    # DATA PREPROCESSING  
    if selected=='Data Preprocessing':
        # st.header("🛠️ Data Preprocessing")
        revert = st.button("Revert to Original Dataset",key="revert_button")

        if revert:
            st.session_state.new_df = df.copy()

        # REMOVING UNWANTED COLUMNS
        st.subheader("Remove Unwanted Columns")
        columns_to_remove = st.multiselect(label='Select Columns to Remove',options=st.session_state.new_df.columns)

        if st.button("Remove Selected Columns"):
            if columns_to_remove:
                st.session_state.new_df = preprocessing_function.remove_selected_columns(st.session_state.new_df,columns_to_remove)
                st.success("Selected Columns Removed Sucessfully")
                
        st.dataframe(st.session_state.new_df)

        # CONVERT DATA TYPES
        st.subheader("Convert Data Types of Columns")
        column_type_dict = {}
        columns_for_type_conversion = st.multiselect("Select Columns to Convert Data Types", options=
            st.session_state.new_df.columns)
        if columns_for_type_conversion:
            for column in columns_for_type_conversion:
                new_dtype = st.selectbox(f"Select new data type for column '{column}'", 
                    options=['int', 'float', 'str', 'category', 'bool'], key=f"dtype_{column}")
                column_type_dict[column] = new_dtype
            if st.button("Convert Data Types"):
                st.session_state.new_df = preprocessing_function.convert_data_types(st.session_state.new_df, column_type_dict)
                st.success("Data Types Converted Successfully")
            else:
                st.info("Please select columns and data types, then click 'Convert Data Types'")

        st.dataframe(st.session_state.new_df)

       

       # Handle missing values in the dataset
        st.subheader("Handle Missing Data")
        missing_count = st.session_state.new_df.isnull().sum()

        if missing_count.any():

            selected_missing_option = st.selectbox(
                "Select how to handle missing data:",
                ["Remove Rows in Selected Columns", "Fill Missing Data in Selected Columns (Numerical Only)"]
            )

            if selected_missing_option == "Remove Rows in Selected Columns":
                columns_to_remove_missing = st.multiselect("Select columns to remove rows with missing data", options=st.session_state.new_df.columns)
                if st.button("Remove Rows with Missing Data"):
                    st.session_state.new_df = preprocessing_function.remove_rows_with_missing_data(st.session_state.new_df, columns_to_remove_missing)
                    st.success("Rows with missing data removed successfully.")

            elif selected_missing_option == "Fill Missing Data in Selected Columns (Numerical Only)":
                numerical_columns_to_fill = st.multiselect("Select numerical columns to fill missing data", options=st.session_state.new_df.select_dtypes(include=['number']).columns)
                fill_method = st.selectbox("Select fill method:", ["mean", "median", "mode"])
                if st.button("Fill Missing Data"):
                    if numerical_columns_to_fill:
                        st.session_state.new_df = preprocessing_function.fill_missing_data(st.session_state.new_df, numerical_columns_to_fill, fill_method)
                        st.success(f"Missing data in numerical columns filled with {fill_method} successfully.")

                    else:
                        st.warning("Please select a column to fill in the missing data")

            function.display_missing_values(st.session_state.new_df)

        else:
            st.info("The dataset does not contain any missing values")

        encoding_tooltip = '''**One-Hot encoding** converts categories into binary values (0 or 1). It's like creating checkboxes for each category. This makes it possible for computers to work with categorical data.
        **Label encoding** assigns unique numbers to categories. It's like giving each category a name (e.g., Red, Green, Blue becomes 1, 2, 3). This helps computers understand and work with categories.
        '''
        st.subheader("Encode Categorical Data")

        new_df_categorical_columns = st.session_state.new_df.select_dtypes(include=['object']).columns

        if not new_df_categorical_columns.empty:
            select_categorical_columns = st.multiselect("Select Columns to perform encoding",new_df_categorical_columns)

            #choose the encoding method
            encoding_method = st.selectbox("Select Encoding Method:",['One Hot Encoding','Label Encoding'],help=encoding_tooltip)
    

            if st.button("Apply Encoding"):
                if encoding_method=="One Hot Encoding":
                    st.session_state.new_df = preprocessing_function.one_hot_encode(st.session_state.new_df,select_categorical_columns)
                    st.success("One-Hot Encoding Applied Sucessfully")

                if encoding_method=="Label Encoding":
                    st.session_state.new_df = preprocessing_function.label_encode(st.session_state.new_df,select_categorical_columns)
                    st.success("Label Encoding Applied Sucessfully")


            st.dataframe(st.session_state.new_df)
        else:
            st.info("The dataset does not contain any categorical columns")

        feature_scaling_tooltip='''**Standardization** scales your data to have a mean of 0 and a standard deviation of 1. It helps in comparing variables with different units. Think of it like making all values fit on the same measurement scale.
        **Min-Max scaling** transforms your data to fall between 0 and 1. It's like squeezing data into a specific range. This makes it easier to compare data points that vary widely.'''


        st.subheader("Feature Scaling")
        new_df_numerical_columns = st.session_state.new_df.select_dtypes(include=['number']).columns
        selected_columns = st.multiselect("Select Numerical Columns to Scale", new_df_numerical_columns)

        scaling_method = st.selectbox("Select Scaling Method:", ['Standardization', 'Min-Max Scaling'],help=feature_scaling_tooltip)

        if st.button("Apply Scaling"):
            if selected_columns:
                if scaling_method == "Standardization":
                    st.session_state.new_df = preprocessing_function.standard_scale(st.session_state.new_df, selected_columns)
                    st.success("Standardization Applied Successfully.")
                elif scaling_method == "Min-Max Scaling":
                    st.session_state.new_df = preprocessing_function.min_max_scale(st.session_state.new_df, selected_columns)
                    st.success("Min-Max Scaling Applied Successfully.")
            else:
                st.warning("Please select numerical columns to scale.")

        st.dataframe(st.session_state.new_df)

        st.subheader("Identify and Handle Outliers")

        
        # Select numeric column for handling outliers
        


        selected_numeric_column = st.selectbox("Select Numeric Column for Outlier Handling:", new_df_numerical_columns)
        st.write(selected_numeric_column)

        
        # Display outliers in a box plot
        fig, ax = plt.subplots()
        ax = sns.boxplot(data=st.session_state.new_df, x=selected_numeric_column)
        st.pyplot(fig)


        outliers = preprocessing_function.detect_outliers_zscore(st.session_state.new_df, selected_numeric_column)
        if outliers:
            st.warning("Detected Outliers:")
            st.write(outliers)
        else:
            st.info("No outliers detected using IQR.")


        # Choose handling method
        outlier_handling_method = st.selectbox("Select Outlier Handling Method:", ["Remove Outliers", "Transform Outliers"])

        # Perform outlier handling based on the method chosen
        if st.button("Apply Outlier Handling"):
            if outlier_handling_method == "Remove Outliers":
               
                st.session_state.new_df = preprocessing_function.remove_outliers(st.session_state.new_df, selected_numeric_column,outliers)
                st.success("Outliers removed successfully.")

            elif outlier_handling_method == "Transform Outliers":
                # Provide options for transforming outliers (e.g., capping, log transformation)
                # Update st.session_state.new_df after transforming outliers
                st.session_state.new_df = preprocessing_function.transform_outliers(st.session_state.new_df, selected_numeric_column,outliers)
                st.success("Outliers transformed successfully.")

        # Show the updated dataset
        st.dataframe(st.session_state.new_df)
        
        if st.session_state.new_df is not None:
            # Convert the DataFrame to CSV
            csv = st.session_state.new_df.to_csv(index=False)
            # Encode as base64
            b64 = base64.b64encode(csv.encode()).decode()
            # Create a download link
            href = f'data:file/csv;base64,{b64}'
            # Display a download button
            st.markdown(f'<a href="{href}" download="preprocessed_data.csv"><button>Download Preprocessed Data</button></a>', unsafe_allow_html=True)
        else:
            st.warning("No preprocessed data available to download.")
if selected == 'Machine Learning':
    num_columns, cat_columns, bool_columns = function.categorical_numerical(st.session_state.new_df)

    checked = st.checkbox("Drop Non-Numeric Columns Before Training Model")
    if checked:
        st.session_state.new_df = st.session_state.new_df.drop(columns=cat_columns)
        st.write("Non-numeric columns dropped:")
        st.write(cat_columns)
        st.success("Non-Numeric Columns Dropped Successfully")
    else:
        st.info("Proceeding without dropping non-numeric columns may lead to errors during model training.")

    if bool_columns:
        st.write("Note: The dataset contains boolean columns. They will be treated as numeric (0 and 1) during model training.")
        boolfix = st.checkbox("Convert Boolean Columns to Numeric (0 and 1)")
        if boolfix:
            for col in bool_columns:
                st.session_state.new_df[col] = st.session_state.new_df[col].astype(int)
            st.success("Boolean Columns Converted to Numeric Successfully")

    model = st.selectbox(
        "Select Model Type", 
        ["regression", "classification", "neuralnetwork for regression", "neuralnetwork for classification"]
    )

    # Show hyperparameters ONLY for neural network models
    show_nn = model in ["neuralnetwork for regression", "neuralnetwork for classification"]
    if show_nn:
        st.write("### Neural Network Hyperparameters (Adjust Before Training)")
        n_hidden_layers = st.number_input(
            "Number of Hidden Layers", min_value=1, max_value=5, value=2
        )
        neurons_per_layer = st.number_input(
            "Neurons per Hidden Layer", min_value=1, max_value=512, value=64
        )
        learning_rate = st.number_input(
            "Learning Rate", min_value=0.0001, max_value=1.0, value=0.001, step=0.0001, format="%.4f"
        )
        epochs = st.number_input(
            "Epochs", min_value=10, max_value=500, value=50, step=10
        )
        batch_size = st.number_input(
            "Batch Size", min_value=1, max_value=256, value=32
        )
    else:
        # Default values; not shown to user
        n_hidden_layers = None
        neurons_per_layer = None
        learning_rate = None
        epochs = None
        batch_size = None

    target_column = st.selectbox(
        label="Enter the Target Column Name", 
        options=st.session_state.new_df.columns
    )

    if st.button("Train Model"):
        if target_column in st.session_state.new_df.columns:
            if model == "regression":
                from machinelearningfunctions import regression_model
                regression_model(st.session_state.new_df, target_column)
            elif model == "classification":
                from machinelearningfunctions import classification_model
                classification_model(st.session_state.new_df, target_column)
            elif model == 'neuralnetwork for regression':
                from machinelearningfunctions import neural_network
                neural_network(
                    st.session_state.new_df, 
                    target_column,
                    n_hidden_layers=n_hidden_layers,
                    neurons_per_layer=neurons_per_layer,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    batch_size=batch_size
                )
            elif model == 'neuralnetwork for classification':
                from machinelearningfunctions import neural_network_classifier
                neural_network_classifier(
                    st.session_state.new_df, 
                    target_column,
                    n_hidden_layers=n_hidden_layers,
                    neurons_per_layer=neurons_per_layer,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    batch_size=batch_size
                )
        else:
            st.error("The specified target column does not exist in the dataset. Please enter a valid column name.")
