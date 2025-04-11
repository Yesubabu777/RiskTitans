import streamlit as st
import pandas as pd
import pickle
import json
import plotly.graph_objects as go

# --- Add background image using custom CSS ---
def add_bg_image(image_path):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_path});
            background-size: cover;
            background-position: center;
        }}
        </style>
        """, 
        unsafe_allow_html=True
    )

# Call the function to add the background image


# Cache the state-county mapping loading
@st.cache_data
def load_state_county_map():
    with open(r"C:\Users\Lenovo\state_county_final_dict.json", 'r') as f:
        return json.load(f)

# Cache the model loading
@st.cache_resource
def load_models(risk_columns):
    models = {} 
    for risk_col in risk_columns:
        model_path = rf'C:\Users\Lenovo\{risk_col}_model.pkl' # Absolute path for each model
        with open(model_path, 'rb') as file:
            models[risk_col] = pickle.load(file)
    return models

# Load the state-county mapping (cached)
state_county_map = load_state_county_map()

# --- Load the models (cached) ---
risk_columns = ['Risk_PN_ensemble', 'Risk_NP_ensemble', 'Risk_PF_ensemble', 'Risk_FP_ensemble', 'Risk_FN_ensemble', 'Risk_NF_ensemble']
models = load_models(risk_columns)

# --- Title ---
st.title("Risk Score Prediction for your Business")

# --- Sidebar Inputs ---
st.sidebar.header("Enter few details listed below:")

# Select State
state = st.sidebar.selectbox("Select State", list(state_county_map.keys()))

# Dynamically show counties based on state
county_options = state_county_map[state]
county = st.sidebar.selectbox("Select County", county_options)

# Other Inputs
hour = st.sidebar.slider("What's the timeframe you are looking at?", 0, 23)

business_category = st.sidebar.selectbox(
    "Business Category",
    ['Airports & Air Transport', 'Hospitals & Healthcare Facilities',
       'Power Plants (Electricity Generation & Distribution)',
       'Water Treatment & Utilities']  # You can populate this dynamically too if needed
)

incident_type = st.sidebar.selectbox(
    "Incident Type",
    ['Fire', 'Tornado', 'Severe Storm', 'Hurricane', 'Flood',
       'Severe Ice Storm', 'Snowstorm', 'Mud/Landslide', 'Earthquake',
       'Coastal Storm']  # Customize this list as needed
)

# Select Current State of Business
business_state = st.sidebar.selectbox(
    "What is the current state of your business?",
    ['Partial', 'Full Operational', 'Non Operational']
)

# Create DataFrame for model input
input_data = pd.DataFrame({
    'name': [county],  # Assuming 'name' = county
    'state': [state],
    'Hour': [hour],
    'Business_category': [business_category],
    'incidentType': [incident_type]
})

# --- Prediction ---
if st.sidebar.button("Predict Risk Scores"):
    st.subheader("Predicted Risk Scores")

    predictions = {}
    for risk_col in risk_columns:
        prediction = models[risk_col].predict(input_data)[0]
        predictions[risk_col] = prediction

    # Prepare risk scores based on business state
    if business_state == 'Partial':
        transitions = {
            'Partial to Non Operational': predictions['Risk_PN_ensemble'],
            'Partial to Full Operational': predictions['Risk_PF_ensemble']
        }
        colors = ['lightblue', 'deepskyblue']  # Lighter shades for better contrast
    elif business_state == 'Full Operational':
        transitions = {
            'Full to Non Operational': predictions['Risk_FN_ensemble'],
            'Full to Partial Operational': predictions['Risk_FP_ensemble']
        }
        colors = ['lightgreen', 'limegreen']  # Lighter shades for better contrast
    elif business_state == 'Non Operational':
        transitions = {
            'Non Operational to Full Operational': predictions['Risk_NF_ensemble'],
            'Non Operational to Partial': predictions['Risk_NP_ensemble']
        }
        colors = ['lightyellow', 'gold']  # Lighter shades for better contrast

    # Identify which transition has a higher value
    transition_labels = list(transitions.keys())
    transition_values = list(transitions.values())
    max_value = max(transition_values)
    max_index = transition_values.index(max_value)

    # Set the darker color for the riskier transition
    bar_colors = [colors[0] if i != max_index else colors[1] for i in range(len(transitions))]

    # Create the bar chart
    fig = go.Figure(data=[go.Bar(
        x=transition_labels,
        y=transition_values,  # Keep the values in points (not percentage)
        marker=dict(color=bar_colors),
        text=[f'{value:.2f} points' for value in transition_values],
        textposition='outside',
    )])

    # Set labels and title
    fig.update_layout(
        title=f"Risk Scores for {business_state} Business",
        xaxis_title="Risk Transition",
        yaxis_title="Risk Score (points)",
        template="plotly_dark"  # Make the background dark for better visibility
    )

    # Show the bar chart
    st.plotly_chart(fig)

    # Conclusive line
    st.write(f"**Conclusion**: The transition from '{transition_labels[max_index]}' is more likely to happen as it has the higher risk score of **{max_value:.2f} points**.")
# --- Load incident dataset (could be historical or synthetic) ---
@st.cache_data
def load_incident_data():
    return pd.read_csv(r"C:\Users\Lenovo\changed_data.csv")  # Path to your data file

incident_df = load_incident_data()

# --- Filter top 15 counties in selected state and incident type ---
# --- Filter data based on selected state and incident type ---
filtered_df = incident_df[
    (incident_df['state'] == state) & 
    (incident_df['incidentType'] == incident_type)
]

# --- Count number of incidents per county ---
incident_counts = filtered_df['name'].value_counts().reset_index()
incident_counts.columns = ['County', 'Incident Count']

# --- Take Top 15 Counties ---
top_15 = incident_counts.head(15)

# --- Plot using Plotly ---
fig_top15 = go.Figure(data=[go.Bar(
    x=top_15['County'],  # County names
    y=top_15['Incident Count'],
    marker_color='orange',
    text=top_15['Incident Count'],
    textposition='outside',
)])

fig_top15.update_layout(
    title=f"Top 15 Counties in {state} with Most '{incident_type}' Incidents",
    xaxis_title="County",
    yaxis_title="Number of Incidents",
    template="plotly_white"
)

# --- Display chart ---
st.subheader("Top 15 Counties by Incident Count")
st.plotly_chart(fig_top15)
