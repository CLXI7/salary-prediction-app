import streamlit as st, pandas as pd, joblib, numpy as np

df_orig = pd.read_csv('ds_salaries.csv')

st.title("💵 Data Science Salary Predictor")

# Load
model = joblib.load('best_model.pkl')
le_exp = joblib.load('le_exp.pkl')
le_company = joblib.load('le_company.pkl')
X_columns = joblib.load('feature_columns.pkl')  # Saved list of processed features

# Country code to country name mapping
country_codes = {
    'ES': 'Spain', 'US': 'United States', 'CA': 'Canada', 'DE': 'Germany',
    'GB': 'United Kingdom', 'NG': 'Nigeria', 'IN': 'India', 'HK': 'Hong Kong',
    'PT': 'Portugal', 'NL': 'Netherlands', 'CH': 'Switzerland', 'CF': 'Central African Republic',
    'FR': 'France', 'AU': 'Australia', 'FI': 'Finland', 'UA': 'Ukraine',
    'IE': 'Ireland', 'IL': 'Israel', 'GH': 'Ghana', 'AT': 'Austria',
    'CO': 'Colombia', 'SG': 'Singapore', 'SE': 'Sweden', 'SI': 'Slovenia',
    'MX': 'Mexico', 'UZ': 'Uzbekistan', 'BR': 'Brazil', 'TH': 'Thailand',
    'HR': 'Croatia', 'PL': 'Poland', 'KW': 'Kuwait', 'VN': 'Vietnam',
    'CY': 'Cyprus', 'AR': 'Argentina', 'AM': 'Armenia', 'BA': 'Bosnia and Herzegovina',
    'KE': 'Kenya', 'GR': 'Greece', 'MK': 'North Macedonia', 'LV': 'Latvia',
    'RO': 'Romania', 'PK': 'Pakistan', 'IT': 'Italy', 'MA': 'Morocco',
    'LT': 'Lithuania', 'BE': 'Belgium', 'AS': 'American Samoa', 'IR': 'Iran',
    'HU': 'Hungary', 'SK': 'Slovakia', 'CN': 'China', 'CZ': 'Czech Republic',
    'CR': 'Costa Rica', 'TR': 'Turkey', 'CL': 'Chile', 'PR': 'Puerto Rico',
    'DK': 'Denmark', 'BO': 'Bolivia', 'PH': 'Philippines', 'DO': 'Dominican Republic',
    'EG': 'Egypt', 'ID': 'Indonesia', 'AE': 'United Arab Emirates', 'MY': 'Malaysia',
    'JP': 'Japan', 'EE': 'Estonia', 'HN': 'Honduras', 'TN': 'Tunisia',
    'RU': 'Russia', 'DZ': 'Algeria', 'IQ': 'Iraq', 'BG': 'Bulgaria',
    'JE': 'Jersey', 'RS': 'Serbia', 'NZ': 'New Zealand', 'MD': 'Moldova',
    'LU': 'Luxembourg', 'MT': 'Malta'
}

# Invert the dictionary for reverse mapping
country_names_to_codes = {v: k for k, v in country_codes.items()}

# Employment type codes and full names
employment_type_map = {
    'FT': 'Full-time',
    'PT': 'Part-time',
    'CT': 'Contract',
    'FL': 'Freelance'
}
employment_type_reverse = {v: k for k, v in employment_type_map.items()}

# Company size codes and full names
company_size_map = {
    'S': 'Small (<50)',
    'M': 'Medium (50-250)',
    'L': 'Large (250+)'
}
company_size_reverse = {v: k for k, v in company_size_map.items()}



# User input
exp = st.selectbox("Experience Level", ['Entry','Mid','Senior','Executive'])
remote = st.selectbox("Remote Ratio", [0, 50, 100])
company_size_full = st.selectbox("Company Size", [company_size_map[cs] for cs in ['S', 'M', 'L']])
company_size = company_size_reverse[company_size_full]
emp_type_full = st.selectbox("Employment Type", [employment_type_map[et] for et in sorted(df_orig['employment_type'].unique())])
emp_type = employment_type_reverse[emp_type_full]
job_title = st.selectbox("Job Title", sorted(df_orig['job_title'].unique()))
residence_full = st.selectbox(
    "Employee Residence",
    sorted(country_names_to_codes.keys())
)
residence = country_names_to_codes[residence_full]

comp_loc_full = st.selectbox(
    "Company Location",
    sorted(country_names_to_codes.keys())
)
comp_loc = country_names_to_codes[comp_loc_full]

# Input frame
inp = {
    'experience_level': le_exp.transform([exp])[0],
    'remote_ratio': remote,
    'company_size': le_company.transform([company_size])[0],
}
# Add one-hot features
for cat in ['employment_type_'+emp_type, 'job_title_'+job_title,
            'employee_residence_'+residence, 'company_location_'+comp_loc]:
    inp[cat] = 1

X_inp = pd.DataFrame([inp]).reindex(columns=X_columns, fill_value=0)

# Predict
if st.button("Predict"):
    sal = model.predict(X_inp)[0]
    st.success(f"Estimated Salary: ${sal:,.2f} USD")
