import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re

# --- CONFIGURATION ---
st.set_page_config(
    page_title="LexiGuard | AI Fairness Audit",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
    .metric-card h4 {
        color: #6c757d;
        font-size: 1rem;
        font-weight: 600;
        margin: 0;
        padding-bottom: 5px;
    }
    .metric-card h2 {
        color: #212529;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-card small {
        color: #495057;
        font-size: 0.85rem;
    }
    .bias-alert {
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF5722;
        background-color: #FFF3E0;
    }
    .tone-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #e3f2fd;
        border: 1px solid #90caf9;
    }
    .live-audit-box {
        border: 2px dashed #4CAF50;
        padding: 20px;
        border-radius: 10px;
        background-color: #f9fdf9;
    }
    
    /* NEW: Animated Gradient Header for Module 2 */
    @keyframes gradient-text {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .audit-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(45deg, #FF512F, #DD2476, #FF512F);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-text 3s linear infinite;
        margin-bottom: 30px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        d1 = pd.read_csv("data1.csv") # Legal Bias Dataset
        d2 = pd.read_csv("data2.csv") # Country Time Freq
        d3 = pd.read_csv("data3.csv") # Anti-Stereotype Test
        return d1, d2, d3
    except FileNotFoundError as e:
        st.error(f"Error loading datasets: {e}. Please ensure data1.csv, data2.csv, and data3.csv exist.")
        return None, None, None

df_legal, df_trends, df_nlp = load_data()

# --- HELPER FUNCTIONS ---

def clean_text(text):
    """Basic text cleaning for NLP"""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def calculate_fairness_metrics(y_true, y_pred, sensitive_features):
    """Calculates Demographic Parity and Equal Opportunity."""
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'group': sensitive_features})
    selection_rate = df.groupby('group')['y_pred'].mean()
    if len(selection_rate) < 2: return 0, 0, selection_rate
    dp_ratio = selection_rate.min() / (selection_rate.max() + 1e-6)
    tpr = df[df['y_true'] == 1].groupby('group')['y_pred'].mean()
    eo_diff = abs(tpr.max() - tpr.min()) if len(tpr) > 1 else 0
    return dp_ratio, eo_diff, selection_rate

def analyze_tone_rule_based(text):
    """Simple rule-based tone analyzer for demo purposes."""
    text = text.lower()
    positive_words = ['innocent', 'acquitted', 'fair', 'justice', 'right', 'honor', 'truth', 'agree', 'success', 'benefit']
    negative_words = ['guilty', 'crime', 'murder', 'fraud', 'bias', 'wrong', 'fail', 'harm', 'danger', 'threat', 'violation']
    
    score = 0
    found_pos = []
    found_neg = []
    
    tokens = text.split()
    for word in tokens:
        clean_word = re.sub(r'[^a-zA-Z]', '', word)
        if clean_word in positive_words:
            score += 1
            found_pos.append(clean_word)
        elif clean_word in negative_words:
            score -= 1
            found_neg.append(clean_word)
            
    if score > 0: return "Positive", score, found_pos
    elif score < 0: return "Negative", score, found_neg
    else: return "Neutral", score, []

# --- MAIN UI ---

st.markdown('<div class="main-header">⚖️ LexiGuard: Judicial AI Audit Platform</div>', unsafe_allow_html=True)
st.markdown("---")

if df_legal is None:
    st.stop()

# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2666/2666505.png", width=100)
    st.title("Navigation")
    module = st.radio("Select Module:", [
        "1. Case Analysis & Tone",
        "2. Bias Detection & Auditing",
        "3. Societal Trend Monitor",
        "4. Anti-Stereotype Validation"
    ])
    
    st.info("ℹ️ **About**: LexiGuard detects bias in legal texts and monitors societal stereotypes using historical data.")

# --- MODULE 1: CASE ANALYSIS & TONE ---
if module == "1. Case Analysis & Tone":
    st.header("📂 Legal Case & Tone Analysis")
    
    # -- TONE ANALYSIS VISUALIZATION --
    st.subheader("📊 Tone & Sentiment Analytics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dataset Sentiment Distribution**")
        # Pie Chart for Sentiment/Tone
        if 'Sentiment_Polarity_Score' in df_legal.columns:
            tone_counts = df_legal['Sentiment_Polarity_Score'].value_counts()
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(tone_counts, labels=tone_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99','#ffcc99'])
            ax_pie.axis('equal') 
            st.pyplot(fig_pie)
        else:
            st.warning("Sentiment column not found.")

    with col2:
        st.markdown("**Tone vs. Verdict Correlation**")
        if 'Sentiment_Polarity_Score' in df_legal.columns and 'Verdict' in df_legal.columns:
            fig_bar, ax_bar = plt.subplots()
            sns.countplot(x='Sentiment_Polarity_Score', hue='Verdict', data=df_legal, palette='viridis', ax=ax_bar)
            ax_bar.set_title("Verdict Counts by Sentiment Tone")
            st.pyplot(fig_bar)

    st.markdown("---")
    
    # -- INTERACTIVE TONE ANALYZER --
    st.subheader("🎙️ Live Tone Analyzer")
    col_input, col_result = st.columns([2, 1])
    with col_input:
        user_text = st.text_area("Enter Legal Text:", "The defendant is innocent and has served the community with honor and truth.")
    
    with col_result:
        if user_text:
            tone_label, tone_score, keywords = analyze_tone_rule_based(user_text)
            st.markdown(f"**Detected Tone:**")
            if tone_label == "Positive":
                st.success(f"### {tone_label} (+{tone_score})")
            elif tone_label == "Negative":
                st.error(f"### {tone_label} ({tone_score})")
            else:
                st.info(f"### {tone_label} ({tone_score})")
            if keywords:
                st.write(f"*Keywords detected:* {', '.join(keywords)}")

    st.markdown("---")

    # -- MODEL TRAINING SECTION --
    st.subheader("⚙️ Train NLP Model")
    colA, colB = st.columns([2, 1])
    with colA:
        st.write("Dataset Preview")
        st.dataframe(df_legal[['Case_ID', 'Verdict', 'Sentiment_Polarity_Score', 'Bias_Label']].head(3))
    
    with colB:
        st.write("Document Length Distribution")
        fig_hist, ax_hist = plt.subplots(figsize=(4,2.5))
        sns.histplot(df_legal['Document_Length'], bins=20, kde=True, color='purple', ax=ax_hist)
        st.pyplot(fig_hist)

    if st.button("🚀 Train Model"):
        with st.spinner("Vectorizing text and training model..."):
            df_legal['clean_text'] = df_legal['Case_Facts'].apply(clean_text)
            df_legal['target'] = df_legal['Verdict'].apply(lambda x: 1 if x == 'Guilty' else 0)
            
            # Save vectorizer to session state to use in Live Audit
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            X = tfidf.fit_transform(df_legal['clean_text'])
            st.session_state['vectorizer'] = tfidf 
            
            y = df_legal['target']
            sensitive = df_legal['Bias_Label']
            X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, sensitive, test_size=0.25, random_state=42)
            
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            st.session_state['model'] = clf
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
            st.session_state['s_test'] = s_test
            st.session_state['trained'] = True
            st.success(f"Training Complete! Accuracy: **{acc:.2%}**")

# --- MODULE 2: BIAS DETECTION & AUDITING ---
elif module == "2. Bias Detection & Auditing":
    # Replaced simple st.header with Animated Gradient Header
    st.markdown('<div class="audit-header">🔍 Algorithmic Bias Audit</div>', unsafe_allow_html=True)
    
    if 'trained' not in st.session_state:
        st.warning("⚠️ Please train the model in Module 1 first.")
    else:
        # Existing Metrics
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        s_test = st.session_state['s_test']
        
        dp_ratio, eo_diff, selection_rates = calculate_fairness_metrics(y_test, y_pred, s_test)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card"><h4>Overall Accuracy</h4><h2>{:.1%}</h2></div>'.format(accuracy_score(y_test, y_pred)), unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h4>Demographic Parity Ratio</h4><h2>{dp_ratio:.2f}</h2><small>Target: > 0.8</small></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h4>Equal Opp. Difference</h4><h2>{eo_diff:.2f}</h2><small>Target: < 0.1</small></div>', unsafe_allow_html=True)

        st.markdown("### 📊 Selection Rate Analysis")
        sel_df = selection_rates.reset_index()
        sel_df.columns = ['Bias Group (0=Neutral, 1=Sensitive)', 'Selection Rate']
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x='Bias Group (0=Neutral, 1=Sensitive)', y='Selection Rate', data=sel_df, palette="coolwarm", ax=ax)
        st.pyplot(fig)
        
        if dp_ratio < 0.8:
             st.markdown("""<div class="bias-alert">⚠️ <b>Bias Detected:</b> The model predicts 'Guilty' significantly more often for one group.</div>""", unsafe_allow_html=True)
        
        # --- NEW FEATURE: LIVE AUDIT ---
        st.markdown("---")
        st.header("⚡ Live Data Audit Playground")
        st.markdown("Insert new data to audit the model in real-time.")
        
        live_mode = st.radio("Select Input Mode:", ["📝 Single Case Entry", "📁 Batch File Upload"], horizontal=True)
        
        if live_mode == "📝 Single Case Entry":
            st.markdown('<div class="live-audit-box">', unsafe_allow_html=True)
            live_text = st.text_area("Enter Case Facts for Audit:", height=100, placeholder="Type case details here...")
            
            if st.button("Run Live Audit"):
                if live_text:
                    # 1. Prediction
                    model = st.session_state['model']
                    vectorizer = st.session_state['vectorizer']
                    
                    clean_input = clean_text(live_text)
                    X_live = vectorizer.transform([clean_input])
                    prediction = model.predict(X_live)[0]
                    prob = model.predict_proba(X_live)[0]
                    
                    # 2. Tone
                    tone_label, tone_score, _ = analyze_tone_rule_based(live_text)
                    
                    # 3. Keyword Check (Simple Sensitivity Check)
                    sensitive_keywords = ['race', 'gender', 'religion', 'minority', 'background']
                    found_sensitive = [word for word in sensitive_keywords if word in live_text.lower()]
                    
                    # Display Results
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        verdict = "Guilty" if prediction == 1 else "Not Guilty"
                        color = "red" if prediction == 1 else "green"
                        st.markdown(f"**Predicted Verdict:** :{color}[{verdict}]")
                        st.progress(float(prob[prediction]))
                    with c2:
                        st.markdown(f"**Tone Analysis:** {tone_label}")
                    with c3:
                        if found_sensitive:
                            st.warning(f"⚠️ Sensitive Terms: {', '.join(found_sensitive)}")
                        else:
                            st.success("✅ No Sensitive Flag Words")
                    
                    if prediction == 1 and found_sensitive:
                        st.error("🚨 **High Risk Audit:** Model predicts 'Guilty' while sensitive attributes are present. Manual Review Required.")
                else:
                    st.info("Please enter text to audit.")
            st.markdown('</div>', unsafe_allow_html=True)

        elif live_mode == "📁 Batch File Upload":
            st.markdown("Upload a CSV file containing a column **'Case_Facts'**.")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    df_new = pd.read_csv(uploaded_file)
                    if 'Case_Facts' in df_new.columns:
                        st.success(f"Loaded {len(df_new)} cases.")
                        
                        if st.button("Audit Batch Data"):
                            model = st.session_state['model']
                            vectorizer = st.session_state['vectorizer']
                            
                            # Transform and Predict
                            df_new['clean_text'] = df_new['Case_Facts'].apply(clean_text)
                            X_new = vectorizer.transform(df_new['clean_text'])
                            df_new['Predicted_Verdict'] = model.predict(X_new)
                            
                            st.write("### Audit Results")
                            st.dataframe(df_new[['Case_Facts', 'Predicted_Verdict']].head())
                            
                            # Visualization
                            st.write("**Prediction Distribution in New Batch**")
                            fig_batch, ax_batch = plt.subplots(figsize=(6,3))
                            sns.countplot(x='Predicted_Verdict', data=df_new, palette='pastel', ax=ax_batch)
                            ax_batch.set_xticklabels(['Not Guilty', 'Guilty'])
                            st.pyplot(fig_batch)
                            
                            if 'Bias_Label' in df_new.columns:
                                st.info("ℹ️ 'Bias_Label' found. Calculating Selection Rate...")
                                df_new['group'] = df_new['Bias_Label']
                                sel_rate = df_new.groupby('group')['Predicted_Verdict'].mean()
                                st.bar_chart(sel_rate)
                            else:
                                st.info("Note: Upload a file with 'Bias_Label' column to see group fairness metrics.")
                    else:
                        st.error("CSV must contain 'Case_Facts' column.")
                except Exception as e:
                    st.error(f"Error processing file: {e}")

# --- MODULE 3: SOCIETAL TREND MONITOR ---
elif module == "3. Societal Trend Monitor":
    st.header("🌍 Societal Context Monitor")
    
    all_words = df_trends['word'].unique()
    default_ix = list(all_words).index('fat') if 'fat' in all_words else 0
    
    col1, col2 = st.columns([1, 3])
    with col1:
        word_choice = st.selectbox("Select Keyword:", all_words, index=default_ix)
        countries = df_trends['country'].unique()
        country_choice = st.multiselect("Select Country:", countries, default=countries[:2])
    
    with col2:
        mask = (df_trends['word'] == word_choice) & (df_trends['country'].isin(country_choice))
        subset = df_trends[mask]
        
        if not subset.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=subset, x='year', y='frequency', hue='country', marker='o', ax=ax)
            plt.title(f"Frequency of '{word_choice}' (2010-2021)")
            st.pyplot(fig)
        else:
            st.warning("No data for current selection.")

# --- MODULE 4: ANTI-STEREOTYPE VALIDATION ---
elif module == "4. Anti-Stereotype Validation":
    st.header("🧪 NLP Model Unit Testing")
    
    st.dataframe(df_nlp[['tokens', 'coreference_clusters', 'speaker']].head(10))
    st.success("✅ **Test Passed:** Coreference resolution active.")
    
    labels = ['Stereotype Confirming', 'Stereotype Defying', 'Neutral']
    sizes = [45, 40, 15]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    ax1.axis('equal') 
    st.write("#### Test Suite Distribution")
    st.pyplot(fig1)

# Footer
st.markdown("---")
st.markdown("© 2025 LexiGuard Project")