"""
SmartPay Guard - ML Based Fraud Detection System
================================================
An interactive Streamlit web application that uses a trained Random Forest model
to detect suspicious or fraudulent transactions in real time.

Designed for beginners, hackathons, and non-technical judges.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SmartPay Guard | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .status-normal {
        background-color: #DCFCE7;
        border: 2px solid #22C55E;
        color: #15803D;
        padding: 18px;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .status-fraud {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        color: #B91C1C;
        padding: 18px;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Saved Model and Dataset
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model_and_metrics():
    model_path = os.path.join(PROJECT_ROOT, "models", "fraud_model.joblib")
    metrics_path = os.path.join(PROJECT_ROOT, "models", "model_metrics.json")
    
    # Auto-train if files are missing
    if not os.path.exists(model_path) or not os.path.exists(metrics_path):
        from models.train_model import train_and_evaluate_model
        train_and_evaluate_model()
        
    model = joblib.load(model_path)
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    return model, metrics

@st.cache_data
def load_data():
    data_path = os.path.join(PROJECT_ROOT, "data", "transactions.csv")
    if not os.path.exists(data_path):
        from data.generate_data import save_dataset
        save_dataset(data_path)
    df = pd.read_csv(data_path)
    return df

try:
    model, metrics = load_model_and_metrics()
    df = load_data()
except Exception as e:
    st.error(f"Error loading system files: {e}")
    st.stop()

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=70)
    st.title("SmartPay Guard")
    st.markdown("**ML Fraud Detection System**")
    st.markdown("---")
    
    app_mode = st.radio(
        "Navigation",
        [
            "🔍 Live Fraud Predictor",
            "📊 Fraud Analytics Dashboard",
            "🎯 Model Performance & Metrics",
            "📁 Dataset Explorer",
            "ℹ️ Beginner's Guide & About"
        ]
    )
    
    st.markdown("---")
    st.info(
        "💡 **Hackathon Tip:**\n\n"
        "SmartPay Guard uses a **Random Forest Classifier** trained on transaction records "
        "to score fraud probability in milliseconds."
    )
    st.caption("Built with Python • Scikit-learn • Streamlit")

# =========================================================
# PAGE 1: LIVE FRAUD PREDICTOR
# =========================================================
if app_mode == "🔍 Live Fraud Predictor":
    st.markdown('<div class="main-header">🛡️ Real-Time Transaction Fraud Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Enter the transaction parameters below or pick a preset demo scenario to test the Machine Learning model.</div>',
        unsafe_allow_html=True
    )
    
    # Preset Demo Buttons
    st.subheader("⚡ Quick Demo Presets")
    col_pre1, col_pre2, col_pre3 = st.columns(3)
    
    if "preset_version" not in st.session_state:
        st.session_state.preset_version = 0
        
    with col_pre1:
        if st.button("🟢 Load Typical Normal Transaction", use_container_width=True):
            st.session_state.preset_version += 1
            st.session_state.amount = 45.50
            st.session_state.type = "Purchase"
            st.session_state.channel = "POS_Terminal"
            st.session_state.hour = 14
            st.session_state.age = 420
            st.session_state.daily_count = 2
            st.session_state.failed_logins = 0
            st.session_state.distance = 4.2
            st.session_state.foreign = "No (Domestic)"
            st.rerun()

    with col_pre2:
        if st.button("🚨 Load High-Risk Suspicious Transaction", use_container_width=True):
            st.session_state.preset_version += 1
            st.session_state.amount = 4850.00
            st.session_state.type = "Transfer"
            st.session_state.channel = "Mobile_App"
            st.session_state.hour = 3
            st.session_state.age = 8
            st.session_state.daily_count = 11
            st.session_state.failed_logins = 3
            st.session_state.distance = 1850.0
            st.session_state.foreign = "Yes (International)"
            st.rerun()

    with col_pre3:
        if st.button("🔄 Reset Inputs to Default", use_container_width=True):
            st.session_state.preset_version += 1
            st.session_state.amount = 120.00
            st.session_state.type = "Payment"
            st.session_state.channel = "Mobile_App"
            st.session_state.hour = 11
            st.session_state.age = 180
            st.session_state.daily_count = 1
            st.session_state.failed_logins = 0
            st.session_state.distance = 12.0
            st.session_state.foreign = "No (Domestic)"
            st.rerun()

    st.markdown("---")
    
    # Input Form
    with st.form("transaction_input_form"):
        st.subheader("📝 Transaction Details")
        
        pv = st.session_state.preset_version
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("##### 💳 Payment & Channel Details")
            
            amount = st.number_input(
                "Transaction Amount ($)",
                min_value=1.0,
                max_value=25000.0,
                value=float(st.session_state.get("amount", 120.00)),
                step=10.0,
                key=f"amount_{pv}",
                help="The monetary value of the transaction in USD."
            )
            
            type_options = ["Payment", "Transfer", "Cash_Out", "Debit", "Purchase"]
            curr_type = st.session_state.get("type", "Payment")
            type_idx = type_options.index(curr_type) if curr_type in type_options else 0
            transaction_type = st.selectbox(
                "Transaction Type",
                type_options,
                index=type_idx,
                key=f"type_{pv}",
                help="Category of financial operation."
            )
            
            channel_options = ["Mobile_App", "Web_Browser", "POS_Terminal", "ATM"]
            curr_channel = st.session_state.get("channel", "Mobile_App")
            channel_idx = channel_options.index(curr_channel) if curr_channel in channel_options else 0
            channel = st.selectbox(
                "Transaction Channel",
                channel_options,
                index=channel_idx,
                key=f"channel_{pv}",
                help="Platform or device interface used to make the payment."
            )
            
            time_of_day_hour = st.slider(
                "Time of Day (Hour: 00:00 to 23:00)",
                min_value=0,
                max_value=23,
                value=int(st.session_state.get("hour", 11)),
                key=f"hour_{pv}",
                help="24-hour format: 0 = midnight, 12 = noon, 23 = 11 PM."
            )
            
        with col_right:
            st.markdown("##### 👤 User Profile & Security Indicators")
            
            account_age_days = st.number_input(
                "Account Age (Days)",
                min_value=1,
                max_value=2000,
                value=int(st.session_state.get("age", 180)),
                step=10,
                key=f"age_{pv}",
                help="How long the user has had an active account. Brand new accounts carry higher risk."
            )
            
            daily_transaction_count = st.slider(
                "Transactions Completed Today",
                min_value=1,
                max_value=25,
                value=int(st.session_state.get("daily_count", 1)),
                key=f"daily_{pv}",
                help="Number of transactions initiated by this account in the last 24 hours."
            )
            
            failed_login_attempts = st.slider(
                "Failed Login Attempts Prior to Transaction",
                min_value=0,
                max_value=5,
                value=int(st.session_state.get("failed_logins", 0)),
                key=f"failed_{pv}",
                help="Number of incorrect password/PIN attempts recorded prior to this session."
            )
            
            distance_from_home_km = st.number_input(
                "Distance From Registered Home (km)",
                min_value=0.1,
                max_value=10000.0,
                value=float(st.session_state.get("distance", 12.0)),
                step=5.0,
                key=f"dist_{pv}",
                help="Distance between the transaction location and the user's primary billing location."
            )
            
            foreign_options = ["No (Domestic)", "Yes (International)"]
            curr_foreign = st.session_state.get("foreign", "No (Domestic)")
            foreign_idx = foreign_options.index(curr_foreign) if curr_foreign in foreign_options else 0
            is_foreign_str = st.selectbox(
                "Foreign / Cross-Border Transaction?",
                foreign_options,
                index=foreign_idx,
                key=f"foreign_{pv}"
            )
            is_foreign_transaction = 1 if "Yes" in is_foreign_str else 0
            
        submit_btn = st.form_submit_button("🔍 Analyze Transaction with AI", use_container_width=True)

    # -----------------------------------------------------
    # Prediction & Explanation
    # -----------------------------------------------------
    if submit_btn:
        # Prepare input data matching pipeline expectation
        input_data = pd.DataFrame([{
            'amount': amount,
            'transaction_type': transaction_type,
            'channel': channel,
            'time_of_day_hour': time_of_day_hour,
            'account_age_days': account_age_days,
            'daily_transaction_count': daily_transaction_count,
            'failed_login_attempts': failed_login_attempts,
            'distance_from_home_km': distance_from_home_km,
            'is_foreign_transaction': is_foreign_transaction
        }])
        
        # Inference using trained pipeline
        prediction = model.predict(input_data)[0]
        prediction_probs = model.predict_proba(input_data)[0]
        fraud_prob = float(prediction_probs[1]) * 100
        normal_prob = float(prediction_probs[0]) * 100
        
        st.markdown("### 📊 Prediction Result")
        
        # Display Banner
        if prediction == 1 or fraud_prob >= 50.0:
            st.markdown(
                '<div class="status-fraud">🚨 ALERT: TRANSACTION FLAGGED AS SUSPICIOUS / FRAUD</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="status-normal">✅ APPROVED: NORMAL LEGITIMATE TRANSACTION</div>',
                unsafe_allow_html=True
            )
            
        # Metric columns
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric(label="Fraud Risk Probability", value=f"{fraud_prob:.1f}%")
        with res_col2:
            st.metric(label="Legitimate Probability", value=f"{normal_prob:.1f}%")
        with res_col3:
            if fraud_prob < 30:
                risk_status = "🟢 Low Risk"
            elif fraud_prob < 60:
                risk_status = "🟡 Medium Risk"
            else:
                risk_status = "🔴 High Risk"
            st.metric(label="Overall Risk Rating", value=risk_status)
            
        # Visual Risk Progress Bar
        st.markdown("**Fraud Risk Scale:**")
        st.progress(int(min(max(fraud_prob, 0), 100)))
        
        # -------------------------------------------------
        # Intelligent Plain-English Explanation Engine
        # -------------------------------------------------
        st.markdown("### 💡 Why Did the Model Make This Decision?")
        
        reasons = []
        
        # Check rule anomalies based on domain logic
        if amount > 3000:
            reasons.append(f"⚠️ **Exceptionally High Amount**: At **${amount:,.2f}**, this transaction is drastically higher than average customer transactions.")
        elif amount > 1500:
            reasons.append(f"⚠️ **Elevated Transaction Amount**: **${amount:,.2f}** is in the top 15% bracket for transaction sizes.")
            
        if 1 <= time_of_day_hour <= 5:
            reasons.append(f"⚠️ **Unusual Nighttime Timing**: Occurred at **{time_of_day_hour:02d}:00 AM**, a time window historically linked to unauthorized account takeovers.")
            
        if failed_login_attempts >= 3:
            reasons.append(f"⚠️ **Failed Password Attempts**: **{failed_login_attempts} failed login attempts** were logged immediately prior to this transaction, signaling potential brute-force access.")
        elif failed_login_attempts >= 1:
            reasons.append(f"ℹ️ **Recent Login Friction**: {failed_login_attempts} failed login attempt was noted.")
            
        if distance_from_home_km > 500:
            reasons.append(f"⚠️ **Abnormal Geographical Distance**: Initiated **{distance_from_home_km:,.1f} km** away from the account holder's registered home city.")
            
        if is_foreign_transaction == 1:
            reasons.append("⚠️ **Cross-Border Activity**: The transaction was routed as an international payment, triggering elevated fraud monitoring.")
            
        if account_age_days < 20:
            reasons.append(f"⚠️ **Brand New Account**: The user account is only **{account_age_days} days old**, lacking an established transaction baseline.")
            
        if daily_transaction_count > 7:
            reasons.append(f"⚠️ **Rapid Transaction Velocity**: This is the **{daily_transaction_count}th transaction today**, representing an unusual burst of activity.")
            
        if transaction_type in ["Transfer", "Cash_Out"]:
            reasons.append(f"ℹ️ **High-Velocity Payment Type**: **{transaction_type}** transactions carry higher inherent risk than point-of-sale retail swipes.")
            
        if not reasons:
            reasons.append("✅ **Consistent User Profile**: Transaction amount, location, channel, and time of day fully align with standard legitimate customer behavior.")
            reasons.append("✅ **Clean Security Record**: Zero failed login attempts and normal daily frequency.")

        with st.expander("🔎 View Detailed Factor Breakdown", expanded=True):
            for reason in reasons:
                st.markdown(f"- {reason}")
                
        # Feature Importance Insights
        st.markdown("#### 🌲 What Key Factors Does the Random Forest Rely On?")
        fi_df = pd.DataFrame(metrics["feature_importances"][:7])
        
        # Clean feature names for beginner display
        clean_names = {
            "amount": "Transaction Amount ($)",
            "distance_from_home_km": "Distance from Home (km)",
            "account_age_days": "Account Age (Days)",
            "time_of_day_hour": "Time of Day (Hour)",
            "failed_login_attempts": "Failed Login Attempts",
            "daily_transaction_count": "Daily Transaction Frequency",
            "is_foreign_transaction_1": "Foreign / International",
            "transaction_type_Payment": "Type: Payment",
            "transaction_type_Transfer": "Type: Transfer",
            "channel_Mobile_App": "Channel: Mobile App"
        }
        fi_df["display_feature"] = fi_df["feature"].map(lambda x: clean_names.get(x, x))
        
        fig, ax = plt.subplots(figsize=(8, 3.2))
        colors = ['#EF4444' if i < 2 else '#3B82F6' for i in range(len(fi_df))]
        y_pos = np.arange(len(fi_df))
        ax.barh(y_pos, fi_df["importance"], color=colors, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(fi_df["display_feature"])
        ax.invert_yaxis()  # top-down
        ax.set_xlabel("Relative Importance (%)")
        ax.set_title("Top 7 Drivers of Fraud Detection in Random Forest Model", fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# =========================================================
# PAGE 2: FRAUD ANALYTICS DASHBOARD
# =========================================================
elif app_mode == "📊 Fraud Analytics Dashboard":
    st.markdown('<div class="main-header">📊 SmartPay Transaction Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Exploratory Data Analysis (EDA) of 5,000 synthetic transaction records.</div>', unsafe_allow_html=True)
    
    # KPI Row
    total_txns = len(df)
    total_fraud = int(df['is_fraud'].sum())
    total_normal = total_txns - total_fraud
    fraud_pct = (total_fraud / total_txns) * 100
    avg_normal_amt = df[df['is_fraud'] == 0]['amount'].mean()
    avg_fraud_amt = df[df['is_fraud'] == 1]['amount'].mean()
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Transactions", f"{total_txns:,}")
    kpi2.metric("Suspicious / Fraud Cases", f"{total_fraud:,}", f"{fraud_pct:.1f}% rate")
    kpi3.metric("Avg Normal Transaction", f"${avg_normal_amt:,.2f}")
    kpi4.metric("Avg Fraud Transaction", f"${avg_fraud_amt:,.2f}", f"+${avg_fraud_amt - avg_normal_amt:,.2f}")
    
    st.markdown("---")
    
    # Charts Row 1
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.subheader("💳 Fraud Rate by Transaction Type")
        type_stats = df.groupby('transaction_type')['is_fraud'].agg(['count', 'mean']).reset_index()
        type_stats['fraud_rate'] = type_stats['mean'] * 100
        type_stats = type_stats.sort_values(by='fraud_rate', ascending=False)
        
        fig1, ax1 = plt.subplots(figsize=(6, 3.8))
        ax1.bar(type_stats['transaction_type'], type_stats['fraud_rate'], color='#3B82F6')
        ax1.set_ylabel("Fraud Rate (%)")
        ax1.set_title("Transfers & Cash Outs Have the Highest Fraud Incidence", fontsize=10, fontweight='bold')
        plt.xticks(rotation=20)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)
        st.caption("Insight: Transfers and Cash Outs are favored by fraudsters to quickly siphon funds.")

    with col_c2:
        st.subheader("📱 Fraud Rate by Channel")
        chan_stats = df.groupby('channel')['is_fraud'].agg(['count', 'mean']).reset_index()
        chan_stats['fraud_rate'] = chan_stats['mean'] * 100
        chan_stats = chan_stats.sort_values(by='fraud_rate', ascending=False)
        
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        ax2.bar(chan_stats['channel'], chan_stats['fraud_rate'], color='#10B981')
        ax2.set_ylabel("Fraud Rate (%)")
        ax2.set_title("Fraud Incidence Across Platforms", fontsize=10, fontweight='bold')
        plt.xticks(rotation=20)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
        st.caption("Insight: Mobile Apps and Web Browsers represent prime targets for remote fraud.")
        
    st.markdown("---")
    
    # Charts Row 2
    col_c3, col_c4 = st.columns(2)
    
    with col_c3:
        st.subheader("⏰ Transaction Volume by Hour of Day")
        hourly_fraud = df[df['is_fraud'] == 1].groupby('time_of_day_hour').size()
        hourly_normal = df[df['is_fraud'] == 0].groupby('time_of_day_hour').size()
        
        fig3, ax3 = plt.subplots(figsize=(6, 3.8))
        ax3.plot(hourly_normal.index, hourly_normal.values, label='Normal Transactions', color='#22C55E', linewidth=2)
        ax3.plot(hourly_fraud.index, hourly_fraud.values, label='Fraud Transactions', color='#EF4444', linewidth=2, linestyle='--')
        ax3.set_xlabel("Hour of Day (0-23)")
        ax3.set_ylabel("Number of Transactions")
        ax3.set_title("Spike in Fraud During Midnight Hours (01:00 - 05:00)", fontsize=10, fontweight='bold')
        ax3.legend()
        ax3.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
        st.caption("Insight: Nighttime hours see lower normal volume but concentrated fraud spikes.")

    with col_c4:
        st.subheader("💰 Transaction Amount Comparison")
        fig4, ax4 = plt.subplots(figsize=(6, 3.8))
        
        # Clip amounts for clear boxplot visualization
        normal_amts = df[df['is_fraud'] == 0]['amount'].clip(upper=4000)
        fraud_amts = df[df['is_fraud'] == 1]['amount'].clip(upper=4000)
        
        ax4.boxplot([normal_amts, fraud_amts], tick_labels=['Normal ($)', 'Fraud ($)'], patch_artist=True,
                    boxprops=dict(facecolor='#E0E7FF', color='#4338CA'),
                    medianprops=dict(color='#DC2626', linewidth=2))
        ax4.set_ylabel("Amount ($)")
        ax4.set_title("Fraudulent Transactions Have Much Higher Median Amounts", fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)
        st.caption("Insight: Fraudsters attempt significantly larger amounts before accounts get frozen.")

# =========================================================
# PAGE 3: MODEL PERFORMANCE & METRICS
# =========================================================
elif app_mode == "🎯 Model Performance & Metrics":
    st.markdown('<div class="main-header">🎯 Random Forest Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detailed evaluation on an unseen hold-out test set (20% of dataset = 1,000 transactions).</div>', unsafe_allow_html=True)
    
    # Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%", help="Percentage of total correct predictions (Normal + Fraud).")
    m2.metric("Precision", f"{metrics['precision'] * 100:.1f}%", help="Out of all transactions flagged as fraud, how many were truly fraud?")
    m3.metric("Recall", f"{metrics['recall'] * 100:.1f}%", help="Out of all actual fraud attacks, how many did we detect?")
    m4.metric("F1-Score", f"{metrics['f1_score'] * 100:.1f}%", help="Harmonic mean balancing Precision and Recall.")
    m5.metric("ROC-AUC", f"{metrics['roc_auc'] * 100:.1f}%", help="Model's ability to discriminate between normal and fraud across all thresholds.")
    
    st.markdown("---")
    
    col_cm, col_exp = st.columns([1, 1])
    
    with col_cm:
        st.subheader("🗂️ Confusion Matrix Heatmap")
        cm = np.array(metrics['confusion_matrix'])
        
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        cax = ax_cm.matshow(cm, cmap='Blues')
        fig_cm.colorbar(cax)
        
        for i in range(2):
            for j in range(2):
                ax_cm.text(j, i, f"{cm[i, j]:,}", ha='center', va='center', color='black' if cm[i, j] < 500 else 'white', fontsize=14, fontweight='bold')
                
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(['Pred: Normal', 'Pred: Fraud'])
        ax_cm.set_yticklabels(['Actual: Normal', 'Actual: Fraud'])
        ax_cm.set_xlabel("Predicted Label", labelpad=10)
        ax_cm.set_ylabel("True Actual Label")
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)
        
    with col_exp:
        st.subheader("📖 Understanding the Evaluation Metrics")
        st.markdown(f"""
        In financial fraud detection, evaluation metrics have specific business meanings:
        
        - **True Negatives ({cm[0][0]})**: Legitimate transactions correctly approved without customer inconvenience.
        - **False Positives ({cm[0][1]})**: Legitimate transactions mistakenly flagged as fraud ("false alarms"). Lower is better to avoid annoying good users.
        - **False Negatives ({cm[1][0]})**: Real fraud that slipped past the system unnoticed. In banking, this represents financial loss.
        - **True Positives ({cm[1][1]})**: Fraud attempts successfully intercepted and blocked!
        
        #### Why Precision & Recall Matter More Than Simple Accuracy
        If 90% of transactions are normal, a dumb model that predicts "Normal" for everything would have 90% accuracy, but catch **0%** of fraud!
        That's why our system balances **Precision ({metrics['precision']*100:.1f}%)** and **Recall ({metrics['recall']*100:.1f}%)** using `class_weight='balanced'`.
        """)

    st.markdown("---")
    st.subheader("🌲 Why Random Forest Classifier?")
    st.markdown("""
    1. **Ensemble Voting**: Combines 120 individual decision trees to make robust predictions, drastically reducing the risk of overfitting.
    2. **Handles Non-Linear Patterns**: Fraud patterns involve complex interactions (e.g., high amount *combined with* midnight hour *and* new device) that linear models miss.
    3. **Resistant to Noise & Outliers**: Financial transaction data often has extreme outliers (e.g., legitimate high-wealth purchases); Random Forests handle these gracefully.
    4. **Explainable Feature Importance**: Unlike black-box neural networks, Random Forest allows bank risk officers to see exactly which features drove the decision.
    """)

# =========================================================
# PAGE 4: DATASET EXPLORER
# =========================================================
elif app_mode == "📁 Dataset Explorer":
    st.markdown('<div class="main-header">📁 Transaction Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Browse and filter the underlying transaction records.</div>', unsafe_allow_html=True)
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_status = st.selectbox("Filter by Fraud Status", ["All Transactions", "Normal Only", "Suspicious/Fraud Only"])
    with col_f2:
        filter_type = st.multiselect("Filter by Type", options=df['transaction_type'].unique(), default=list(df['transaction_type'].unique()))
    with col_f3:
        max_price = st.slider("Max Amount Filter ($)", min_value=10.0, max_value=float(df['amount'].max()), value=float(df['amount'].max()))
        
    filtered_df = df.copy()
    if filter_status == "Normal Only":
        filtered_df = filtered_df[filtered_df['is_fraud'] == 0]
    elif filter_status == "Suspicious/Fraud Only":
        filtered_df = filtered_df[filtered_df['is_fraud'] == 1]
        
    if filter_type:
        filtered_df = filtered_df[filtered_df['transaction_type'].isin(filter_type)]
        
    filtered_df = filtered_df[filtered_df['amount'] <= max_price]
    
    st.markdown(f"**Showing {len(filtered_df):,} matching transactions:**")
    st.dataframe(filtered_df.head(250), use_container_width=True)
    
    st.markdown("#### 📐 Statistical Summary")
    st.dataframe(filtered_df.describe().round(2), use_container_width=True)

# =========================================================
# PAGE 5: BEGINNER'S GUIDE & ABOUT
# =========================================================
elif app_mode == "ℹ️ Beginner's Guide & About":
    st.markdown('<div class="main-header">ℹ️ Beginner\'s Guide to ML Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">How machine learning works in modern FinTech and banking.</div>', unsafe_allow_html=True)
    
    st.markdown(r"""
    ### 🚀 The Hackathon Story
    Every day, payment processors like PayPal, Visa, Stripe, and UPI handle billions of dollars in payments.
    Traditional rule-based systems (e.g. *"block anything over $500"*) fail because:
    - Fraudsters adapt quickly.
    - Honest customers get their cards blocked when buying a laptop or traveling abroad.
    
    **SmartPay Guard** solves this by using Machine Learning to analyze 9 distinct behavioral signals simultaneously.
    
    ---
    
    ### 🧠 How It Works Step-by-Step
    1. **Data Collection**: When a user clicks 'Pay', the app collects transaction metrics (amount, time, device channel, distance).
    2. **Feature Preprocessing**:
       - Numbers (amount, hour, distance) are standardized so no single huge number dominates.
       - Categories (Transfer, Payment, Mobile App) are converted into machine-readable numeric formats (One-Hot Encoding).
    3. **Random Forest Inference**:
       - 120 decision trees evaluate the transaction features.
       - Each tree casts a vote: *Normal* or *Fraud*.
       - The percentage of fraud votes becomes the **Fraud Risk Probability**.
    4. **Explainability Engine**:
       - Instead of just saying "Yes/No", the system highlights the exact risk factors that caused the flag (e.g. midnight transaction + high amount + failed logins).
       
    ---
    
    ### 📚 Key Terminology for Non-Programmers
    | Term | What It Means |
    |---|---|
    | **Features** | The clues or inputs the model looks at (amount, time, location). |
    | **Target / Label** | What we are trying to predict (`0 = Normal`, `1 = Fraud`). |
    | **Training** | The process where the algorithm learns patterns by studying thousands of past examples. |
    | **Test Set** | A set of records the model has never seen before, used to test real-world accuracy. |
    | **Precision** | Accuracy when flagging fraud (preventing false alarms). |
    | **Recall** | Ability to catch actual thieves (preventing financial loss). |
    """)
    
    st.success("🎉 You are now ready to present **SmartPay Guard** with confidence at your hackathon!")
