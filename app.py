import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os
import zipfile
import requests
import json

def download_model_weights():
    """Downloads fine-tuned model weights directly from Hugging Face Hub."""
    destination = "model_save"
    if not os.path.exists(destination):
        with st.spinner("Downloading fine-tuned transformer weights from Hugging Face Hub... This may take a moment."):
            
            # Direct Hugging Face download URL
            url = "https://huggingface.co/hrithikacharjee/BERA/resolve/main/model_save.zip"
            
            response = requests.get(url, stream=True)
            with open("model_save.zip", "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Unzip into the workspace directory
            with zipfile.ZipFile("model_save.zip", "r") as zip_ref:
                zip_ref.extractall(".")
            
            # Clean up cache zip file
            os.remove("model_save.zip")
            st.success("Model weights loaded successfully from Hugging Face Hub!")

# Trigger the robust Hugging Face download check on initialization
download_model_weights()

MODEL_NAME = 'bert-base-multilingual-cased'

# Defined to accommodate the 3-class prediction probabilities used in the analysis flow
SENTIMENT_LABELS = ['Negative', 'Neutral', 'Positive']
DISPLAY_LABELS = ['Negative', 'Positive'] 
EMOTION_LABELS = ['Angry', 'Fear', 'Happy', 'Love', 'Sad']

class BeraMultiTaskModel(nn.Module):
    def __init__(self, model_name, num_sentiment=3, num_emotion=5):
        super(BeraMultiTaskModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.attn_W = nn.Linear(768, 768)
        self.attn_v = nn.Linear(768, 1, bias=False)
        self.sentiment_hidden = nn.Linear(768, 256)
        self.sentiment_output = nn.Linear(256, num_sentiment)
        self.emotion_hidden = nn.Linear(768, 256)
        self.emotion_output = nn.Linear(256, num_emotion)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state 
        attn_scores = self.attn_v(torch.tanh(self.attn_W(sequence_output))).squeeze(-1)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        alphas = torch.softmax(attn_scores, dim=-1).unsqueeze(1)
        h_review = torch.bmm(alphas, sequence_output).squeeze(1)
        s_hidden = self.activation(self.sentiment_hidden(h_review))
        sentiment_logits = self.sentiment_output(self.dropout(s_hidden))
        e_hidden = self.activation(self.emotion_hidden(h_review))
        emotion_logits = self.emotion_output(self.dropout(e_hidden))
        return sentiment_logits, emotion_logits

@st.cache_resource
def load_assets():
    tokenizer = AutoTokenizer.from_pretrained("./model_save")
    model = BeraMultiTaskModel(
        model_name=MODEL_NAME, 
        num_sentiment=len(SENTIMENT_LABELS), 
        num_emotion=len(EMOTION_LABELS)
    )
    weights_path = "./model_save/bera_3_0_weights.pth"
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return tokenizer, model

st.set_page_config(page_title="BERA Dashboard", layout="wide")

# Ensure a persistent file exists to store user registration data
USER_DB_FILE = "users.json"

def load_registered_users():
    """Loads accounts from the local JSON file database."""
    # Base admin account always exists
    default_users = {"admin": "bera2026"}
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w") as f:
            json.dump(default_users, f)
        return default_users
    try:
        with open(USER_DB_FILE, "r") as f:
            return json.load(f)
    except:
        return default_users

def save_new_user(username, password):
    """Saves a newly registered account directly into the database."""
    users = load_registered_users()
    users[username] = password
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f)

# Initialize login states
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def login():
    st.title("🛍️ BERA Portal")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    # Load all registered users dynamically
    registered_users = load_registered_users()

    with tab1:
        user = st.text_input("Username", key="log_user")
        pw = st.text_input("Password", type="password", key="log_pw")
        if st.button("Login"):
            # Check against the dynamic database dictionary
            if user in registered_users and registered_users[user] == pw:
                st.session_state['logged_in'] = True
                st.rerun()
            else: 
                st.error("Invalid credentials. Please register if you don't have an account.")
                
    with tab2:
        st.subheader("Create New Account")
        reg_email = st.text_input("Email", key="reg_email")
        reg_user = st.text_input("New Username", key="reg_user")
        reg_pw = st.text_input("New Password", type="password", key="reg_pw")
        
        if st.button("Register Now"):
            if not reg_user.strip() or not reg_pw.strip():
                st.error("Username and Password fields cannot be empty!")
            elif reg_user in registered_users:
                st.error("This username already exists! Choose a unique username.")
            else:
                # Instantly save account to the json database matrix
                save_new_user(reg_user, reg_pw)
                st.success(f"🎉 Account '{reg_user}' successfully created! You can now switch to the 'Login' tab to access the app.")

# --- ROUTING LOGIC ---
if not st.session_state['logged_in']:
    login()
else:
    # Everything below runs only if logged_in is True
    tokenizer, model = load_assets()
    st.sidebar.title("BERA Navigation")
    page = st.sidebar.radio("Go to", ["Analysis", "About BERA", "Our Team"])
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

    if page == "Analysis":
        st.title("📊 BERA Review Intelligence")
        input_method = st.radio("Input Method", ["Manual Entry", "Bulk CSV Upload"])
        reviews = []
        df_base = None

        if input_method == "Manual Entry":
            if 'num_reviews' not in st.session_state:
                st.session_state.num_reviews = 1
            reviews_input = []
            for i in range(st.session_state.num_reviews):
                text = st.text_area(f"Review {i+1}", key=f"manual_review_{i}")
                reviews_input.append(text)
            if st.button("➕ Add Review"):
                st.session_state.num_reviews += 1
                st.rerun() 
            reviews = [r for r in reviews_input if r.strip() != ""]
        else:
            file = st.file_uploader("Upload CSV", type=['csv'])
            if file:
                df_base = pd.read_csv(file)
                if 'review' in df_base.columns:
                    reviews = df_base['review'].tolist()
                else: 
                    st.error("CSV must have a column named 'review'")

        if st.button("Analyze with BERA AI"):
            if not reviews:
                st.warning("Please enter at least one review or upload a CSV!")
            else:
                results = []
                with st.spinner("Analyzing Bangla/Banglish sentiments..."):
                    for text in reviews:
                        encoded = tokenizer(text, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
                        with torch.no_grad():
                            s_logits, e_logits = model(encoded['input_ids'], encoded['attention_mask'])
                            
                            s_probs = torch.softmax(s_logits, dim=1)
                            p_neg = s_probs[0][0].item()
                            p_neu = s_probs[0][1].item()
                            p_pos = s_probs[0][2].item()
                            
                            if (p_pos + (p_neu * 0.6)) > p_neg:
                                s_final = "Positive"
                            else:
                                s_final = "Negative"
                            
                            e_pred = torch.argmax(e_logits, dim=1).item()
                        
                        results.append({
                            "review": text, 
                            "Sentiment": s_final, 
                            "Emotion": EMOTION_LABELS[e_pred],
                            "Score_Pos": round(p_pos, 3),
                            "Score_Neg": round(p_neg, 3)
                        })
                
                df_final = pd.DataFrame(results)
                if df_base is not None and 'product' in df_base.columns:
                    df_final['product'] = df_base['product']
                else:
                    df_final['product'] = "General"
                st.session_state['results'] = df_final

        if 'results' in st.session_state:
            res = st.session_state['results']
            st.divider()

            f1, f2, f3 = st.columns(3)
            with f1: view = st.multiselect("Show Output", ["Sentiment", "Emotion", "Score_Pos", "Score_Neg"], default=["Sentiment", "Emotion"])
            with f2: s_filt = st.multiselect("Sentiment Filter", DISPLAY_LABELS, default=DISPLAY_LABELS)
            with f3: e_filt = st.multiselect("Emotion Filter", EMOTION_LABELS, default=EMOTION_LABELS)
            
            p_filt = st.multiselect("Product Filter", res['product'].unique(), default=res['product'].unique())
            filtered = res[(res['Sentiment'].isin(s_filt)) & (res['Emotion'].isin(e_filt)) & (res['product'].isin(p_filt))]
            
            display = st.radio("View Mode", ["Exact Reviews", "Numbers Only"], horizontal=True)

            if display == "Numbers Only":
                st.write(f"### Total Filtered Reviews: {len(filtered)}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Sentiment Breakdown")
                    df_sent = filtered['Sentiment'].value_counts().reset_index()
                    df_sent.columns = ['Sentiment', 'Count'] 
                    st.dataframe(df_sent, hide_index=True, use_container_width=True)
                with col2:
                    st.markdown("#### Emotion Breakdown")
                    df_emo = filtered['Emotion'].value_counts().reset_index()
                    df_emo.columns = ['Emotion', 'Count'] 
                    st.dataframe(df_emo, hide_index=True, use_container_width=True)
            else:
                cols = ['review'] + view
                st.dataframe(filtered[cols], use_container_width=True)

    elif page == "About BERA":
        st.title("About BERA")
        st.markdown("### Bangla E-commerce Review Architecture")
        
        try:
            _, center_q, _ = st.columns([2, 2, 2])
            with center_q:
                st.image("quote.png", use_container_width=True)
        except:
            st.info("Your most unhappy customers are your greatest source of learning. — Bill Gates")

        st.markdown("#### Project Overview")
        st.write("BERA is a multi-task transformer-based framework designed for Bangladeshi e-commerce reviews featuring Bangla, English, and Banglish script.")

        try:
            _, center_m, _ = st.columns([1.5, 2, 1.5])
            with center_m:
                st.image("Model_Architecture.png", caption="BERA Multi-Task Pipeline", use_container_width=True)
        except:
            st.warning("Model Architecture image not found.")

        comparison_data = {
            "Feature": ["Base Backbone", "Input Handling", "Task Setup", "Attention Mechanism"],
            "Standard BanglaBERT": ["12-layer Transformer", "Standard tokenization", "Single-task", "Standard self-attention"],
            "BERA (Our Model)": ["12-layer BanglaBERT", "Code-mixing aware", "Multi-task (Sentiment + Emotion)", "Review-level Attention Pooling"]
        }
        st.table(pd.DataFrame(comparison_data))

    elif page == "Our Team":
        st.title("Our Team")
        st.markdown("### BERA Project Contributors")
        st.divider()

        # Hrithik
        c1_img, c1_txt = st.columns([1, 4])
        with c1_txt:
            st.markdown("#### Hrithik Acharjee")
            st.markdown("**Lead Developer and Machine Learning Researcher** (50% Contribution)")
            st.write("ULAB ID: 222014033")
            st.write("- Architected the BERA Framework\n- Managed Model Training & Optimization\n- Developed Inference Dashboard")
            st.markdown("[LinkedIn Profile](https://www.linkedin.com/in/hrithikacharjee)")

        st.divider()

        # Shahil
        c2_img, c2_txt = st.columns([1, 4])
        with c2_txt:
            st.markdown("#### Md. Shahil Siyam")
            st.markdown("**Data Engineer** (25% Contribution)")
            st.write("ULAB ID: 22201429")
            st.write("- Data Preprocessing & Script Normalization\n- Dataset Curation")

        st.divider()

        # Luna
        c3_img, c3_txt = st.columns([1, 4])
        with c3_txt:
            st.markdown("#### Umme Hani Luna")
            st.markdown("**Research & Documentation** (15% Contribution)")
            st.write("ULAB ID: 222014002")
            st.write("- Literature Review\n- Presentation Design")

        st.divider()

        # Maisha
        c4_img, c4_txt = st.columns([1, 4])
        with c4_txt:
            st.markdown("#### Maliha Rahman Maisha")
            st.markdown("**Technical Writer** (10% Contribution)")
            st.write("ULAB ID: 222014086")
            st.write("- Report Structuring & Editing")

        st.divider()
        st.markdown("#### Research Supervision: **Md. Ahsan Ullah** (ULAB)")