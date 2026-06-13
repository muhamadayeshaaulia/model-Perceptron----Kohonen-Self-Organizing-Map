import streamlit as st

def apply_custom_styles():
    """Mengoleskan gaya CSS kustom untuk membuat UI Streamlit modern, elegan, dan premium (adaptif Light/Dark mode)."""
    st.markdown(
        """
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap');

        /* Terapkan font secara global, biarkan warna teks diatur secara alami oleh tema Streamlit */
        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif !important;
        }

        /* Latar belakang adaptif dengan gradien radial halus */
        .stApp {
            background: radial-gradient(circle at 50% 0%, var(--secondary-background-color) 0%, var(--background-color) 100%) !important;
        }

        /* Container halaman */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            max-width: 1200px !important;
        }

        /* Sidebar Styling (Solid & Adaptif) */
        section[data-testid="stSidebar"], 
        section[data-testid="stSidebar"] > div {
            background-color: var(--secondary-background-color) !important;
            opacity: 1 !important;
            backdrop-filter: none !important;
        }
        
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.15) !important;
        }
        
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
            background-color: var(--secondary-background-color) !important;
        }

        /* Header Style dengan Font Kustom */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            color: inherit !important; /* Mewarisi warna teks tema (putih di dark, gelap di light) */
        }
        
        /* h1 khusus dengan gradien */
        .stApp h1:not(.no-gradient) {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800 !important;
            font-size: 2.5rem !important;
            margin-bottom: 1.5rem !important;
            line-height: 1.25 !important;
        }
        
        h2 {
            border-bottom: none !important;
            font-weight: 700 !important;
            margin-top: 2rem !important;
            margin-bottom: 1rem !important;
        }

        h3 {
            font-weight: 600 !important;
            margin-top: 1.5rem !important;
        }

        /* Pembatas Elemen (Sleek Divider Adaptif) */
        hr {
            margin: 2rem 0 !important;
            border: 0 !important;
            height: 1px !important;
            background: linear-gradient(to right, rgba(128, 128, 128, 0.05), rgba(128, 128, 128, 0.25), rgba(128, 128, 128, 0.05)) !important;
        }

        /* Card Metrik & Kartu Kustom (Glassmorphism & Bayangan Lembut Adaptif) */
        div[data-testid="metric-container"], .premium-card, .custom-card {
            background-color: var(--secondary-background-color) !important;
            border: 1px solid rgba(128, 128, 128, 0.18) !important;
            border-radius: 16px !important;
            padding: 1.25rem 1.5rem !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            color: inherit !important; /* Warisi warna teks yang sesuai */
        }
        
        div[data-testid="metric-container"]:hover, .premium-card:hover, .custom-card:hover {
            transform: translateY(-4px) !important;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.05) !important;
            border-color: rgba(79, 70, 229, 0.4) !important;
        }

        /* Style khusus untuk custom-card agar isinya pas */
        .custom-card {
            height: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
        }

        .custom-card h3, .custom-card h4 {
            margin-top: 0 !important;
            margin-bottom: 0.75rem !important;
            color: inherit !important;
        }

        .custom-card p, .custom-card li, .custom-card span {
            color: inherit !important;
            opacity: 0.85 !important;
            font-size: 0.92rem !important;
            line-height: 1.5 !important;
        }

        /* Nilai Metrik */
        div[data-testid="stMetricValue"] {
            font-family: 'Outfit', sans-serif !important;
            font-size: 2.25rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em !important;
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            color: inherit !important;
            opacity: 0.75 !important;
        }

        /* Input Fields (Sleek Border & Focus Ring Adaptif) */
        div[data-baseweb="input"], div[data-baseweb="select"], .stNumberInput input, .stTextInput input {
            border-radius: 12px !important;
            border: 1.5px solid rgba(128, 128, 128, 0.25) !important;
            background-color: var(--background-color) !important;
            font-weight: 500 !important;
            color: inherit !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {
            border-color: #4F46E5 !important;
            box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.15) !important;
        }

        /* Tombol Utama (Gradien Premium dengan Hover Mikro-Animasi) */
        div.stButton > button {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
            color: white !important;
            border: none !important;
            padding: 0.75rem 2.25rem !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.3) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            width: 100% !important;
            cursor: pointer !important;
        }
        
        div.stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px 0 rgba(79, 70, 229, 0.45) !important;
            background: linear-gradient(135deg, #4338CA 0%, #6D28D9 100%) !important;
            color: white !important;
        }
        
        div.stButton > button:active {
            transform: translateY(0) !important;
            box-shadow: 0 4px 10px 0 rgba(79, 70, 229, 0.3) !important;
        }
        
        div.stButton > button:disabled {
            background: rgba(128, 128, 128, 0.15) !important;
            color: rgba(128, 128, 128, 0.5) !important;
            box-shadow: none !important;
            transform: none !important;
            cursor: not-allowed !important;
        }

        /* Tab Modern (Pill-Style Adaptif) */
        button[data-baseweb="tab"] {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            border-radius: 10px !important;
            padding: 0.6rem 1.25rem !important;
            margin-right: 0.5rem !important;
            border: 1px solid transparent !important;
            background-color: transparent !important;
            color: inherit !important;
            opacity: 0.75 !important;
            transition: all 0.2s ease !important;
        }
        
        button[data-baseweb="tab"]:hover {
            color: #4F46E5 !important;
            background-color: rgba(79, 70, 229, 0.08) !important;
            opacity: 1 !important;
        }
        
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: rgba(79, 70, 229, 0.12) !important;
            color: #4F46E5 !important;
            border: 1px solid rgba(79, 70, 229, 0.25) !important;
            opacity: 1 !important;
        }

        /* Expanders yang Dihaluskan (Adaptif) */
        .streamlit-expanderHeader {
            background-color: var(--secondary-background-color) !important;
            border: 1px solid rgba(128, 128, 128, 0.15) !important;
            border-radius: 12px !important;
            padding: 0.75rem 1.25rem !important;
            font-weight: 600 !important;
            font-family: 'Outfit', sans-serif !important;
            color: inherit !important;
            transition: all 0.2s ease !important;
            margin-bottom: 0.5rem !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02) !important;
        }
        
        .streamlit-expanderHeader:hover {
            border-color: #4F46E5 !important;
            background-color: rgba(79, 70, 229, 0.02) !important;
        }
        
        .streamlit-expanderContent {
            border: 1px solid rgba(128, 128, 128, 0.15) !important;
            border-top: none !important;
            border-bottom-left-radius: 12px !important;
            border-bottom-right-radius: 12px !important;
            margin-top: -0.75rem !important;
            margin-bottom: 0.75rem !important;
            padding: 1.25rem !important;
            background-color: var(--background-color) !important;
            color: inherit !important;
        }

        /* Alert/Notifikasi Modern (Adaptif) */
        div[data-testid="stNotification"] {
            border-radius: 12px !important;
            border: 1px solid rgba(128, 128, 128, 0.15) !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03) !important;
            font-weight: 500 !important;
        }
        
        .stAlert {
            border-radius: 12px !important;
        }

        /* DataFrame & Tabel Modern (Adaptif) */
        .stDataFrame, [data-testid="stTable"] {
            border: 1px solid rgba(128, 128, 128, 0.15) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02) !important;
        }

        /* Animasi Transisi Halaman */
        .stApp {
            animation: fadeInPage 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        @keyframes fadeInPage {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* List Spacing */
        li {
            margin-bottom: 0.5rem !important;
        }
        
        /* Caption */
        .stCaption {
            color: inherit !important;
            opacity: 0.65 !important;
            font-size: 0.85rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
