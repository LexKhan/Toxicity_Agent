import streamlit as st
from agent import ToxicityAgent
import pandas as pd
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="Toxicity Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main containers */
    .main-header {
        padding: 1.5rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    
    /* Classification boxes */
    .toxic-box {
        padding: 1.5rem;
        border-radius: 8px;
        background-color: #fff5f5;
        border-left: 4px solid #dc2626;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .neutral-box {
        padding: 1.5rem;
        border-radius: 8px;
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .good-box {
        padding: 1.5rem;
        border-radius: 8px;
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Explanation boxes with colored backgrounds */
    .explanation-toxic {
        padding: 1rem;
        border-radius: 6px;
        background-color: #fee2e2;
        border: 1px solid #fca5a5;
        color: #991b1b;
        margin: 1rem 0;
    }
    
    .explanation-neutral {
        padding: 1rem;
        border-radius: 6px;
        background-color: #fef3c7;
        border: 1px solid #fcd34d;
        color: #92400e;
        margin: 1rem 0;
    }
    
    .explanation-good {
        padding: 1rem;
        border-radius: 6px;
        background-color: #d1fae5;
        border: 1px solid #6ee7b7;
        color: #065f46;
        margin: 1rem 0;
    }
    
    /* Message box */
    .message-box {
        padding: 1rem;
        border-radius: 6px;
        background-color: #f3f4f6;
        border-left: 3px solid #6b7280;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        font-size: 15px;
        font-weight: 500;
        border: none;
        transition: background-color 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 6px;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f9fafb;
    }
    
    /* Headers */
    h1 {
        color: #111827;
        font-weight: 700;
    }
    
    h2 {
        color: #374151;
        font-weight: 600;
    }
    
    h3 {
        color: #4b5563;
        font-weight: 600;
    }
    
    /* Classification badges */
    .badge-toxic {
        background-color: #dc2626;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-neutral {
        background-color: #f59e0b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-good {
        background-color: #16a34a;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        padding: 2rem 0;
        border-top: 1px solid #e5e7eb;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    with st.spinner('Initializing AI Agent... This may take a minute...'):
        try:
            st.session_state.agent = ToxicityAgent()
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            st.info("Make sure you have run `preprocess_data.py` and that Ollama is running with qwen2.5:7b installed.")
            st.stop()

if 'history' not in st.session_state:
    st.session_state.history = []

# Title
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("Toxicity Detection System")
st.markdown("Analyze content for toxicity and receive constructive feedback")
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    **System Components:**
    - QWEN 2.5 7B via Ollama
    - RAG with 3,000 examples
    - Fully local processing
    - No data transmission
    """)
    
    st.header("Statistics")
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        
        col1, col2, col3 = st.columns(3)
        toxic_count = len(df_history[df_history['classification'] == 'TOXIC'])
        neutral_count = len(df_history[df_history['classification'] == 'NEUTRAL'])
        good_count = len(df_history[df_history['classification'] == 'GOOD'])
        
        col1.metric("Toxic", toxic_count)
        col2.metric("Neutral", neutral_count)
        col3.metric("Good", good_count)
        
        st.markdown("---")
        
        # Show recent history
        st.subheader("Recent Analyses")
        for item in reversed(st.session_state.history[-5:]):
            classification_labels = {
                "TOXIC": "🔴 Toxic",
                "NEUTRAL": "🟡 Neutral",
                "GOOD": "🟢 Good"
            }
            st.caption(f"{classification_labels[item['classification']]} - {item['content'][:40]}...")
    else:
        st.info("No analyses yet. Start by analyzing some content.")
    
    st.markdown("---")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# Main content
tab1, tab2, tab3 = st.tabs(["Single Analysis", "Batch Analysis", "Examples"])

# Tab 1: Single Analysis
with tab1:
    st.header("Analyze Content")
    
    # Text input
    content = st.text_area(
        "Enter content to analyze:",
        height=150,
        placeholder="Type or paste any text content here...",
        help="Enter any comment, message, or text you want to analyze for toxicity"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analyze_button = st.button("Analyze Content", type="primary", key="analyze_single")
    
    with col2:
        if st.button("Clear", key="clear_single"):
            st.rerun()
    
    if analyze_button:
        if content.strip():
            with st.spinner('Analyzing content...'):
                try:
                    start_time = time.time()
                    result = st.session_state.agent.detect_and_respond(content)
                    end_time = time.time()
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Analysis Results")
                    
                    classification = result['classification']
                    
                    # Classification badge with appropriate styling
                    if classification == 'TOXIC':
                        st.markdown('<span class="badge-toxic">TOXIC</span>', unsafe_allow_html=True)
                        st.caption(f"Analysis completed in {end_time - start_time:.2f} seconds")
                    elif classification == 'NEUTRAL':
                        st.markdown('<span class="badge-neutral">NEUTRAL</span>', unsafe_allow_html=True)
                        st.caption(f"Analysis completed in {end_time - start_time:.2f} seconds")
                    else:
                        st.markdown('<span class="badge-good">GOOD</span>', unsafe_allow_html=True)
                        st.caption(f"Analysis completed in {end_time - start_time:.2f} seconds")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Explanation with color-coded background
                    st.markdown("**Explanation:**")
                    
                    if classification == 'TOXIC':
                        st.markdown(f'<div class="explanation-toxic">{result["explanation"]}</div>', unsafe_allow_html=True)
                    elif classification == 'NEUTRAL':
                        st.markdown(f'<div class="explanation-neutral">{result["explanation"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="explanation-good">{result["explanation"]}</div>', unsafe_allow_html=True)
                    
                    # Save to history
                    st.session_state.history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'content': content[:100],
                        'classification': classification,
                        'explanation': result['explanation']
                    })
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
        else:
            st.warning("Please enter some content to analyze.")

# Tab 2: Batch Analysis
with tab2:
    st.header("Batch Analysis")
    st.markdown("Analyze multiple pieces of content at once")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a CSV file with a 'content' column",
        type=['csv'],
        help="CSV file should have a column named 'content' with text to analyze"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'content' not in df.columns:
                st.error("CSV file must have a 'content' column")
            else:
                st.success(f"Loaded {len(df)} items")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("Analyze All", type="primary", key="batch_analyze"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    
                    for idx, row in df.iterrows():
                        status_text.text(f"Analyzing {idx + 1}/{len(df)}...")
                        progress_bar.progress((idx + 1) / len(df))
                        
                        result = st.session_state.agent.detect_and_respond(row['content'])
                        results.append({
                            'content': row['content'],
                            'classification': result['classification'],
                            'explanation': result['explanation'],
                            'message': result['message_to_author']
                        })
                    
                    results_df = pd.DataFrame(results)
                    
                    st.success("Analysis complete")
                    
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    toxic = len(results_df[results_df['classification'] == 'TOXIC'])
                    neutral = len(results_df[results_df['classification'] == 'NEUTRAL'])
                    good = len(results_df[results_df['classification'] == 'GOOD'])
                    
                    col1.metric("Toxic", toxic, f"{toxic/len(results_df)*100:.1f}%")
                    col2.metric("Neutral", neutral, f"{neutral/len(results_df)*100:.1f}%")
                    col3.metric("Good", good, f"{good/len(results_df)*100:.1f}%")
                    
                    # Show results
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name=f"toxicity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    # Manual batch input
    st.markdown("---")
    st.subheader("Manual Batch Input")
    
    batch_input = st.text_area(
        "Enter one item per line:",
        height=200,
        placeholder="Item 1\nItem 2\nItem 3...",
        key="batch_manual"
    )
    
    if st.button("Analyze Batch", key="batch_manual_analyze"):
        if batch_input.strip():
            items = [line.strip() for line in batch_input.split('\n') if line.strip()]
            
            progress_bar = st.progress(0)
            results = []
            
            for idx, item in enumerate(items):
                progress_bar.progress((idx + 1) / len(items))
                result = st.session_state.agent.detect_and_respond(item)
                results.append({
                    'content': item,
                    'classification': result['classification']
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
        else:
            st.warning("Please enter some content")

# Tab 3: Examples
with tab3:
    st.header("Example Analyses")
    
    examples = {
        "TOXIC": [
            "You're such a fucking idiot, nobody wants to hear from you.",
            "Women shouldn't be allowed to vote, they're too emotional.",
            "Kill yourself, you worthless piece of trash."
        ],
        "NEUTRAL": [
            "I disagree with your methodology and would like to see more data.",
            "Can you provide sources for these claims?",
            "I'm not convinced by this argument, but I'm open to discussion."
        ],
        "GOOD": [
            "This is excellent work! Thank you for sharing your insights.",
            "I really appreciate your thoughtful perspective on this issue.",
            "Great contribution to the community, keep up the good work!"
        ]
    }
    
    for category, texts in examples.items():
        st.subheader(f"{category} Examples")
        
        for i, text in enumerate(texts, 1):
            with st.expander(f"Example {i}: {text[:60]}..."):
                st.write(f"**Content:** {text}")
                
                if st.button(f"Analyze this example", key=f"{category}_{i}"):
                    with st.spinner('Analyzing...'):
                        result = st.session_state.agent.detect_and_respond(text)
                        
                        st.markdown(f"**Classification:** {result['classification']}")
                        
                        # Color-coded explanation
                        if result['classification'] == 'TOXIC':
                            st.markdown(f'<div class="explanation-toxic">{result["explanation"]}</div>', unsafe_allow_html=True)
                        elif result['classification'] == 'NEUTRAL':
                            st.markdown(f'<div class="explanation-neutral">{result["explanation"]}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="explanation-good">{result["explanation"]}</div>', unsafe_allow_html=True)
                        
                        if result['message_to_author'] != 'N/A':
                            st.markdown("**Message to Author:**")
                            st.markdown(f'<div class="message-box">{result["message_to_author"]}</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("Toxicity Detection System | Powered by QWEN 2.5 via Ollama | 100% Local & Private")
st.markdown('</div>', unsafe_allow_html=True)