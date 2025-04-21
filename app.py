import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import io
import base64
from datetime import datetime, timedelta

# Download NLTK resources if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Configure page
st.set_page_config(
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the look and feel
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #25D366;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #075E54;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #128C7E;
    }
    .footer {
        text-align: center;
        padding: 20px;
        font-size: 14px;
        color: #888;
        margin-top: 40px;
    }
    .highlight {
        color: #128C7E;
        font-weight: bold;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .welcome-header {
        color: #25D366;
        font-weight: bold;
        text-align: center;
    }
    .welcome-text {
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .how-to-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .how-to-header {
        color: #128C7E;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    .upload-zone {
        border: 2px dashed #ddd;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .tabs-wrapper .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .tabs-wrapper .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .tabs-wrapper .stTabs [aria-selected="true"] {
        background-color: #25D366 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Create a header with a WhatsApp-themed design
st.markdown('<h1 class="main-header">WhatsApp Chat Analyzer</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Create a container with padding for centering
    container = st.container()
    # Create columns to center the image
    col1, col2, col3 = st.columns([2,4,2])
    with col2:
        # Display the image centered in the middle column
        st.image("https://cdn-icons-png.flaticon.com/512/0/191.png", use_container_width=True)
    st.markdown("---")
    st.markdown("### Created with ❤️ by:")
    st.markdown("Luv Jain • Neelesh Pahuja • Pratham Shukla")
    st.markdown("---")
    
    # Upload file
    uploaded_file = st.file_uploader("Choose a WhatsApp chat export file (.txt)", type=["txt"])
    
    # Add theme selector
    theme = st.selectbox("Select Theme", ["WhatsApp Green", "Dark Mode", "Ocean Blue", "Sunset Orange"])
    
    # Color schemes based on theme
    color_schemes = {
        "WhatsApp Green": {"primary": "#25D366", "secondary": "#128C7E", "accent": "#075E54", "text": "#000000"},
        "Dark Mode": {"primary": "#192734", "secondary": "#15202B", "accent": "#1DA1F2", "text": "#FFFFFF"},
        "Ocean Blue": {"primary": "#039BE5", "secondary": "#0288D1", "accent": "#01579B", "text": "#000000"},
        "Sunset Orange": {"primary": "#FF6F00", "secondary": "#F57C00", "accent": "#E65100", "text": "#000000"}
    }
    
    # Current color scheme
    colors = color_schemes[theme]
    
    # Add export options
    st.markdown("### Export Options")
    export_format = st.radio("Select export format:", ["CSV", "Excel", "JSON"])

# Function to calculate sentiment scores
def analyze_sentiment(message):
    return sia.polarity_scores(message)['compound']

# Function to download dataframe
def get_download_link(df, filename, file_format):
    if file_format == 'CSV':
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download {filename}</a>'
    elif file_format == 'Excel':
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download {filename}</a>'
    elif file_format == 'JSON':
        json_data = df.to_json(orient='records')
        b64 = base64.b64encode(json_data.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="{filename}.json">Download {filename}</a>'
    return href

if uploaded_file is not None:
    # Add a debug section
    with st.expander("Debug Information", expanded=False):
        st.write("### File Information")
        st.write(f"File name: {uploaded_file.name}")
        st.write(f"File size: {uploaded_file.size} bytes")
        
        # Read raw file
        bytes_data = uploaded_file.getvalue()
        
        # Try multiple encodings
        encodings = ["utf-8", "latin-1", "utf-16", "cp1252"]
        data = None
        
        for encoding in encodings:
            try:
                data = bytes_data.decode(encoding)
                st.success(f"Successfully decoded with {encoding}")
                
                # Show sample of raw data
                st.write("### Raw Data Sample (first 500 chars)")
                st.text_area("Raw data", data[:500], height=200)
                
                # Check for common format indicators
                if "WhatsApp" in data[:200]:
                    st.write("✅ WhatsApp text detected")
                
                lines = data.split('\n')
                st.write(f"Total lines in file: {len(lines)}")
                
                # Show first 5 lines
                st.write("### First 5 lines:")
                for i, line in enumerate(lines[:5]):
                    st.code(line, language=None)
                
                break
            except UnicodeDecodeError:
                continue
        
        if data is None:
            st.error("Could not decode the file with any standard encoding.")
            st.stop()
    
    # Process data
    if st.button("Process Chat Data"):
        # Reset the file position
        uploaded_file.seek(0)
        
        # Read the data again
        bytes_data = uploaded_file.getvalue()
        try:
            data = bytes_data.decode("utf-8")
        except UnicodeDecodeError:
            data = bytes_data.decode("latin-1")
        
        # Process the data
        with st.spinner("Processing..."):
            df = preprocessor.preprocess(data)
            
            if df.empty:
                st.error("No messages found in the chat data. Please check the file format.")
                st.stop()
            
            # Add sentiment analysis
            df['sentiment'] = df['message'].apply(analyze_sentiment)
            df['sentiment_category'] = pd.cut(
                df['sentiment'],
                bins=[-1, -0.3, 0.3, 1],
                labels=['Negative', 'Neutral', 'Positive']
            )
            
            # Calculate message length
            df['message_length'] = df['message'].apply(len)
            
            # Calculate response time
            df['response_time_minutes'] = None
            for user in df['user'].unique():
                user_msgs = df[df['user'] == user].copy()
                if len(user_msgs) > 1:
                    user_msgs = user_msgs.sort_values(by='date')
                    user_msgs['response_time_minutes'] = (user_msgs['date'].diff().dt.total_seconds() / 60)
                    df.loc[user_msgs.index, 'response_time_minutes'] = user_msgs['response_time_minutes']
            
            st.success(f"Successfully processed {len(df)} messages!")
            
            # Show sample data
            st.write("### Sample Processed Data")
            st.dataframe(df.head())
            
            # Extract unique users
            user_list = df['user'].unique().tolist()
            user_list = [user for user in user_list if user != "System"]
            user_list.sort()
            user_list.insert(0, "Overall")
            
            # Store the dataframe in session state so it persists
            st.session_state['df'] = df
            st.session_state['selected_user'] = "Overall"
            st.session_state['user_list'] = user_list

    # Add this section to check if data has been processed
    if 'df' in st.session_state:
        df = st.session_state['df']
        user_list = st.session_state['user_list']
        
        # Update selected user if user changes selection
        selected_user = st.selectbox("Select User for Analysis", user_list, key="user_select")
        
        # Add date range filter
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Filter by date range
        mask = (df['only_date'] >= start_date) & (df['only_date'] <= end_date)
        filtered_df = df[mask]
        
        # Filter by user if not Overall
        if selected_user != 'Overall':
            filtered_df = filtered_df[filtered_df['user'] == selected_user]
        
        # Show export options
        st.markdown("### Export Analysis Data")
        download_filename = f"whatsapp_analysis_{selected_user.replace(' ', '_')}_{start_date}_to_{end_date}"
        st.markdown(get_download_link(filtered_df, download_filename, export_format), unsafe_allow_html=True)
        
        # Analysis tabs
        st.markdown('<div class="tabs-wrapper">', unsafe_allow_html=True)
        tabs = st.tabs(["Overview", "Activity Analysis", "Content Analysis", "Sentiment Analysis", "Advanced Metrics"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        with tabs[0]:  # Overview tab
            st.markdown('<h2 class="sub-header">Chat Overview</h2>', unsafe_allow_html=True)
            
            # Calculate stats
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, filtered_df)
            
            # Display stats with improved UI
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-value">{}</div>'.format(num_messages), unsafe_allow_html=True)
                st.markdown('<div>Messages</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_cols[1]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-value">{}</div>'.format(words), unsafe_allow_html=True)
                st.markdown('<div>Words</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_cols[2]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-value">{}</div>'.format(num_media_messages), unsafe_allow_html=True)
                st.markdown('<div>Media Shared</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with metric_cols[3]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-value">{}</div>'.format(num_links), unsafe_allow_html=True)
                st.markdown('<div>Links Shared</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Find busiest users in the group (only for overall selection)
            if selected_user == 'Overall':
                st.markdown('<h3 class="sub-header">Most Active Users</h3>', unsafe_allow_html=True)
                busy_users, busy_percent = helper.most_busy_users(filtered_df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create a nicer Plotly bar chart
                    busy_users_df = busy_users.reset_index()
                    busy_users_df.columns = ['name', 'message_count']
                    fig = px.bar(
                         busy_users_df,
                         x='name',
                         y='message_count',
                         labels={'name': 'User', 'message_count': 'Number of Messages'},
                         title="Message Count by User",
                         color='message_count',
                          color_continuous_scale='Viridis'
                          )
                    fig.update_layout(
                        xaxis_title="User",
                        yaxis_title="Message Count",
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.dataframe(busy_percent, use_container_width=True)
            
            # Overall chat timeline
            st.markdown('<h3 class="sub-header">Chat Timeline</h3>', unsafe_allow_html=True)
            
            # Create a better timeline visualization with Plotly
            timeline = helper.monthly_timeline(selected_user, filtered_df)
            fig = px.line(
                timeline, 
                x='time', 
                y='message',
                markers=True,
                title="Monthly Message Activity",
                labels={'message': 'Number of Messages', 'time': 'Month'},
                line_shape='spline'
            )
            fig.update_traces(line=dict(color=colors['primary'], width=3))
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Number of Messages"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:  # Activity Analysis tab
            st.markdown('<h2 class="sub-header">Activity Analysis</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<h3 class="sub-header">Daily Activity</h3>', unsafe_allow_html=True)
                daily_timeline = helper.daily_timeline(selected_user, filtered_df)
                
                # Create a better daily timeline with Plotly
                fig = px.line(
                    daily_timeline, 
                    x='only_date', 
                    y='message',
                    title="Daily Message Activity",
                    labels={'message': 'Number of Messages', 'only_date': 'Date'}
                )
                fig.update_traces(line=dict(color=colors['secondary'], width=2))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown('<h3 class="sub-header">Hourly Activity</h3>', unsafe_allow_html=True)
                # Create hourly activity chart
                hourly_activity = filtered_df.groupby('hour').count()['message'].reset_index()
                
                fig = px.bar(
                    hourly_activity,
                    x='hour',
                    y='message',
                    title="Hourly Message Activity",
                    labels={'message': 'Number of Messages', 'hour': 'Hour of Day'}
                )
                fig.update_traces(marker_color=colors['primary'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Weekly activity analysis
            st.markdown('<h3 class="sub-header">Weekly Activity Patterns</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                busy_day = helper.week_activity_map(selected_user, filtered_df)
                fig = px.bar(
                    x=busy_day.index,
                    y=busy_day.values,
                    title="Messages by Day of Week",
                    labels={'x': 'Day', 'y': 'Number of Messages'},
                    color=busy_day.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    xaxis_title="Day of Week",
                    yaxis_title="Number of Messages",
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                busy_month = helper.month_activity_map(selected_user, filtered_df)
                fig = px.bar(
                    x=busy_month.index,
                    y=busy_month.values,
                    title="Messages by Month",
                    labels={'x': 'Month', 'y': 'Number of Messages'},
                    color=busy_month.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Number of Messages",
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap
            st.markdown('<h3 class="sub-header">Weekly Activity Heatmap</h3>', unsafe_allow_html=True)
            user_heatmap = helper.activity_heatmap(selected_user, filtered_df)
            
            fig = px.imshow(
                user_heatmap,
                labels=dict(x="Time of Day", y="Day of Week", color="Message Count"),
                x=user_heatmap.columns,
                y=user_heatmap.index,
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Day of Week"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:  # Content Analysis tab
            st.markdown('<h2 class="sub-header">Content Analysis</h2>', unsafe_allow_html=True)
            
            # Word Cloud
            st.markdown('<h3 class="sub-header">Word Cloud</h3>', unsafe_allow_html=True)
            df_wc = helper.create_wordcloud(selected_user, filtered_df)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            ax.axis("off")
            st.pyplot(fig)
            
            # Most common words
            st.markdown('<h3 class="sub-header">Most Common Words</h3>', unsafe_allow_html=True)
            most_common_df = helper.most_common_words(selected_user, filtered_df)
            
            if not most_common_df.empty:
                fig = px.bar(
                    most_common_df,
                    x=1,
                    y=0,
                    orientation='h',
                    title="Most Common Words",
                    labels={'1': 'Frequency', '0': 'Word'}
                )
                fig.update_traces(marker_color=colors['primary'])
                fig.update_layout(
                    xaxis_title="Frequency",
                    yaxis_title="Word"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No word frequency data available.")
            
            # Emoji analysis
            st.markdown('<h3 class="sub-header">Emoji Analysis</h3>', unsafe_allow_html=True)
            emoji_df = helper.emoji_helper(selected_user, filtered_df)
            
            if not emoji_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(emoji_df, use_container_width=True)
                
                with col2:
                    top_emojis = emoji_df.head(10)
                    fig = px.pie(
                        top_emojis,
                        values=1,
                        names=0,
                        title="Top 10 Emojis",
                        hole=0.3
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No emojis used in the selected messages.")
            
            # Message length distribution - FIXED
            st.markdown('<h3 class="sub-header">Message Length Distribution</h3>', unsafe_allow_html=True)
            
            # Create adaptive bins for message lengths based on the data
            max_length = max(filtered_df['message_length'])
            # Define initial bins
            bins = [0, 10, 25, 50, 100, 200]
            
            # Add larger bins only if needed
            if max_length > 200:
                bins.append(500)
            if max_length > 500:
                bins.append(1000)
            # Always add the max length as the final bin
            if max_length > bins[-1]:
                bins.append(max_length)
            
            # Create the length categories with the fixed bins
            filtered_df['length_category'] = pd.cut(filtered_df['message_length'], bins=bins)
            length_dist = filtered_df['length_category'].value_counts().sort_index()
            
            fig = px.bar(
                x=[str(cat) for cat in length_dist.index],
                y=length_dist.values,
                title="Message Length Distribution",
                labels={'x': 'Character Count Range', 'y': 'Number of Messages'}
            )
            fig.update_traces(marker_color=colors['secondary'])
            fig.update_layout(
                xaxis_title="Message Length (characters)",
                yaxis_title="Number of Messages"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[3]:  # Sentiment Analysis tab
            st.markdown('<h2 class="sub-header">Sentiment Analysis</h2>', unsafe_allow_html=True)
            
            # Overall sentiment distribution
            sentiment_counts = filtered_df['sentiment_category'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Message Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'Positive': '#25D366',
                        'Neutral': '#FFD700',
                        'Negative': '#FF6347'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sentiment over time
                filtered_df['month_year'] = filtered_df['date'].dt.strftime('%Y-%m')
                sentiment_timeline = filtered_df.groupby('month_year')['sentiment'].mean().reset_index()
                
                fig = px.line(
                    sentiment_timeline,
                    x='month_year',
                    y='sentiment',
                    title="Sentiment Trend Over Time",
                    labels={'sentiment': 'Average Sentiment (-1 to 1)', 'month_year': 'Month'}
                )
                fig.update_traces(line=dict(color=colors['accent'], width=3))
                st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment by user (if Overall view)
            if selected_user == 'Overall':
                st.markdown('<h3 class="sub-header">Sentiment by User</h3>', unsafe_allow_html=True)
                
                user_sentiment = filtered_df.groupby('user')['sentiment'].mean().sort_values(ascending=False)
                
                fig = px.bar(
                    x=user_sentiment.index,
                    y=user_sentiment.values,
                    title="Average Sentiment by User",
                    labels={'x': 'User', 'y': 'Average Sentiment (-1 to 1)'},
                    color=user_sentiment.values,
                    color_continuous_scale=['#FF6347', '#FFD700', '#25D366']
                )
                fig.update_layout(
                    xaxis_title="User",
                    yaxis_title="Average Sentiment"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[4]:  # Advanced Metrics tab
            st.markdown('<h2 class="sub-header">Advanced Metrics</h2>', unsafe_allow_html=True)
            
            # Response time analysis (if applicable)
            if 'response_time_minutes' in filtered_df.columns:
                st.markdown('<h3 class="sub-header">Response Time Analysis</h3>', unsafe_allow_html=True)
                
                # Filter out extreme values and nulls
                response_df = filtered_df[
                    (filtered_df['response_time_minutes'].notna()) & 
                    (filtered_df['response_time_minutes'] < 1440)  # Less than 24 hours
                ]
                
                if not response_df.empty:
                    # Average response time by user
                    avg_response = response_df.groupby('user')['response_time_minutes'].mean().sort_values()
                    
                    fig = px.bar(
                        x=avg_response.index,
                        y=avg_response.values,
                        title="Average Response Time by User",
                        labels={'x': 'User', 'y': 'Average Response Time (minutes)'}
                    )
                    fig.update_traces(marker_color=colors['primary'])
                    fig.update_layout(
                        xaxis_title="User",
                        yaxis_title="Average Response Time (minutes)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Response time distribution
                    fig = px.histogram(
                        response_df,
                        x='response_time_minutes',
                        nbins=20,
                        title="Response Time Distribution",
                        labels={'response_time_minutes': 'Response Time (minutes)'}
                    )
                    fig.update_traces(marker_color=colors['secondary'])
                    fig.update_layout(
                        xaxis_title="Response Time (minutes)",
                        yaxis_title="Frequency"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No valid response time data available.")
            
            # Chat activity patterns
            st.markdown('<h3 class="sub-header">Chat Activity Patterns</h3>', unsafe_allow_html=True)
            
            # Messages per day of week and hour
            day_hour_heatmap = pd.pivot_table(
                filtered_df,
                index='day_name',
                columns='hour',
                values='message',
                aggfunc='count',
                fill_value=0
            )
            
            # Ensure proper day ordering
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_hour_heatmap = day_hour_heatmap.reindex(day_order)
            
            fig = px.imshow(
                day_hour_heatmap,
                labels=dict(x="Hour of Day", y="Day of Week", color="Message Count"),
                x=day_hour_heatmap.columns,
                y=day_hour_heatmap.index,
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                title="Message Activity by Day and Hour",
                xaxis_title="Hour of Day",
                yaxis_title="Day of Week"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Chat momentum (messages per day over time)
            st.markdown('<h3 class="sub-header">Chat Momentum</h3>', unsafe_allow_html=True)
            
            # Group by date and count messages
            daily_counts = filtered_df.groupby('only_date').size().reset_index(name='count')
            daily_counts.columns = ['date', 'count']
            
            # Calculate moving average
            daily_counts['7_day_avg'] = daily_counts['count'].rolling(window=7, min_periods=1).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_counts['date'],
                y=daily_counts['count'],
                mode='lines',
                name='Daily Messages',
                line=dict(color='rgba(0,0,255,0.3)')
            ))
            
            fig.add_trace(go.Scatter(
                x=daily_counts['date'],
                y=daily_counts['7_day_avg'],
                mode='lines',
                name='7-Day Average',
                line=dict(color=colors['primary'], width=3)
            ))
            
            fig.update_layout(
                title="Chat Activity Over Time",
                xaxis_title="Date",
                yaxis_title="Number of Messages"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Chat initiative analysis
            st.markdown('<h3 class="sub-header">Chat Initiative Analysis</h3>', unsafe_allow_html=True)
            
            # Only show if there are multiple users
            if len(filtered_df['user'].unique()) > 1:
                # Who starts conversations most often?
                # Group messages by date and get first message of each day
                daily_first = filtered_df.sort_values('date').groupby('only_date').first()
                conversation_starters = daily_first['user'].value_counts()
                
                fig = px.pie(
                    values=conversation_starters.values,
                    names=conversation_starters.index,
                    title="Who Starts Conversations Most Often?",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Chat initiative analysis requires multiple users in the chat.")
                
            # Media sharing patterns
            if filtered_df['media_message'].sum() > 0:
                st.markdown('<h3 class="sub-header">Media Sharing Patterns</h3>', unsafe_allow_html=True)
                
                # Group by month and count media messages
                media_by_month = filtered_df[filtered_df['media_message'] == 1].groupby('month')['message'].count()
                
                fig = px.bar(
                    x=media_by_month.index,
                    y=media_by_month.values,
                    title="Media Messages by Month",
                    labels={'x': 'Month', 'y': 'Number of Media Messages'}
                )
                fig.update_traces(marker_color=colors['accent'])
                fig.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Media Messages Count"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # User engagement metrics
            st.markdown('<h3 class="sub-header">User Engagement Metrics</h3>', unsafe_allow_html=True)
            
            # Calculate metrics per user
            if len(filtered_df['user'].unique()) > 1:
                user_metrics = []
                
                for user in filtered_df['user'].unique():
                    if user == 'System':
                        continue
                        
                    user_df = filtered_df[filtered_df['user'] == user]
                    
                    # Calculate metrics
                    total_messages = len(user_df)
                    avg_length = user_df['message_length'].mean()
                    media_count = user_df['media_message'].sum()
                    links_count = user_df['url_count'].sum() if 'url_count' in user_df.columns else 0
                    avg_sentiment = user_df['sentiment'].mean()
                    
                    user_metrics.append({
                        'User': user,
                        'Total Messages': total_messages,
                        'Avg Message Length': round(avg_length, 1),
                        'Media Shared': media_count,
                        'Links Shared': links_count,
                        'Avg Sentiment': round(avg_sentiment, 2)
                    })
                
                metrics_df = pd.DataFrame(user_metrics)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Show a radar chart for comparing users
                fig = go.Figure()
                
                # Normalize the metrics for better comparison
                radar_df = metrics_df.copy()
                for col in ['Total Messages', 'Avg Message Length', 'Media Shared', 'Links Shared']:
                    if radar_df[col].max() > 0:
                        radar_df[col] = radar_df[col] / radar_df[col].max()
                
                # Convert sentiment score to 0-1 range for visualization
                radar_df['Avg Sentiment'] = (radar_df['Avg Sentiment'] + 1) / 2
                
                # Create radar chart
                for _, row in radar_df.iterrows():
                    fig.add_trace(go.Scatterpolar(
                        r=[row['Total Messages'], row['Avg Message Length'], 
                           row['Media Shared'], row['Links Shared'], 
                           row['Avg Sentiment']],
                        theta=['Messages', 'Msg Length', 'Media', 'Links', 'Sentiment'],
                        fill='toself',
                        name=row['User']
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title="User Engagement Comparison"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("User engagement metrics require multiple users in the chat.")

else:
    # Replace the welcome screen in app.py with this design
 if uploaded_file is None:
    # Logo
    st.markdown("""
    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
        <div style="background-color: #25D366; width: 100px; height: 100px; border-radius: 50%; display: flex; justify-content: center; align-items: center;">
            <svg width="60" height="60" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M20 2H4C2.9 2 2 2.9 2 4V22L6 18H20C21.1 18 22 17.1 22 16V4C22 2.9 21.1 2 20 2Z" fill="white"/>
            </svg>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    
    
    # Subtitle
    
    # Title
    
    
    # Subtitle
    st.markdown('<p style="font-size: 1.2rem; color: #555; margin-bottom: 40px;">Discover insights from your conversations with detailed statistics and interactive visualizations</p>', unsafe_allow_html=True)
    
    # How to use section
    st.markdown('<h2 style="color: #128C7E; text-align: center; margin-bottom: 30px;">How to use</h2>', unsafe_allow_html=True)
    
    # Steps
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin: 10px; text-align: center;">
            <div style="display: flex; justify-content: center; margin-bottom: 15px;">
                <div style="background-color: white; border-radius: 50%; width: 60px; height: 60px; display: flex; justify-content: center; align-items: center;">
                    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z" fill="#25D366"/>
                    </svg>
                </div>
            </div>
            <h3 style="color: #333; margin-bottom: 10px;">Export Chat</h3>
            <p style="color: #666;">Export your WhatsApp chat without media from the mobile app</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin: 10px; text-align: center;">
            <div style="display: flex; justify-content: center; margin-bottom: 15px;">
                <div style="background-color: white; border-radius: 50%; width: 60px; height: 60px; display: flex; justify-content: center; align-items: center;">
                    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M13 2.05v3.03c3.39.49 6 3.39 6 6.92 0 .9-.18 1.75-.5 2.54l2.57 1.53c.56-1.24.88-2.62.88-4.07 0-5.18-3.95-9.45-8.95-9.95zM12 19c-3.87 0-7-3.13-7-7 0-3.53 2.61-6.43 6-6.92V2.05C5.94 2.55 2 6.81 2 12c0 5.52 4.47 10 9.99 10 3.31 0 6.24-1.61 8.06-4.09l-2.6-1.53C16.17 17.98 14.21 19 12 19z" fill="#25D366"/>
                    </svg>
                </div>
            </div>
            <h3 style="color: #333; margin-bottom: 10px;">Process Data</h3>
            <p style="color: #666;">Click "Process Chat Data" to analyze your conversation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin: 10px; text-align: center;">
            <div style="display: flex; justify-content: center; margin-bottom: 15px;">
                <div style="background-color: white; border-radius: 50%; width: 60px; height: 60px; display: flex; justify-content: center; align-items: center;">
                    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M9 16h6v-6h4l-7-7-7 7h4v6zm-4 2h14v2H5v-2z" fill="#25D366"/>
                    </svg>
                </div>
            </div>
            <h3 style="color: #333; margin-bottom: 10px;">Upload File</h3>
            <p style="color: #666;">Upload the exported text file using the sidebar menu</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin: 10px; text-align: center;">
            <div style="display: flex; justify-content: center; margin-bottom: 15px;">
                <div style="background-color: white; border-radius: 50%; width: 60px; height: 60px; display: flex; justify-content: center; align-items: center;">
                    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4zm2 2H5V5h14v14z" fill="#25D366"/>
                    </svg>
                </div>
            </div>
            <h3 style="color: #333; margin-bottom: 10px;">Explore Insights</h3>
            <p style="color: #666;">Discover patterns and statistics through interactive visualizations</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Export instructions
    st.markdown('<div style="background-color: #f8f9fa; border-radius: 10px; padding: 30px; margin-top: 40px;">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #128C7E; margin-bottom: 20px;">How to export your WhatsApp chat:</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display: flex; align-items: flex-start; margin-bottom: 15px;">
        <div style="background-color: #25D366; color: white; width: 25px; height: 25px; border-radius: 50%; display: flex; justify-content: center; align-items: center; margin-right: 15px;">1</div>
        <div>Open the chat you want to analyze</div>
    </div>
    
    <div style="display: flex; align-items: flex-start; margin-bottom: 15px;">
        <div style="background-color: #25D366; color: white; width: 25px; height: 25px; border-radius: 50%; display: flex; justify-content: center; align-items: center; margin-right: 15px;">2</div>
        <div>Tap the three dots (menu) in the top right</div>
    </div>
    
    <div style="display: flex; align-items: flex-start; margin-bottom: 15px;">
        <div style="background-color: #25D366; color: white; width: 25px; height: 25px; border-radius: 50%; display: flex; justify-content: center; align-items: center; margin-right: 15px;">3</div>
        <div>Select "More" > "Export chat"</div>
    </div>
    
    <div style="display: flex; align-items: flex-start;">
        <div style="background-color: #25D366; color: white; width: 25px; height: 25px; border-radius: 50%; display: flex; justify-content: center; align-items: center; margin-right: 15px;">4</div>
        <div>Choose "Without media"</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Upload zone
    st.markdown('<div style="border: 2px dashed #ddd; border-radius: 10px; padding: 30px; text-align: center; margin-top: 40px;">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a WhatsApp chat export file (.txt)", type=["txt"], key="welcome_uploader")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)