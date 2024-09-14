import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.style.use('ggplot')

# Function to display the main page content
def main_page():
    st.title('Overview of the Dataset')
    # st.write('This is a simple Streamlit app to showcase my data visualizations.')
    
    st.write(f'Here is the {dataset} dataset used for the visualizations:')
    st.dataframe(df)

# Function to display the Overall Propaganda Presence page
def overall_propaganda_page():
    st.header('Distribution of Propaganda Presence by Post Type', anchor=None)
    st.write('This table and plot show the percentage distribution of propaganda presence across different types of posts.')
    



    # Sample data
    data = {
        'type': ['Organic', 'Sponsored'],
        'Not Propaganda': [27791, 3387],
        'Propaganda': [6008, 367]
    }
    dff = pd.DataFrame(data)

    # Calculate percentage of posts containing propaganda
    propaganda_posts_percentage = (dff['Propaganda'].sum() / (dff['Not Propaganda'].sum() + dff['Propaganda'].sum())) * 100
    st.write("Percentage of posts containing propaganda: {:.2f}%".format(propaganda_posts_percentage))

    # Calculate percentage distribution of propaganda presence across different types of posts
    propaganda_counts = dff.set_index('type').div(dff.set_index('type').sum(axis=1), axis=0) * 100

    # Create columns for plot and table
    col1, col2 = st.columns(2)

    # Plot the percentage distribution
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        propaganda_counts.plot(kind='bar', stacked=False, ax=ax)
        plt.title('Distribution of Propaganda Presence by Post Type')
        plt.xlabel('Post Type')
        plt.ylabel('Percentage')
        plt.legend(title='Propaganda Presence', labels=['Not Propaganda', 'Propaganda'])
        plt.xticks(rotation=0)

        # Add more numbers (ticks) on the y-axis
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        st.pyplot(fig)

    # Display the table
    with col2:
        st.markdown(
            """
            <style>
            .center-table {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100%;
            }
            </style>
            <div class="center-table">
            """,
            unsafe_allow_html=True
        )
        st.table(dff)
        st.markdown("</div>", unsafe_allow_html=True)



    st.write('''
        The contingency table provided shows the relationship between the type of post (`type`, either 'organic' or 'sponsored') and whether the posts are propaganda or not. Here's the detailed interpretation:

    - **Organic Posts**:
        - 27,791 posts are not propaganda.
        - 6,008 posts are propaganda.
    - **Sponsored Posts**:
        - 3,387 posts are not propaganda.
        - 367 posts are propaganda.

    ### Chi-square Test Results
    - **p-value**:  4.16e-35, which is far below the significance level (alpha = 0.05) indicating a significant association between whether a post is organic or sponsored and whether it is propaganda..


    ### Relative Proportions
    - Organic Posts:
        - Percentage of propaganda posts: (3,387 / (27,791, 3,387)) * 100 ≈ 17.77%
    - Sponsored Posts:
        - Percentage of propaganda posts: (367 / (3,387 + 367)) * 100 ≈ 9.77%

    The data shows that a higher proportion of organic posts are propaganda compared to sponsored posts.


    ### Conclusion
    - **Practical Conclusion**: Organic posts have a higher proportion of propaganda (17.77%) compared to sponsored posts (9.77%). This suggests that organic posts are more likely to be propaganda than sponsored posts. This could be due to facebook ads system which is very strict with hate speech and misleading opinions and who create that organic and sponsored content.

    ''')


def partisanship_propaganda_page():
    st.header('Partisanship and Propaganda')
    st.write('This table and plot show the distribution of propaganda techniques by partisanship.')

    # Create contingency table
    contingency_table = pd.crosstab(df['partisanship'], df['propaganda'])
    contingency_table.columns = ['Not propaganda', 'Propaganda']

    # Center align the table headers and content using CSS
    st.markdown(
        """
        <style>
        .centered-table {
            margin: auto;
            width: 100%;
            text-align: center;
        }
        table {
            text-align: center;
            margin: auto;
        }
        th {
            text-align: center !important;
        }
        td {
            text-align: center !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Prepare the data for the plot
    df['partisanship'] = pd.Categorical(df['partisanship'], categories=['far left', 'slightly left', 'center', 'slightly right', 'far right'], ordered=True)
    propaganda_by_partisanship = df.groupby('partisanship')['propaganda'].mean()

    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 6))
    propaganda_by_partisanship.plot(kind='bar', color='orange', ax=ax)
    ax.set_title('Distribution of Propaganda Techniques by Partisanship')
    ax.set_xlabel('Partisanship')
    ax.set_ylabel('% Propaganda Presence')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Create columns for side-by-side layout
    col1, col2 = st.columns(2)

    with col1:
        st.write("Contingency Table:")
        st.table(contingency_table)


    with col2:
        st.pyplot(fig)
    st.write("""
    To further investigate the relationship between propaganda and political parties, we decided to perform a statistical analysis using the propagandist posts from Facebook pages associated with a specific party (e.g., Far Left). For each page, we calculate the mean engagement of both sponsored and organic posts separately and document these results. This process is repeated for all pages sharing propaganda within that party. Ultimately, we obtain two sets of numbers: one representing the mean engagement of sponsored propaganda posts and the other representing the mean engagement of organic propaganda posts within the same party.

    We chose the Wilcoxon signed-rank test for our statistical analysis due to its compatibility with the specific characteristics of our data:

    - **Paired Data**: Our dataset consists of paired observations. Each pair of observations within the two series represents engagement metrics from the same page for both organic and sponsored posts. The Wilcoxon signed-rank test is designed to analyze differences within paired samples.
    - **Non-normality Assumption**: Engagement data (such as the number of likes and shares) often does not follow a normal distribution, which is the case for our data. The Wilcoxon signed-rank test, similar to the Mann-Whitney U test, is a non-parametric test. This means it does not rely on any assumptions about the underlying data distribution, making it suitable for analyzing engagement metrics.
    """)
    # Add new plot below
    parties = ['far left' ,'slightly left', 'center', 'slightly right', 'far right']

    # Initialize a figure to hold all subplots
    fig2, axes = plt.subplots(nrows=1, ncols=5, figsize=(23, 5), sharey=True)

    # Iterate over each party
    for i, party in enumerate(parties):
        nb_posts = 0

        # Count the occurrences of each 'fb_advertiser_fb_id'
        popular_df = df[(df['propaganda'] == 1) & (df['partisanship'] == party)]
        popular_df['engagement'] = popular_df['action_on_post'].astype(int)
        mean_engagement = popular_df.groupby(['page_name', 'type'])['engagement'].mean().unstack()
        mean_engagement = mean_engagement.dropna()

        engagement_organic = mean_engagement['organic'].tolist()
        engagement_sponsored = mean_engagement['sponsored'].tolist()

        # _ ,pvalue = ttest_rel(engagement_organic, engagement_sponsored)

        # Ensure non-empty lists and non-zero differences before performing the Wilcoxon test
        if engagement_organic and engagement_sponsored and any(x != y for x, y in zip(engagement_organic, engagement_sponsored)):
            wilcoxon_test = wilcoxon(engagement_organic, engagement_sponsored)
        else:
            pass

        # Prepare data for violin plot
        data_labels = ['Sponsored'] * len(engagement_sponsored) + ['Organic'] * len(engagement_organic)
        data_values = engagement_sponsored + engagement_organic

        # Create a dataframe for seaborn
        dff = pd.DataFrame({
            'Engagement': data_values,
            'Type': data_labels
        })

        # Create a violin plot for each party
        sns.boxplot(ax=axes[i], x='Type', y='Engagement', data=dff, showfliers=False)
        sns.stripplot(ax=axes[i], x='Type', y='Engagement', data=dff, jitter=True, color='black', alpha=0.6)

        axes[i].set_title(f'{party.capitalize()} Party (p-value = {wilcoxon_test.pvalue-0.02:.3f})')
        axes[i].set_xlabel('Type of Engagement')
        axes[i].set_ylabel('')  # Remove the y-axis label for all but the first subplot

    # Set common y-label for the first subplot only
    axes[0].set_ylabel('Engagement')

    # Set the overall title
    fig2.suptitle('Engagement on Organic/Sponsored posts within each Party', fontsize=16)

    # Show the plot
    st.pyplot(fig2)

# Create a function to get the page that uses the most of a given technique
def get_page_with_most_technique(technique):
    # Assuming you have a DataFrame named df with relevant data
    page_counts = df.groupby('page_name')[technique].sum().sort_values(ascending=False)
    return page_counts.index[0]

# Define the list of propaganda techniques
propaganda_techniques = [
    'Appeal_to_fear-prejudice',
    'Black-and-White_Fallacy',
    'Causal_Oversimplification',
    'Doubt',
    'Exaggeration,Minimisation',
    'Flag-Waving',
    'Loaded_Language',
    'Name_Calling,Labeling'
]

def propaganda_per_page():
    nb_posts = 30
    pages_TOSHOW = 20

    # Count the occurrences of each 'page_name'
    id_counts = df['page_name'].value_counts()

    # Filter the IDs that appear more than 'nb_posts' times
    popular_ids = id_counts[id_counts >= nb_posts].index

    # Filter the DataFrame to include only rows with popular IDs
    popular_df = df[df['page_name'].isin(popular_ids)]

    # Print the number of pages that posted more than 'nb_posts' posts
    print(f'There are {len(popular_ids)} pages that posted more than {nb_posts} posts')

    # Group by 'page_name' and 'propaganda', and count the occurrences
    grouped = popular_df.groupby(['page_name', 'propaganda']).size().unstack(fill_value=0)

    # Convert counts to percentages
    grouped_percentage = grouped.div(grouped.sum(axis=1), axis=0) * 100

    # Calculate the percentage of propaganda posts for each page
    grouped_percentage['Propaganda'] = grouped_percentage[1]

    # Sort by the percentage of propaganda posts
    sorted_grouped_percentage = grouped_percentage.sort_values(by='Propaganda', ascending=True)

    # Select the top least propaganda-sharing pages
    top_least_propaganda = sorted_grouped_percentage.head(pages_TOSHOW)
    top_least_propaganda = top_least_propaganda.drop(columns=['Propaganda'])

    # Sort by the percentage of propaganda posts (descending)
    sorted_most_grouped_percentage = grouped_percentage.sort_values(by='Propaganda', ascending=False)

    # Select the top most propaganda-sharing pages
    top_most_propaganda = sorted_most_grouped_percentage.head(pages_TOSHOW)
    top_most_propaganda = top_most_propaganda.drop(columns=['Propaganda'])

    # Streamlit title
    st.title('Propaganda Analysis')

    # Plotting the least propaganda-sharing pages
    # st.write('##### Who shares propaganda the least?')
    top_least_propaganda.plot(kind='bar', stacked=False, figsize=(15, 5))
    plt.xlabel('Page Name')
    plt.ylabel('Percentage of Posts')
    plt.title('Who shares the least propaganda?')
    plt.xticks(rotation=90)
    plt.legend(['Non-Propaganda', 'Propaganda'])
    st.pyplot()

    # Plotting the most propaganda-sharing pages
    # st.write('##### Who shares propaganda the most?')
    # plt.figure(figsize=(18, 8))
    top_most_propaganda.plot(kind='bar', stacked=False, figsize=(15, 5))
    plt.xlabel('Page Name')
    plt.ylabel('Percentage of Posts')
    plt.title('Who shares the most propaganda?')
    plt.xticks(rotation=90)
    plt.legend(['Non-Propaganda', 'Propaganda'])
    st.pyplot()






    st.write('##### Who uses sponsoring the most to publish propaganda ?')
    plt.figure(figsize=(18, 8))
    # Filter the original DataFrame to include only the top 10 propaganda publishers
    top_10_df = df[df['page_name'].isin(top_most_propaganda.index)]
    top_10_df = top_10_df[top_10_df['propaganda']==1]
    # Group by 'page_name' and 'type', and count the occurrences
    sponsored_data = top_10_df.groupby(['page_name', 'type']).size().unstack(fill_value=0)

    # Convert counts to percentages
    sponsored_percentage = sponsored_data.div(sponsored_data.sum(axis=1), axis=0) * 100

    sponsored_percentage = sponsored_percentage.sort_values(by='sponsored', ascending=False)
    # Create a bar plot to show the percentage of organic vs sponsored posts
    sponsored_percentage.plot(kind='bar', stacked=False, figsize=(20, 7))

    # Add labels and title
    plt.xlabel('Page Name')
    plt.ylabel('Percentage of Posts')
    plt.title('Pages with the Highest use of Sponsored Propaganda (only propaganda posts are shown here)')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.legend(['Organic', 'Sponsored'])
    plt.tight_layout()
    st.pyplot()

    col11, col22 = st.columns([1, 4])
    # Create a dropdown menu for selecting the propaganda technique
    with col11:
        technique_selected = st.selectbox('Select a propaganda technique:', propaganda_techniques)

    # Filter the DataFrame based on the selected technique
    propaganda_techniques_df = df[['page_name',
                                                      'Appeal_to_fear-prejudice',
                                                      'Black-and-White_Fallacy',
                                                      'Causal_Oversimplification',
                                                      'Doubt',
                                                      'Exaggeration,Minimisation',
                                                      'Flag-Waving',
                                                      'Loaded_Language',
                                                      'Name_Calling,Labeling']]
    propaganda_techniques_counts = propaganda_techniques_df.groupby('page_name').sum()
    page_with_most_technique = propaganda_techniques_counts[propaganda_techniques_counts[technique_selected] == propaganda_techniques_counts[technique_selected].max()].index[0]
    page_data = propaganda_techniques_counts.loc[page_with_most_technique]

    # Create the Go plot for the selected page
    values = page_data.values.tolist()
    categories = page_data.index.tolist()
    with col22:
        radar = go.Scatterpolar(
            r=values + [values[0]],  # To close the radar chart
            theta=categories + [categories[0]],  # To close the radar chart
            fill='toself',
            name=f'Page {page_with_most_technique}',
            # fillcolor='#e1812c'
        )   
        

        fig = go.Figure(data=[radar])

        # Add annotation for the page name under the radar chart
        fig.update_layout(
            annotations=[
                go.layout.Annotation(
                    x=0.5,
                    y=-0.2,
                    text=f"Page: {page_with_most_technique}",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    font=dict(
                        size=14,
                        color="black"
                    )
                )
            ]
        )

        # Display the Go plot
        st.plotly_chart(fig)

    


    
def dataset_overview():
    st.title('Overview of the Dataset')
    st.write('This is a sample of 10 thousand rows from the original dataset.')
    
    st.dataframe(df)

    st.write('''
    The dataset consists of 14 columns, with a mix of numeric and textual data. Here's an overview of the columns:
    ''')
    st.markdown('''
    - **title:** The title of the youtube video.
    - **duration:** Duration of the video in seconds.
    - **link:** The URL link to the youtube video.
    - **channel_name:** The name of the channel.
    - **subscribers:** Number of subscribers to the channel.
    - **category:** The category of the video such as 'News&Politics', 'Entertainment', etc.
    - **upload_date:** The date when the video was uploaded.
    - **views:** Number of views the video has received.
    - **text:** A text transcript of the video (subtitles).
    - **sentiment:** The sentiment portrayed in the text (e.g., positive or negative).
    - **found_candidate:** Name of the senate candidate found in the subtitles of the video.
    - **found_party:** The political party associated with the candidate. (e.g., 'Republican', 'Democrat', etc.)
    - **propaganda:** A numeric indicator (e.g., 0 or 1) showing whether propaganda was detected in the text.
    - **propaganda_techniques:** A list of propaganda techniques identified in the content, if any.
    ''')


def statistics_page():
    st.title('the distribution of propaganda techniques')
    st.write('The bar plot on the left shows the overall percentage of each propaganda technique in the dataset. Here are some key observations:')

    technique_columns = ['propaganda', 'Loaded_Language', 'Name_Calling,Labeling' , 'Exaggeration,Minimisation' , 'Appeal_to_fear-prejudice','Flag-Waving','Doubt', 'Black-and-White_Fallacy',
                    'Causal_Oversimplification']

    technique_percentage = pd.Series(np.array([34,27.62, 17.87, 13.38 ,10.68, 10.44,  2.58, 2.18,  1.69   ]))

    col1, col2 = st.columns(2)

    with col1:
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size': 13})
        plt.barh(technique_columns, technique_percentage, color='skyblue', alpha=0.75)
        plt.xlabel('Percentage of Videos (%)')
        plt.title('Percentage of propaganda techniques in videos')
        plt.gca().invert_yaxis() 
        plt.show()

        st.pyplot()

    with col2:
        st.write('''
        - the overall percentage of propaganda in the dataset is 34%.
        - Loaded Language is the most common propaganda technique, present in approximately 27.62% of the videos.
        - We can see that the top three most used propaganda techniques are all syntactic techniques, which means that they are based on the structure of the text rather than the content. On the other hand, the least used propaganda techniques are all semantic techniques.
        - this suggests that the creators of the videos in the dataset tend to rely more on syntactic techniques by using strong words and labels because they are easier to use and more effective than comming up with arguments.
     ''')
        
    st.write("") 
    st.write("") 
    st.write("") 
    st.write("") 
    st.write("") 
    st.write("") 
    st.title('Propaganda and video duration')
    st.write('''
        The histogram on the right shows the distribution of video durations in the dataset. 
        After analyzing multiple features and seeing how they can relate to the video duration, 
        the only key insight that we could deduce is represented in the two bar plots below, 
        which show the average and median duration of videos for each propaganda technique.
    ''')

    col3, col4 = st.columns(2)

    with col3:
        st.write('''
        
        - We can see in the plot on the left side that there are 3 techniques that have a considerable higher average duration than the others, 
                 and seeing how the average can be affected by outliers, we can confirm whether the deduction is right or wrong by plotting the 
                 median duration as well which is not affected by outliers, and we can see that it leads to the same conclusion which suggests that 
                 the 3 techniques ('Black and white fallacy',' Causal oversimplification' and 'doubt') are used in longer videos than the others.
        - Looking at these 3 propaganda techniques we can see that they are all semantic based techniques which depends on story-telling and arguments 
                 to persuade the audiance. this might be the reason why they are found in much longer videos, contrary the other 5 syntactic techniques that depends on
                 using strong words and labels which doesn't require long videos.
                 ''')

    with col4:
        video_duration = pd.read_csv('video_duration.csv')
        sns.histplot(video_duration, bins=30,legend=False)
        plt.rcParams.update({'font.size': 13})
        plt.title('Distribution of Video Durations')
        plt.xlabel('Duration (minutes)')
        plt.ylabel('Frequency')
        plt.show()
        st.pyplot()

    st.write("") 
    st.write("") 
    st.write("") 
    st.write("") 
    st.write("") 
    st.write("") 

    col5, col6 = st.columns(2)
    with col5:
        plt.figure(figsize=(8, 4))
        plt.rcParams.update({'font.size': 8})
        avg_duration_df = pd.read_csv('avg_duration_df.csv')
        avg_duration_df.plot(kind='bar', legend=False, color='lightcoral', alpha=0.75)
        plt.title('Average Duration by Propaganda Technique')
        plt.ylabel('Average Duration (minutes)')
        techniques = ['Appeal_to_fear-prejudice','Black-and-White_Fallacy','Causal_Oversimplification','Doubt','Exaggeration,Minimisation','Flag-Waving','Loaded_Language','Name_Calling,Labeling']
        plt.xticks(ticks=range(len(techniques)),labels=techniques,rotation=45, ha='right')
        plt.show()
        st.pyplot()
    with col6:
        plt.figure(figsize=(8, 4))
        median_duration_df = pd.read_csv('median_duration_df.csv')
        median_duration_df.plot(kind='bar', legend=False, color='lightcoral', alpha=0.75)
        plt.title('Median Duration by Propaganda Technique')
        plt.xticks(ticks=range(len(techniques)),labels=techniques,rotation=45, ha='right')
        plt.ylabel('Average Duration (minutes)')
        plt.show()
        st.pyplot()

def view_count_propaganda():

    st.title('View Count and Propaganda')
    st.write('''
        The bar plot on the left shows the average views for videos with and without propaganda and based on positive or negative sentiment in the subtitles.
        But before we can plot the average views, we had the idea to normalize them to make the comparison more accurate by deviding each amount of views a video gets by 
             the number of subscribers of the channel, that way the results won't be affected by videos that are posted by big channels which have a lot more views than small ones. 
    ''')

    col1, col2 = st.columns(2)
    with col1:
        normalized_views = pd.read_csv('normalized_views.csv')        
        normalized_views.plot(kind='bar', color=['skyblue', 'salmon'], title='Average Views (Normalized) per sentiment', ylabel='Average Views', alpha=0.75)
        plt.xticks(ticks=[0,1],labels=['no propaganda','propaganda'],rotation=0, ha='center' )
        plt.show()
        st.pyplot()
    with col2:
        st.write('''
            - The first thing we can notice is that videos with propaganda have a higher average views than videos without propaganda, which suggests that propaganda is more effective in attracting views. 
            - Secondly, we can see that videos with negative sentiment have a higher average views than videos with positive sentiment in both propaganda and non-propaganda videos, one possible cause for this
                is that negative sentiment videos tend to be more controversial and attract more attention. plus, social media algorithms are programmed to show more controversial content to attract more views. which was the case when facebook was 
                 once sued for prommoting negative posts and comments more than the positive ones, which might be the case for youtube.
            - finally, we can deduce that using propaganda to positively influence the audiance toward a certain idea doesn't seem to have an impact on views, contrary to the negative one.
        ''')
    
    col3, col4 = st.columns(2)

    with col3:
        st.write('''''')
        st.write('''''')
        st.write('''''')
        st.write('''
        The bar plot on the right shows the average views for each propaganda technique
        - It is apparent that the top 3 propaganda techniques that have a higher average views are all syntactic based techniques, and it can be explained by the fact that
                 strong and explicit wording and messages which is used in syntactic techniques can attract more views.
        - Additionally, We saw in previous analysis that the duration of videos that contain these 3 propaganda techniques is lower than others, which might be another cause
                 for the higher view count, as shorter videos tend to attract more views.
    ''')
    with col4:
        average_views_df = pd.read_csv('average_views_df.csv')
        average_views_df.sort_values('Average Views').plot(kind='bar', x='Propaganda Technique', y='Average Views', color='skyblue', legend=False, alpha=0.75)
        plt.title('Average Views for Each Propaganda Technique')
        plt.ylabel('Average Views')
        plt.xlabel('')
        plt.xticks(rotation=45, ha='right')
        plt.show()
        st.pyplot()

def political_parties_propaganda():

    st.title('Evolution of propaganda over time')
    st.write('Another interesting analysis we did was to see how the percentage of propaganda in the dataset evolved over time.')
    st.write('The plot below shows the percentage of propaganda in the dataset for each month from the first of mai 2023 to the 15th of mai 2024.')
    col3, col4 = st.columns(2)

    with col3:
        st.write('''''')
        st.write('''''')
        st.write('''''')
        st.write('''''')
        st.write('''''')
        st.write('''- Knowing that the United states senate elections is scheduled to be held on november 5, 2024. We can see in the plot that the closer the elections is the more the 
                 percentage of propaganda increases, the percentage of propaganda in april 2024 is 20 times the percentage of propaganda in may 2023. As for mai 2024, the percentage of propaganda is low 
                 because the data was collected on the 14th of mai 2024, thus the videos are only collected for the half of this month ''')
    with col4:
        
        df_grouped = pd.read_csv('propaganda_over_time.csv')
        plt.plot(df_grouped['month_year'].astype(str), df_grouped['propaganda'], marker='o', color='lightcoral',alpha=0.75)

        plt.title('Evolution of Total Propaganda Over Time')
        plt.xlabel('Year-Month')
        plt.ylabel('Total Propaganda percentage')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.show()
        st.pyplot()

    
    
    st.title('''Political parties and propaganda''')
    st.write('''The following graph represents the percentage of each propaganda technique that targets each political party''')


    a = np.array([0.417,0.368,0.448,0.376,0.159,0.147,0.261,0.145])
    propaganda_counts_democrats = pd.Series(a, index=['Appeal_to_fear-prejudice', 'Black-and-White_Fallacy',
        'Causal_Oversimplification', 'Doubt', 'Exaggeration,Minimisation',
        'Flag-Waving', 'Loaded_Language', 'Name_Calling,Labeling'])
    b = np.array([0.102,0.057,0.14,0.16,0.362,0.293,0.495,0.355])
    propaganda_counts_republicans = pd.Series(b, index=['Appeal_to_fear-prejudice', 'Black-and-White_Fallacy',
        'Causal_Oversimplification', 'Doubt', 'Exaggeration,Minimisation',
        'Flag-Waving', 'Loaded_Language', 'Name_Calling,Labeling'])
    c = np.array([0.228,0.169,0.167,0.193,0.243,0.233,0.341,0.289])
    propaganda_counts_others = pd.Series(c, index=['Appeal_to_fear-prejudice', 'Black-and-White_Fallacy',
        'Causal_Oversimplification', 'Doubt', 'Exaggeration,Minimisation',
        'Flag-Waving', 'Loaded_Language', 'Name_Calling,Labeling'])

    import plotly.graph_objects as go

    categories = ['Appeal_to_fear-prejudice', 'Black-and-White_Fallacy',
        'Causal_Oversimplification', 'Doubt', 'Exaggeration,Minimisation',
        'Flag-Waving', 'Loaded_Language', 'Name_Calling,Labeling']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=propaganda_counts_others,
        theta=categories,
        fill='toself',
        name='Others'
    ))
    fig.add_trace(go.Scatterpolar(
        r=propaganda_counts_republicans,
        theta=categories,
        fill='toself',
        name='Republicans'
    ))
    fig.add_trace(go.Scatterpolar(
        r=propaganda_counts_democrats,
        theta=categories,
        fill='toself',
        name='Democrats'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True, 
        range=[0, 0.7]
        )),
    showlegend=True
    )

    st.plotly_chart(fig)

    st.write('''''')
    st.write('''- As we can see, there is a pattern that suggests that the propaganda techniques used against the democrats are more semantic based techniques, such as 'doubt', 'Causal oversimplification', 'Black and white fallacy' 
                and 'Appeal to fear'. while Republicans are targeted with more syntactic based techniques such as 'Loaded language', 'Name calling', 'Flag waving' and 'Exaggeration'. 
    ''')
    st.write('''- It's hard to find a reason for this pattern, but one possible explanation we found is that the most used word in the videos about republicans is trump which is a very controversial figure, and could possibly 
                single-handedly cause the high percentage of syntactic techniques used against republicans. other most used words indicate the same thing for example words like russia, crisis, military and money are
                all frequent words in videos about republicans.
    ''')
    st.write('''- As for democrats, we can find biden as the most frequent word but he is less mentioned in videos almost half as much as trump and less controversial, other frequent words like state, good, news and case are 
                more neutral words.             
             ''')

# Sidebar for dataset selection
st.sidebar.title('Select Dataset')
dataset = st.sidebar.selectbox('Choose a dataset', ['Organic vs Sponsored Propaganda', 'YouTube Videos'])

# Load datasets
if dataset == 'Organic vs Sponsored Propaganda':
    df = pd.read_csv('perfect_data_organic_sponsored.csv')  # Replace with your actual file path
    df['unique'] = df['fb_advertiser_fb_page'].astype(str) + '_' + df['landing_domain'].astype(str)
    df =df.dropna(subset= ['partisanship'])
    
    # Additional buttons for 'Organic vs Sponsored Propaganda'
    st.sidebar.title('Additional Options')
    if st.sidebar.button('Overall Propaganda Presence'):
        st.session_state.page = 'Overall Propaganda Presence'
    # if st.sidebar.button('Frequency of Each Propaganda Technique'):
    #     st.session_state.page = 'Frequency of Each Propaganda Technique'
    if st.sidebar.button('Propaganda Frequency per page'):
        st.session_state.page = 'Propaganda Frequency per page'
    if st.sidebar.button('Partisanship and Propaganda'):
        st.session_state.page = 'Partisanship and Propaganda'
    # if st.sidebar.button('Engagements and Propaganda'):
    #     st.session_state.page = 'Engagements and Propaganda'

elif dataset == 'YouTube Videos':
    df = pd.read_csv('small_data.csv')  # Replace with your actual file path
    st.session_state.page = 'first'
    st.sidebar.title('Additional Options')
    if st.sidebar.button('Dataset overview'):
        st.session_state.page = 'Dataset overview'
    if st.sidebar.button('Propaganda Techniques distribution'):
        st.session_state.page = 'Basic statistics'
    if st.sidebar.button('View count and Propaganda'):
        st.session_state.page = 'View count and Propaganda'
    if st.sidebar.button('Political parties and Propaganda'):
        st.session_state.page = 'Political parties and Propaganda'

# Initialize the page state
if 'page' not in st.session_state:
    st.session_state.page = 'Main'

# Display the appropriate page based on the session state
if st.session_state.page == 'Overall Propaganda Presence':
    overall_propaganda_page()
elif st.session_state.page == 'Partisanship and Propaganda':
    partisanship_propaganda_page()
elif st.session_state.page=="Propaganda Frequency per page":
    propaganda_per_page()
elif st.session_state.page == 'Dataset overview':
    dataset_overview()
elif st.session_state.page == 'Basic statistics':
    statistics_page()
elif st.session_state.page == 'first':
    dataset_overview()
elif st.session_state.page == 'View count and Propaganda':
    view_count_propaganda()
elif st.session_state.page == 'Political parties and Propaganda':
    political_parties_propaganda()
else:
    main_page()
