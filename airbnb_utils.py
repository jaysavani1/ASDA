import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, mannwhitneyu
import folium
from folium.plugins import MarkerCluster
import warnings

warnings.filterwarnings("ignore")

# Load the dataset and preprocess each sheet
def load_and_preprocess_data(file_url):
    excel_data = pd.ExcelFile(file_url)
    all_sheets = []

    for sheet in excel_data.sheet_names:
        df = excel_data.parse(sheet)
        df.drop(df.columns[0], axis=1, inplace=True)
        
        city, day_type = sheet.split('_')
        df['city'] = city.capitalize()
        df['day_type'] = day_type.capitalize()
        
        all_sheets.append(df)

    combined_df = pd.concat(all_sheets, ignore_index=True).reset_index(drop=True)
    return combined_df

# Data cleaning and transformation
def clean_and_transform_data(df):
    # Define columns and types
    int_columns = ['person_capacity', 'multi', 'biz', 'cleanliness_rating', 'guest_satisfaction_overall', 'bedrooms']
    float_columns = ['realSum', 'dist', 'metro_dist', 'attr_index', 'attr_index_norm', 'rest_index', 'rest_index_norm', 'lng', 'lat']
    
    df[int_columns] = df[int_columns].astype('int64')
    df[float_columns] = df[float_columns].astype('float64')

    # Map cities to countries
    city_to_country = {
        'Amsterdam': 'Netherlands', 'Athens': 'Greece', 'Berlin': 'Germany', 'Barcelona': 'Spain',
        'Budapest': 'Hungary', 'Lisbon': 'Portugal', 'London': 'United Kingdom', 'Paris': 'France',
        'Rome': 'Italy', 'Vienna': 'Austria'
    }
    df['country'] = df['city'].map(city_to_country)

    # Rename and clean columns
    df.rename(columns={'realSum': 'price'}, inplace=True)
    df.drop(columns=['room_type'], errors='ignore', inplace=True)
    df['room_shared'] = df['room_shared'].astype(str).str.upper()
    df['room_private'] = df['room_private'].astype(str).str.upper()

    # Define room type based on shared/private columns
    conditions = [
        (df['room_shared'] == 'TRUE'),
        (df['room_private'] == 'TRUE') & (df['room_shared'] == 'FALSE'),
        (df['room_shared'] == 'FALSE') & (df['room_private'] == 'FALSE')
    ]
    choices = ['Shared', 'Private', 'Entire Apt']
    df['room_type'] = np.select(conditions, choices, default='Unknown')
    df.drop(columns=['room_shared', 'room_private'], inplace=True)
    
    df['host_is_superhost'] = df['host_is_superhost'].replace({True: 'Superhost', False: 'Normalhost'})
    return df

import matplotlib.pyplot as plt
import seaborn as sns

# Visualization functions
def plot_listings_per_city(df):
    city_listing_counts = df['city'].value_counts().sort_values()
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=city_listing_counts.index, y=city_listing_counts.values, hue=city_listing_counts.index, palette='viridis', dodge=False, legend=False)
    plt.title("Total Listings per City")
    plt.xlabel("City")
    plt.ylabel("Number of Listings")
    
    # Adding annotations
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

    plt.show()

def plot_listings_by_day_type(df):
    listings_day_type = df.groupby(['city', 'day_type']).size().unstack()
    
    ax = listings_day_type.plot(kind='bar', figsize=(10, 6), colormap='Paired')
    plt.title("Listings by Day Type in Each City")
    plt.xlabel("City")
    plt.ylabel("Listings Count")
    plt.legend(title="Day Type")
    
        # Adding annotations
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=6, color='black', xytext=(0, 5), textcoords='offset points')

    plt.show()

def plot_superhost_proportion(df):
    superhost_counts = df[df['host_is_superhost'] == 'Superhost'].groupby('city').size()
    total_counts = df.groupby('city').size()
    superhost_proportion = (superhost_counts / total_counts * 100).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=superhost_proportion.index, y=superhost_proportion.values, hue=superhost_proportion.index, palette='rocket', legend=False)
    plt.title("Superhost Proportion by City")
    plt.xlabel("City")
    plt.ylabel("Superhost Proportion (%)")
    plt.xticks(rotation=45)
    
    # Adding annotations
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

    plt.show()

def plot_entire_home_listings(df):
    entire_home_counts = df[df['room_type'] == 'Entire Apt']['city'].value_counts().sort_values()
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=entire_home_counts.index, y=entire_home_counts.values, hue=entire_home_counts.index, legend=False, palette='Blues')
    plt.title("Entire Home Listings by City")
    plt.xlabel("City")
    plt.ylabel("Number of Listings")
    
    # Adding annotations
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

    plt.show()

def listings_with_multiple_rooms(df, min_rooms=4):
    large_listings = df[df['bedrooms'] > min_rooms]
    room_counts = large_listings['city'].value_counts()

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=room_counts.index, y=room_counts.values,hue=room_counts.index, legend=False, palette='magma')
    plt.title(f"Listings with More Than {min_rooms} Rooms per City")
    plt.xlabel("City")
    plt.ylabel("Number of Listings")
    plt.xticks(rotation=45)
    
    # Adding annotations
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

    plt.show()


def plot_bar_with_annotation(df, x, y, title, xlabel, ylabel, palette=None):
    plt.figure(figsize=(10, 6))
    
    # If palette is specified, use it for coloring
    if palette:
        # Create a color palette from the specified colormap
        num_colors = len(df)
        color_palette = sns.color_palette(palette, num_colors)  # Create a palette from the colormap
        ax = sns.barplot(data=df, x=x, y=y, palette=color_palette)
    else:
        ax = sns.barplot(data=df, x=x, y=y, color='skyblue')  # Default solid color
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    # Add annotations on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    fontsize=12, color='black', 
                    xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()
    plt.show()


def plot_distribution(data, title, xlabel, color, log_transform=False):
    plt.figure(figsize=(10, 6))
    if log_transform:
        data = np.log1p(data)
        xlabel = f"Log({xlabel} + 1)"
    
    sns.histplot(data, bins=30, kde=True, color=color, stat="density", linewidth=2)
    plt.title(title, fontsize=18, fontweight='bold', color='darkblue')
    plt.xlabel(xlabel, fontsize=14, color='darkred')
    plt.ylabel("Density", fontsize=14, color='darkred')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_box_and_hist(data, title, xlabel, ylabel, hist_color='forestgreen', box_color='darkslategray'):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.histplot(data, ax=ax[0], kde=True, color=hist_color)
    sns.boxplot(data=data, ax=ax[1], color=box_color)
    ax[0].set_title(f"{title} - Histogram")
    ax[1].set_title(f"{title} - Box Plot")
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()

# Function for Section 6: Person Capacity Distribution
def plot_person_capacity_distribution(df):
    """
    This function plots the distribution of person capacity across listings and annotates the most common capacity.
    """
    person_capacity_counts = df['person_capacity'].value_counts().sort_index()
    person_capacity_df = person_capacity_counts.reset_index()
    person_capacity_df.columns = ['person_capacity', 'count']

    # Updated color scheme for section 6
    # Plot with the coolwarm colormap
    plot_bar_with_annotation(person_capacity_df, x='person_capacity', y='count',
                            title="Distribution of Person Capacity Across Listings",
                            xlabel="Person Capacity", ylabel="Number of Listings", palette="coolwarm")


    most_common_capacity = person_capacity_df.loc[person_capacity_df['count'].idxmax()]
    print(f"Conclusion: The most common person capacity is {most_common_capacity['person_capacity']} with {most_common_capacity['count']} listings.")

# Function for Section 7: Price Distribution for Weekdays and Weekends
def plot_price_distribution_weekday_weekend(df):
    """
    This function compares the price distributions between weekdays and weekends.
    """
    weekday_prices = df[df['day_type'] == 'Weekdays']['price']
    weekend_prices = df[df['day_type'] == 'Weekends']['price']

    # Section 7: 2x1 chart for weekday and weekend distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))  # 2 rows, 1 column
    axes[0].hist(weekday_prices, color='lightblue', edgecolor='black', bins=20)
    axes[0].set_title("Weekday Price Distribution", fontsize=16)
    axes[0].set_xlabel("Price ($)", fontsize=14)
    axes[0].set_ylabel("Frequency", fontsize=14)

    axes[1].hist(weekend_prices, color='salmon', edgecolor='black', bins=20)
    axes[1].set_title("Weekend Price Distribution", fontsize=16)
    axes[1].set_xlabel("Price ($)", fontsize=14)
    axes[1].set_ylabel("Frequency", fontsize=14)

    plt.tight_layout()  # Adjust subplots to avoid overlap
    plt.show()

    # Section 7: 2x1 chart for log-transformed weekday and weekend distributions
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))  # 2 rows, 1 column
    axes[0].hist(np.log1p(weekday_prices), color='teal', edgecolor='black', bins=20)
    axes[0].set_title("Log-Transformed Weekday Price Distribution", fontsize=16)
    axes[0].set_xlabel("Log(Price + 1)", fontsize=14)
    axes[0].set_ylabel("Frequency", fontsize=14)

    axes[1].hist(np.log1p(weekend_prices), color='darkorange', edgecolor='black', bins=20)
    axes[1].set_title("Log-Transformed Weekend Price Distribution", fontsize=16)
    axes[1].set_xlabel("Log(Price + 1)", fontsize=14)
    axes[1].set_ylabel("Frequency", fontsize=14)

    plt.tight_layout()  # Adjust subplots to avoid overlap
    plt.show()

# Function for Section 8: Guest Satisfaction Ratings Distribution
def plot_guest_satisfaction_distribution(df):
    """
    This function plots the distribution of guest satisfaction ratings.
    """
    guest_satisfaction_ratings = df['guest_satisfaction_overall']
    # Remove barwidth argument and the reference to it
    plot_box_and_hist(guest_satisfaction_ratings, 
                    "Guest Satisfaction Ratings", 
                    xlabel="Guest Satisfaction Rating", 
                    ylabel="Frequency", 
                    hist_color='cornflowerblue', 
                    box_color='lightcoral')

# Function for Section 9: Cleanliness Ratings Distribution
def plot_cleanliness_distribution(df):
    """
    This function plots the distribution of cleanliness ratings.
    """
    cleanliness_ratings = df['cleanliness_rating']
    plot_box_and_hist(cleanliness_ratings, "Cleanliness Ratings",
                      xlabel="Cleanliness Rating", ylabel="Frequency",
                      hist_color='teal', box_color='lightcoral')


# Main Analysis Function
def analyze_data(df):
    # Section 1: Total Listings by City
    print("""
    Section 1: Total Listings by City
    ---------------------------------
    This plot provides an overview of the total Airbnb listings available in each city. Using a horizontal bar chart, it highlights 
    cities with the highest and lowest supply, giving us a snapshot of Airbnb availability across locations.
    """)
    plot_listings_per_city(df)

    # Section 2: Listings by Day Type (Weekday vs. Weekend)
    print("""
    Section 2: Listings by Day Type (Weekday vs. Weekend)
    -----------------------------------------------------
    This analysis breaks down the number of Airbnb listings available on weekdays and weekends for each city. It’s especially 
    insightful for understanding changes in listing availability, as certain cities may have more or fewer listings depending on 
    the day of the week.
    """)
    plot_listings_by_day_type(df)

    # Section 3: Superhost Proportion by City
    print("""
    Section 3: Superhost Proportion by City
    ---------------------------------------
    This function examines the proportion of superhost listings in each city, revealing where superhosts are most prevalent. 
    Superhosts are often more reliable, so this plot provides insight into host quality across locations.
    """)
    plot_superhost_proportion(df)

    # Section 4: Entire Home/Apt Listings by City
    print("""
    Section 4: Entire Home/Apt Listings by City
    -------------------------------------------
    This analysis focuses on the number of listings categorized as 'Entire Home/Apt' in each city. By visualizing full-property 
    rentals, it offers insights into city-level preferences and inventory, which are especially relevant for travelers looking for 
    private spaces.
    """)
    plot_entire_home_listings(df)

    # Section 5: Multi-Room Listings by City
    print("""
    Section 5: Multi-Room Listings by City
    --------------------------------------
    This function highlights cities with a notable number of multi-room listings, catering to travelers in need of larger 
    accommodations. Such insights can be valuable for family and group travel planning.
    """)
    listings_with_multiple_rooms(df)

    # Section 6: Person Capacity Distribution
    print("""
    Section 6: Person Capacity Distribution
    --------------------------------------
    This plot displays the distribution of person capacities across all listings. Understanding the most common person capacity 
    helps to highlight the typical accommodation sizes and potential preferences of guests.
    """)
    plot_person_capacity_distribution(df)

    # Section 7: Price Distribution for Weekdays and Weekends
    print("""
    Section 7: Price Distribution for Weekdays and Weekends
    ------------------------------------------------------
    Here, we compare the price distributions for Airbnb listings on weekdays and weekends. This can help us understand if 
    prices fluctuate based on the day of the week, which might indicate seasonal trends or pricing strategies.
    """)
    plot_price_distribution_weekday_weekend(df)

    # Section 8: Guest Satisfaction Ratings Distribution
    print("""
    Section 8: Guest Satisfaction Ratings Distribution
    ---------------------------------------------------
    In this section, we explore the distribution of guest satisfaction ratings. Understanding how guests feel about their 
    stays can provide valuable insights into the quality of the listings and the overall guest experience.
    """)
    plot_guest_satisfaction_distribution(df)

    # Section 9: Cleanliness Ratings Distribution
    print("""
    Section 9: Cleanliness Ratings Distribution
    --------------------------------------------
    Cleanliness is a critical factor in guest satisfaction. This plot shows the distribution of cleanliness ratings across 
    all listings, giving us an idea of how well hosts are maintaining their properties.
    """)
    plot_cleanliness_distribution(df)
    
def compare_groups(group1_name, group2_name, data, column_to_compare, group_column_name):

    # Set the group column
    data['group'] = data[group_column_name]
    # Filter data for each group
    group1_data = data[data['group'] == group1_name]
    group2_data = data[data['group'] == group2_name]

    # Descriptive statistics
    mean_group1 = round(group1_data[column_to_compare].mean(), 3)
    mean_group2 = round(group2_data[column_to_compare].mean(), 3)
    median_group1 = round(group1_data[column_to_compare].median(), 3)
    median_group2 = round(group2_data[column_to_compare].median(), 3)

    # Normality test (Shapiro-Wilk)
    sw_stat_group1, sw_pval_group1 = shapiro(group1_data[column_to_compare])
    sw_stat_group2, sw_pval_group2 = shapiro(group2_data[column_to_compare])

    # Choose statistical test based on normality
    if sw_pval_group1 > 0.05 and sw_pval_group2 > 0.05:
        # Both normal distributions - use independent t-test
        t_stat, p_value = ttest_ind(group1_data[column_to_compare], group2_data[column_to_compare])
        is_norm_dist = True
    else:
        # Non-normal distributions - use Mann-Whitney U test
        mwu_stat, mwu_p_value = mannwhitneyu(group1_data[column_to_compare], group2_data[column_to_compare])
        is_norm_dist = False

    # Conclusion and higher group determination
    conclusion = "Significant difference" if (p_value < 0.05 if is_norm_dist else mwu_p_value < 0.05) else "No significant difference"
    higher_group = group1_name if mean_group1 > mean_group2 else group2_name

    # Create results DataFrame
    results_df = pd.DataFrame({
        "Group1": group1_name,
        "Group2": group2_name,
        "Mean_G1": mean_group1,
        "Mean_G2": mean_group2,
        "Median_G1": median_group1,
        "Median_G2": median_group2,
        "SW_G1_Stats": sw_stat_group1,
        "SW_G1_Pval": sw_pval_group1,
        "SW_G2_Stats": sw_stat_group2,
        "SW_G2_Pval": sw_pval_group2,
        "Is_Norm_Dist": is_norm_dist,
        "MWU_Stat": mwu_stat if not is_norm_dist else None,
        "MWU_Pval": mwu_p_value if not is_norm_dist else None,
        "Conclusion": conclusion,
        "Result": f"{higher_group} has higher values for {column_to_compare}"
    }, index=[0])
    
    return results_df

def compare_groups_to_dataframe(df):
    res_df = pd.concat([
        compare_groups('Athens', 'Budapest', df, 'price', 'city'),
        compare_groups('Amsterdam', 'Paris', df, 'price', 'city'),
        compare_groups('Lisbon', 'London', df, 'price', 'city'),
        compare_groups('Weekdays', 'Weekends', df, 'price', 'day_type'),
        compare_groups('Superhost', 'Normalhost', df, 'price', 'host_is_superhost'),
        compare_groups('Superhost', 'Normalhost', df, 'dist', 'host_is_superhost'),
        compare_groups('Superhost', 'Normalhost', df, 'metro_dist', 'host_is_superhost'),
        compare_groups('Superhost', 'Normalhost', df, 'cleanliness_rating', 'host_is_superhost'),
        compare_groups('Entire Apt', 'Private', df, 'price', 'room_type')
    ])

    return res_df
    
def compare_room_type_by_superhost_status(df, room_type_column, column_to_analyze, palette=['#1f77b4', '#ff7f0e']):
    """
    Compare the distribution of room types based on superhost status and perform a Chi-squared test.
    
    Parameters:
    - df: DataFrame containing the dataset.
    - room_type_column: The column for room types (e.g., room_type).
    - column_to_analyze: The column to analyze (e.g., price, cleanliness_rating).
    - palette: Custom colors for the bar plot (default is a list with blue and orange).
    
    Returns:
    - A pandas DataFrame with Chi-squared test results and conclusion in a vertical format.
    """
    # Group data by room type and superhost status
    room_type_superhost_count = df.groupby([room_type_column, 'host_is_superhost']).size().reset_index(name='count')

    # Visualize the distribution of room types by superhost status with custom colors
    plt.figure(figsize=(10, 6))
    sns.barplot(data=room_type_superhost_count, x=room_type_column, y='count', hue='host_is_superhost', palette=palette)
    plt.title(f"Room Type Distribution by Superhost Status", fontsize=16, fontweight='bold')
    plt.xlabel("Room Type", fontsize=14)
    plt.ylabel(f"Number of Listings by {column_to_analyze}", fontsize=14)
    plt.xticks(rotation=45)
    plt.show()

    # Perform Chi-squared test
    contingency_table = pd.crosstab(df[room_type_column], df['host_is_superhost'])
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

    # Round the Chi-squared statistic and degrees of freedom to 2 decimal places (exclude p-value)
    chi2_stat = round(chi2_stat, 2)
    dof = round(dof, 2)
    
    # For expected frequencies, round each element in the numpy array
    expected_rounded = [[round(val, 2) for val in row] for row in expected]

    # Create a vertical result DataFrame with each statistic in its own row
    comparison_results = pd.DataFrame({
        "Metric": [
            "Chi2 Statistic",
            "P-Value",
            "Degrees of Freedom",
            "Expected Frequencies",
            "Conclusion"
        ],
        "Value": [
            chi2_stat,
            p_value,  # Do not round p-value
            dof,
            expected_rounded,
            "Significant relationship" if p_value < 0.05 else "No significant relationship"
        ]
    })

    return comparison_results
