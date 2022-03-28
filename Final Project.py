'''
Jun Ahn
CSE 163
Takes in the covid data.csv dataset and naturalearth_lowres
to plot various geospatial data and identify relationships between
covid cases and various factors.
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import scipy as sp
sns.set()
sns.set_style(style='white')


def covid_data_recent_date(covid_data):
    '''
    takes in a covid dataset and returns a new df
    that only has a most recent date data
    '''
    # convert the date column in to a datetime column
    covid_data['date'] = pd.to_datetime(covid_data['date'])
    most_recent_date = covid_data['date'].max()

    # filtering the dataset to only have the most recent date
    recent_date_covid_df = covid_data[covid_data['date'] == most_recent_date]
    return recent_date_covid_df


def world_total_covid_vis(covid_data, world_geo):
    '''
    takes in two datasets: geo map and covid data to
    plot geomap and bar chart of the countries by covid status
    '''
    # filtering the dataset to only have the most recent date
    recent_date_covid_df = covid_data_recent_date(covid_data)

    # merge two datasets
    merged_df = world_geo.merge(recent_date_covid_df,
                                left_on='iso_a3', right_on='iso_code')

    # find top 5 countries with the highest total covid cases
    top_5_total_cases = merged_df.sort_values('total_cases',
                                              ascending=False).iloc[0:5]
    top_5_hotspots = merged_df.sort_values('total_cases_per_million',
                                           ascending=False).iloc[0:5]

    # plot world Total Covid case & Covid Hotspot
    fig, [ax1, ax2] = plt.subplots(nrows=2)
    merged_df.plot(column='total_cases', legend=True,
                   cmap='OrRd', linewidth=0.3, edgecolor='0.5', ax=ax1)
    merged_df.plot(column='total_cases_per_million', legend=True,
                   cmap='OrRd', linewidth=0.3, edgecolor='0.5', ax=ax2)
    ax1.set_title('Total Covid Case Map')
    ax2.set_title('Covid Hot Spot Map (cases per 1,000,000)')
    plt.savefig('Covid Case Map', dpi=300, bbox_inche='tight')

    # plotting bar chart of top 10
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    sns.barplot(x='total_cases', y='name',
                data=top_5_total_cases, orient='h', ax=ax1)
    sns.barplot(x='total_cases_per_million', y='name',
                data=top_5_hotspots, orient='h', ax=ax2)
    ax1.set_title('5 Countries With the Highest Total Covid Cases')
    ax2.set_title('5 Countries with the Highest \
                  Total Covid Cases per 1,000,000')
    ax1.set_xlabel('Total Covid Cases')
    ax2.set_xlabel('Total Covid Cases per 1,000,000')
    ax1.set_ylabel('Country')
    ax2.set_ylabel('Country')
    plt.tight_layout()
    plt.savefig('Total case bar chart', dpi=300, bbox_inche='tight')


def world_vacc_vis(covid_data, world_geo):
    '''
    takes in two datasets: geo map and covid data to
    plot geomap and bar chart of the countries by covid status
    '''

    # prepping world map
    country_vacc = covid_data.\
        groupby('iso_code')['total_vaccinations_per_hundred'].max()
    merged_df = world_geo.merge(country_vacc,
                                left_on='iso_a3', right_on='iso_code')

    # find top 10 countries with the highest vaccination per 100
    # the reason to use per 100 is becuase it is an accurate
    # measure among countries with different populations.
    top_10_df = merged_df.sort_values('total_vaccinations_per_hundred',
                                      ascending=False).iloc[0:10]

    # plot the data
    merged_df.plot(column='total_vaccinations_per_hundred', legend=True,
                   cmap='OrRd', linewidth=0.3, edgecolor='0.5')
    plt.title('World Total Vaccination Map (per hundred)')
    plt.savefig('Total Vaccination Map', dpi=300, bbox_inche='tight')

    # plotting the bar chart of top 10
    sns.catplot(x='name', y='total_vaccinations_per_hundred',
                kind='bar', data=top_10_df, height=10)
    plt.title('Top 10 Countries with the Highest Vaccination per 100')
    plt.xticks(rotation=-45)
    plt.xlabel('Country')
    plt.ylabel('Total Vaccination per 100')
    plt.savefig('Total Vaccination per 100 Bar Chart',
                dpi=300, bbox_inches='tight')


def vacc_vs_cases(covid_data):
    '''
    graphs total_vaccinations vs toatl_case to see
    if there are any relationship between two variables
    '''
    # exclude data where total_vaccinations is null
    # vaccination started year after the pandemic started thus lot of the rows
    # are missing vaccination data which needs to be excluded
    filtered_df = covid_data[covid_data['total_vaccinations'].notna()]
    filtered_df = filtered_df[filtered_df['total_cases'].notna()]
    total_case_max = filtered_df['total_cases'].max()

    # correlation coefficient and p-value
    r, p = sp.stats.pearsonr(filtered_df['total_cases'],
                             filtered_df['total_vaccinations'])

    # plot the data
    plot = sns.regplot(x='total_vaccinations', y='total_cases',
                       data=filtered_df, scatter_kws={'s': 0.1})
    plot.text(10, total_case_max, 'r={:.2f}, p={:.2g}'.format(r, p))
    plt.title('Total Vaccination vs Total Covid')
    plt.xlabel('Total Vaccination')
    plt.ylabel('Total Covid Cases')
    plt.savefig('Total Vaccination vs Total Covid Cases',
                dpi=300, bbox_inches='tight')


def correlation_analysis(covid_data):
    '''
    graphs linear regression model to measure the correlation
    between covid cases with various metrics.
    The metrics measured are:
        population
        population_density
        gdp_per_capita
        handwashing_facilities
        extreme_poverty
        human_development_index
    '''
    covid_recent_date_df = covid_data_recent_date(covid_data)
    covid_recent_date_df = covid_recent_date_df[covid_recent_date_df['total_cases_per_million'].notna() &
         covid_recent_date_df['population'].notna() &
         covid_recent_date_df['population_density'].notna() &
         covid_recent_date_df['gdp_per_capita'].notna() &
         covid_recent_date_df['handwashing_facilities'].notna() &
         covid_recent_date_df['extreme_poverty'].notna() &
         covid_recent_date_df['human_development_index'].notna()]

    fig, [[ax1, ax2], [ax3, ax4], [ax5, ax6]] = plt.subplots(nrows=3, ncols=2)

    # correlation coefficient and p-value
    r_1, p_1 = sp.stats.pearsonr(covid_recent_date_df
                                 ['total_cases_per_million'],
                                 covid_recent_date_df['population'])
    r_2, p_2 = sp.stats.pearsonr(covid_recent_date_df
                                 ['total_cases_per_million'],
                                 covid_recent_date_df['population_density'])
    r_3, p_3 = sp.stats.pearsonr(covid_recent_date_df
                                 ['total_cases_per_million'],
                                 covid_recent_date_df['gdp_per_capita'])
    r_4, p_4 = sp.stats.pearsonr(covid_recent_date_df
                                 ['total_cases_per_million'],
                                 covid_recent_date_df['handwashing_facilities'])
    r_5, p_5 = sp.stats.pearsonr(covid_recent_date_df
                                 ['total_cases_per_million'],
                                 covid_recent_date_df['extreme_poverty'])
    r_6, p_6 = sp.stats.pearsonr(covid_recent_date_df
                                 ['total_cases_per_million'],
                                 covid_recent_date_df['human_development_index'])

    y_var = 'total_cases_per_million'

    # plotting the data
    sns.regplot(x='population', y=y_var,
                data=covid_recent_date_df, ax=ax1)
    ax1.text(0, 300000, 'r={:.2f}, p={:.2g}'.format(r_1, p_1))
    ax1.set_xlabel('population')
    ax1.set_ylabel('')

    sns.regplot(x='population_density', y=y_var,
                data=covid_recent_date_df, ax=ax2)
    ax2.text(0, 300000, 'r={:.2f}, p={:.2g}'.format(r_2, p_2))
    ax2.set_xlabel('population density')
    ax2.set_ylabel('')

    sns.regplot(x='gdp_per_capita', y=y_var,
                data=covid_recent_date_df, ax=ax3)
    ax3.text(0, 300000, 'r={:.2f}, p={:.2g}'.format(r_3, p_3))
    ax3.set_xlabel('gdp per capita')
    ax3.set_ylabel('total cases per million')

    sns.regplot(x='handwashing_facilities', y=y_var,
                data=covid_recent_date_df, ax=ax4)
    ax4.text(0, 300000, 'r={:.2f}, p={:.2g}'.format(r_4, p_4))
    ax4.set_xlabel('handwashing facilities')
    ax4.set_ylabel('total cases per million')

    sns.regplot(x='extreme_poverty', y=y_var,
                data=covid_recent_date_df, ax=ax5)
    ax5.text(0, 300000, 'r={:.2f}, p={:.2g}'.format(r_5, p_5))
    ax5.set_xlabel('extreme poverty')
    ax5.set_ylabel('')

    sns.regplot(x='human_development_index', y=y_var,
                data=covid_recent_date_df, ax=ax6)
    ax6.text(0.4, 300000, 'r={:.2f}, p={:.2g}'.format(r_6, p_6))
    ax6.set_xlabel('human development index')
    ax6.set_ylabel('')
    plt.tight_layout()
    plt.savefig('Correlation Analysis', dpi=300, bbox_inche='tight')


def vacc_analysis(vacc_data):
    '''
    takes in a vaccination dataset and graphs the mortality rate
    of vaccinated population and unvaccinated population
    '''
    vacc_data = vacc_data[vacc_data['outcome'] == 'death']

    vacc_group_df = vacc_data.groupby('age_group')\
        .agg(Vaccinated=('crude_vax_ir', 'mean'),
             Unvaccinated=('crude_unvax_ir', 'mean'))
    vacc_group_df.plot(kind='bar',
                       title='Vaccinated vs Unvaccinated (per 100,000)')
    plt.tight_layout()
    plt.xlabel('Age Group')
    plt.ylabel('Death Cases')
    plt.savefig('Vaccination analysis', dpi=300, bbox_inche='tight')


def main():
    # loading in data
    covid_data = pd.read_csv('covid data.csv')
    vaccination_data = pd.read_csv('vaccination data.csv')
    world_geo = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # functions
    world_total_covid_vis(covid_data, world_geo)
    world_vacc_vis(covid_data, world_geo)
    vacc_vs_cases(covid_data)
    correlation_analysis(covid_data)
    vacc_analysis(vaccination_data)


if __name__ == '__main__':
    main()
