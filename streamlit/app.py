#### To run this app on your machine locally, type:
#### 'streamlit run streamlit/app.py' in the terminal

### importing libraries
import pandas as pd
import streamlit as st
from pycaret.regression import *
from pycaret.regression import  load_model
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import matplotlib as plt
import pickle
from IPython.display import display, HTML

from scipy.special import inv_boxcox
from scipy.stats import boxcox


### Importing Data
@st.cache_data()
def load_data(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

df_engineered = load_data('pickles/df_engineered.pkl')
df_user = load_data('pickles/df_user.pkl')
df_cleaned = load_data('pickles/df_cleaned.pkl')
df_streamlit = load_data('pickles/df_streamlit.pkl')
df_plot_melted = load_data('pickles/df_plot_melted.pkl')
predictions = load_data('pickles/predictions_user.pkl')
combined_metrics = load_data('pickles/combined_metrics.pkl')
importances = load_data('pickles/importances.pkl')
feature_names = load_data('pickles/feature_names.pkl')
lambdas_dict = load_data('pickles/lambdas_dict.pkl')


### Importing Model
@st.cache_resource()
def get_model():
    return load_model('models/finalized_user_model')

model = get_model()


# Custom HTML and CSS for the buttons
st.markdown("""
<style>
    div.stButton > button:first-child {
        background-color: #00cc00; /* Green */
        color: white;
        font-size: 30px;
        height: 3em;
        width: 50%;
        border-radius: 10px 10px 10px 10px;
    }
</style>""", unsafe_allow_html=True)


### Start of streamlit app
menu = st.sidebar.radio(
    label='Choose a Page',
    options=[
        "Introduction", 
        "The Data & Model", 
        "Evaluating the Model", 
        ":sparkles: **:rainbow[Project Cost Calculator]** :sparkles:"
    ],
    help='If you are looking for the calculator, use the Cost Calculator button below'
)



if menu == 'Introduction':
    st.sidebar.header(" ")
    st.sidebar.header(" ")
    st.sidebar.header(" ")
    st.sidebar.header(" ")
    st.sidebar.write(":warning: Click on the Project Cost Calculator to use the model yourself")


    with open('pickles/df_engineered.pkl', 'rb') as f:
        df_engineered = pickle.load(f)
    st.title('Transit Cost Estimator')
    st.write('__________')

    st.subheader('Project Introduction')
    st.write('''
    Efficient public transit systems play a pivotal role in enhancing mobility, reducing carbon footprints, and fostering sustainable city growth. In many American cities, 
    there is an ongoing and concerted effort to improve multi-modal transit and build better transit systems that supplement, or replace, the predominantly car-oriented infrastructure. 
    Often these transit system improvements are subject to scrutiny, as urban rail projects require an extensive up-front investment of public money.  
    Within many transit agencies, financial constraints exist, and officials are often hesitant to allocate significant public funds for long-term projects. This hesitancy is intensified by the potential for unexpected expenses that could jeopardize an entire project. 
    There's a pressing need for a transparent tool that provides accurate cost assessments for urban rail projects that doesn't rely on the participation of government officials.
    
    By equipping our community members, local officials, and advocates with realistic cost projections tailored to individual communities, we can optimize the allocation of public funds and bridge the gap between political action and community need.
    ''')
    st.write('__________')

    st.subheader('Project Purpose')
    st.write(''' 
    The objective of this analysis was to analyze existing data pertaining to the construction costs of train lines and create a predictive model that can estimate the total project cost. The analysis generated two primary outputs:

    1. A specialized model intended for professionals, such as engineers, familiar with particular locales.
    2. A general-use model, designed for anyone who is curious about the costs of expanding transit access.
    
    The purpose of the general use model is to democratize transit cost analysis, allowing residents to understand and vouch for their community's needs without awaiting official estimates.
    ''')
    st.write('__________')
    st.subheader('Improvements Made in V2')
    st.write('''     
        In Fall of 2023, I submitted the first iteration of this app to several relevant communities. I receieved some feedback that warranted revisiting the app and improving some areas of the model that I deemed to be inadequate. Below are the major changes that were made between v1 and v2:
        - Additional Data
            - Between v1 and v2, I added an additional 800 datapoints. The v1 user model was trained on about ~900 data points, while the v2 is trained on ~1700 data points.
            - With the new model, the accuracy improved significantly from an MAE of ~500M USD in v1 to ~430M USD in v2.
        - Train Type Feature
            - The most notable change in the feature set was the inclusion of a 'train_type' feature that indicates what type of transit system was built.
            - v1 was inadequate in predicting the cost to build trams as it counted a subway station with the same weight as a tram station.
            - The tedious process of populating the 'train_type' feature improved the model's tram cost prediction accuracy significantly.
        - UI Improvements
            - Several users indicated that the intended flow the app wasn't apparent and I spent some time making the interface more intentional.
            - I added a currency and length unit conversion feature in the app, to accomodate all users.
        ''')

    st.write('__________')
    st.subheader('Conclusion & Recommendations')
    st.caption('Conclusion')
    st.write('''     
    Both models created in this analysis are sufficiently accurate and easily understandable, which fulfills the goals set above.
    The accuracy achieved by the user model, with a mean absolute error of 430.35M USD and an R-squared of .88, is sufficient for the purposes of creating a user focused model that can estimate the potential cost for a project in a given area. 
    
    Additionally, both models generalize well to new data and didn't show a tendency to overfit as was expected to occur due to the size of the dataset. 
    This model, while accurate, would improve given additional data and would need further analysis before it should be deployed in a production environment that influences financial decisions.
    ''')

    st.caption('Recommendations')
    st.write('''     
    While I'm content with the model's performance, there are several ways to improve the reliability of the model which would likely reduce the variance of the output and the accuracy of the model on unseen data:

    1) Dataset size
        - The overall size of the dataset is still small, relative to what is needed to produce a (more) meaningful model that can reliably generate accurate predictions. If this model were to be used in a production setting, it would need to be trained on significantly more data.

    2) Dataset accuracy
        - In many cases, the accuracy of the underlying data could be improved. This is less an issue of data collection, as often there were few sources to verify the details of a project.
        - The accuracy of the data is also largely a product of the types of data reported, which adds in an additional element of error. (i.e- including rolling stock, bike path improvements, street renovatations were likely often included in the final costs.)
        - There exists a better solution to handle data verification.
        
    3) Expansion of existing features
        - In my attempts to engineer features from the existing data, it's possible that there are other combinations of engineered features that could have yielded better insight for the model. My attempts with feature engineering were not exhaustive.
        - I wasn't able to find a proxy that correlated strongly enough with land aquisition prices. I believe this is an important component that would help the model generalize better with fewer features.
        - Level of Service is also not represented in the dataset, as I wasn't able to reliably determine how often each train would run. This is an important aspect as it dictates many downstream design decisions that cost resources to implement. 

    4) Future Inflation Rates
        - One aspect of the data that is lacking is that the inflation rate for a future project is capped at 1. For instance, a project that takes place today will have an inflation rate of 1 and a project that takes place a year from now is also given an inflation rate of 1, even though there will be some inflation over the course of that year.
        - This however introduces additional potential error as inflation rates are variable and it's difficult to predict the future economic situation.
        - My attempts to resolve this issue resulted in a much less accurate model. If I were to revisit this analysis, I would find a way to incorportate inflation change over the course of a project if it is expected to end in the near future and the funds had been set aside prior to 2021.
    
    5) Financing Costs
        - Many countries take on loans from other countries to fund these projects. With these loans, the country is likely spending more on the project than the initial estimate states. A complete analysis would incorporate this aspect, either by directly providing the loan terms or by using a binary 'financed?' distinction.


    Overall the model is sufficient and functions best when the overall length of the project is less than 15km. With more data, this model would perform better on fringe cases and provide a more robust estimate for common projects in all locales.
    ''')
    
    st.write('__________')
    st.subheader('Acknowledgments')
    st.write('''
    This app uses a machine learning model that was trained (largely) on the data provided by The Transit Project. 
    To learn more about the data provided by the Transit Project, please go to their [webpage](https://transitcosts.com/about/) or view my [github repo](https://github.com/smileshey/TransitCostEstimator) to learn more about how the model was trained.

    In addition to The Transit Project, I used several resources to construct this analysis:

    1. [The Transit Project](https://transitcosts.com/about/)
    2. [Pedestrian Observations](https://pedestrianobservations.com/)
    3. [PyCaret](https://pycaret.gitbook.io/docs/)
    4. [Urban Rail](https://www.urbanrail.net/)
    5. [Soil Grids](https://www.isric.org/explore/soilgrids)
    6. [Plotly](https://plotly.com/python/)
    7. [OpenMeteo](https://open-meteo.com/)
    8. [Streamlit](https://streamlit.io/)
    ''')
    st.write('__________')
    st.subheader('Who am I?')
    st.write('''
    I'm a civil engineer (P.E.) that has an interest in data. You can find more of my work on [my webpage](http://ryanvirg.in/) and get in touch with me there.

    I've spent my career designing and managing projects for a large public works agency in California, which often oversaw projects from the cradle to the grave. My experience in the industry lends itself to this sort of analysis as I'm familiar with the design, planning, and construction of large-scale, long-term public goods. 
    ''')
    st.write('__________')

    st.subheader('What\'s Next?')
    st.write('''
    There are 4 sections in this app:
    1. Introduction (You are here)
    2. The Data & Model
    3. Evaluating the Model
    4. Modelling

    The next two sections describe, in some brevity, the reasoning behind each step of the model development. These sections are important for understanding what the model is good at and what could be improved.

    The last section is your opportunity to use the model to create predictions for an existing or a mythological transit network you decide to dream up.

    Thank you for reading!
    ''')

elif menu == 'The Data & Model':
    st.title('The Data & Model')
    st.write('_________')
    st.write('This app uses a machine learning model that was trained on the data provided by The Transit Project. To learn more about the data, please go to their [webpage](https://transitcosts.com/about/).')
    st.subheader('Why are Transit Projects so Difficult to price?')
    st.write('''
    Let's imagine you're a baker that specializes in cakes. You're accustomed to baking and selling cakes and so you generally know what each component of the cake will cost. If the cake is X layers, you know that flour, salt, eggs, and butter is going to cost Y. But one day, someone comes in and asks you to bake a cake that's significantly larger than all your other cakes. 
    So large that you don't even know if it would fit in your oven or maybe not even in your bakery.

    How do you figure out how much to charge for that cake? 
    
    If you would generally use 2 eggs per cake and this goliath of a cake would require 2,000 eggs, do you charge 1000x as much?
    and then you might realize that your nearby store doesn't even sell eggs in that quantity and that you'd probably need to buy a whole new bakery just to facilitate this order.

    Then, what if, the city gets wind of this cake and requests that you apply for a permit to bake cakes this large? So you go and fill out the requisite forms, but part of the permitting process requires you to hold public meetings for this cake, requiring you to take into account the needs of everyone within a mile radius of your bakery. Each of these individuals may levy slightly different requests and add additional layers of complexity which restrict how you can bake and deliver the cake. 
    
    Before you've even started baking the cake, you've had to hire a bevy of specialized teams that address each element of concern and you still haven't asked the client what color of frosting they'd like.
    
    The analogy eventually breaks down, but I think the point has been made.
    
    When your local transit agency proproses a new train in your area, there's more to it than going to Home Depot and buying the necessary supplies to build the train. Each transit agency around the world has different constraints that factor into the total cost of a transit project. Different countries have different expertise, different access to materials, different ideologies, and different existing regulations.
    It's in these details that we realize why it's so difficult to correctly estimate the cost of large scale transit projects.

    Large scale public works projects that affect the lives of millions are not just about laying tracks and installing signals (which is an aspect of construction that we are fairly good at estimating). They encompass an array of logistical, social, economic, and political considerations (which we are not good at estimating). From ensuring minimal disruption to existing infrastructure and communities to addressing the concerns of multiple stakeholders, the intricacies involved in these endeavors go beyond mere construction.
    It's a dance of diplomacy, engineering, management, and foresight, where each step taken is carefully weighed for its long-term impact on the community, environment, and economy. 
    
    and it's also a dance in which each involved party has a slightly different vision for how the dance should go.

    The purpose of this analysis is to create a model that can, given some information about the project iteself, account for the nuances in country, region, climate, and ideology to provide a relatively accurate estimate of the cost of a transit project. To achieve this goal, we first need a lot of data.
    ''')
    st.write('____________')
    st.subheader('The Data')

    df_engineered

    fig = px.scatter_geo(df_engineered, 
                        lat='lat', 
                        lon='lng', 
                        color="country",
                        hover_name="country",
                        hover_data=['city','length'],
                        projection="natural earth")

    fig.update_geos(
        resolution=110,
        showcoastlines=True, coastlinecolor="Purple",
        showland=True, landcolor="lightgreen",
        showocean=True, oceancolor="LightBlue",
        showlakes=False, lakecolor="Blue",
        showrivers=False, rivercolor="Blue"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        geo_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig,use_container_width=True)

    st.write('''
    On the map above, there are a thousand different projects within 54 different countries spanning from 1965 to 2026. Each project is physically unique and required a different level of engineering, diplomacy, coordination, and foresight to make happen. 
    
    While each project is unique, there are enough similarities that we can begin to identify patterns, discern common challenges, and establish benchmarks for cost and execution in varying contexts. These collective similarities are what we will call a 'dataset'.
    ''')
    st.write('_________')
    st.subheader('The Dataset')
    st.write('''
    The dataset, orginally compiled by [The Transit Project](https://transitcosts.com/about/), combines these collective similarities into a structured table that looks like this: 
    
    ''')
    df_cleaned['start_year'] = df_cleaned['start_year'].astype('str')
    df_cleaned['end_year'] = df_cleaned['end_year'].astype('str')
    df_cleaned['start_year'] = df_cleaned['start_year'].str.replace(',', '', regex=False)
    df_cleaned['end_year'] = df_cleaned['end_year'].str.replace(',', '', regex=False)

    st.dataframe(df_cleaned)
    st.write('''
    The above dataset summarizes the components of a train line on each row. It tells us where and when the project started, when it ended, how long it is and whether the track is above or below us. 
    
    It also normalizes the cost of a project across time and location. By adjusting for [inflation](https://www.imf.org/en/Publications/fandd/issues/Series/Back-to-Basics/Inflation#:~:text=Inflation%20is%20the%20rate%20of,of%20living%20in%20a%20country.) and by using a [PPP rate conversion](https://data.oecd.org/conversion/purchasing-power-parities-ppp.htm), 
    we can take a project that was completed in Hungary in 1985 and know how much that project would cost for us to build today in the USA.
    
    From this, we can begin to form an idea for how much a project will cost. By normalizing each project cost into 2021 United States Dollars (USD) we can then calculate a cost per km (USD/km).
    Now it would be reasonable to just take the cost/km from your country and estimate that a project will cost X USD/km multiplied by the length of the line in km.

    But this logic, while reasonable, will not yield a completely accurate estimate.
    ''')
    st.write('_________')
    st.subheader('Visualizing the Problem')
    st.write('At the outset of this analysis I stated that building out a train network is akin to a dance where each party is of a different opinion regarding how to carry out the dance itself. This becomes more apparent when we look at the differences between project cost estimates within each individual country')

######## Plotting cost per country ########
    df_engineered = df_engineered[(df_engineered['country']!= 'BY') & (df_engineered['country']!= 'VE') & (df_engineered['country']!= 'AR') & (df_engineered['country']!= 'MY')] 
    df_engineered = df_engineered[df_engineered['length']<= 50]
    df_engineered['average_costkm_country'] = df_engineered.groupby('country')['cost_km_2023'].transform('mean')
    df_engineered['average_cost_country'] = df_engineered.groupby('country')['cost_real_2023'].transform('mean')
    df_unique_countries = (df_engineered.drop_duplicates(subset='country')).sort_values(by='average_costkm_country', ascending=False)
    overall_avg = df_unique_countries['average_costkm_country'].mean()

    fig = px.bar(df_unique_countries,
                x='country',
                y='average_costkm_country',
                color = 'average_cost_country',
                color_continuous_scale='viridis',
                height=500)

    fig.add_shape(
        go.layout.Shape(
            type='line',
            x0=df_unique_countries['country'].iloc[0],
            x1=df_unique_countries['country'].iloc[-1],
            y0=overall_avg,
            y1=overall_avg,
            line=dict(color='purple')
        )
    )
    st.plotly_chart(fig,use_container_width=True)

######## END Plotting cost per country ######## 

    st.write('''
    In the above plot, we can see how much transit costs can vary from country to country, but the difference becomes more stark when we look at the differences in construction costs within each country
    ''')

######## Plotting cost variation country ######## 
    fig = px.scatter(df_engineered.sort_values(by='country'), 
                    x='country',
                    y='cost_km_2023',
                    color='length',
                    color_continuous_scale='viridis')

    st.plotly_chart(fig,use_container_width=True)

######## END Plotting cost variation country ######## 

    st.write('''
    In the above plot we have the cost per km in 2021 USD for each project plotted against each country. On the X-axis, you can see the names of each country 
    and the corresponding dots above that country represent the projects constructed there. The tallest dot represents the most expensive project and the lowest dot represents the cheapest (in terms of cost per km).
    
    Now it's a bit easier to understand how much construction costs can vary. Even within very small countries like Taiwan, Singapore, or Hong Kong 
    where you would expect the variation in building conditions to be relatively minimal, the construction costs between projects fluctuate more than 1000%.

    What is the cause for these fluctuations?
    ''')

    st.write('_________')
    st.subheader('What Differences can we Observe?')
    st.write('''
    
    If we continue this train of thought, using Hong Kong as an example, what exactly could differ from project to project within a given country?
    Even within such a small country,like Hong Kong (Which has a land area roughly the size of 2 Londons or 1 Rhode Islands), there's going to be some variation in site conditions.
    
    Sometimes a project may plan to dig through dirt, but find a big boulder in the way (Delays, change orders). Other times, they might discover unexpected underground water sources or unstable soil conditions, which can complicate the construction process.
    In urban areas like Hong Kong, there's also the challenge of building around existing infrastructure, like water mains, sewage systems, electrical grids, and even older (sometimes forgotten) unused public utilities. 
    
    There's the consideration of the socio-economic impact too. Areas with more businesses or residences may require more complex relocation or compensation efforts (variations in land costs).
    And let's not forget about the regulatory landscape; depending on the specific location and nature of the project, different permits, reviews, and environmental impact assessments might be necessary. 
    Each of these variables can introduce delays, require specialized labor, or otherwise increase the resources needed. 
    
    Using the base dataset, this task becomes more difficult. We know, and have shown, that cost of a project is at least somewhat related the length of the line, but beyond that what else could we show? 
    
    This is where we'll need to find ways to expand the dataset. 
    By using the existing data to create new data, we might be able to find new patterns. Within the full analysis that I linked in the introduction, this is called 'Feature Engineering' and 'Feature Engineering' is the process of creating new features (or columns) in the dataset by means of applying domain knowledge.

    In the section below, I'll describe a few of the features that were engineered and plot a few of these variations to see what impact they have on total project cost.
    ''')
    st.write('_________')


    st.subheader('Engineering New Features')
    st.caption('***Duration***')
    st.write('''
    The first feature that was engineered ("created") is the duration of a project. It's also a great way to illustrate the act of feature engineering. I took two existing features 'start_year' and 'end_year' and created a new feature called 'duration.
    Duration is, understandably important, and also more important for the prediction of the total cost of a transit project (in 2021 dollars) than either the start or end year of a project.

    The duration of project is important in of itself, but it also can capture some relevant information about any delays in a project. Its difficult to identify, for 1000 projects, specific magnitudes of project delays.
    However, if we look at the total project duration, we should see some correlation between duration and cost (if this is true).
    ''')

    ######## Plotting cost variation country ######## 
    df_engineered.head(5)
    avg_cost_by_duration = df_engineered.groupby('duration')['cost_km_2023'].mean().reset_index()
    count_by_duration = df_engineered.groupby('duration').size().reset_index(name='count')
    merged_df = avg_cost_by_duration.merge(count_by_duration, on='duration')

    # Create the bar chart colored by the count
    fig = px.bar(merged_df,
                x='duration',
                y='cost_km_2023',
                color='count',
                color_continuous_scale='viridis',
                labels={'count': 'Number of Projects'} 
                )

    fig.add_hline(y=overall_avg, line_dash="dash", line_color="purple", annotation_text="Overall Avg")
    fig.update_layout(title='Average Cost per km by Project Duration',
                    xaxis_title='Duration',
                    yaxis_title='Average Cost per km (2021)')

    st.plotly_chart(fig,use_container_width=True)
    ######## END Plotting cost variation country ######## 
    st.write('''
    This plot however is somewhat inconclusive, potentially because of a lack of data. The majority of projects have a duration under 10 years. Each duration column under 10 years illustrates that as a project takes longer, the project costs more. 
    However after the 10 year mark, outliers (unduly expensive projects) begin to skew the data and there is some variation in project costs.
 
 
    Let's assume that the assumption that the longer the project duration, the more expensive the project is, is valid in most cases. This potentially doesn't apply to each project, but it's an important piece of the puzzle.
    ''')
    st.caption('***Soil Conditions***')
    st.write('''
    Now, what if, the engineers put together a comprehensive plan and an impeccable budget and schedule. The politicans come to the project 
    site for the ceremonial ground breaking ceremony, dig their shovels into the ground, and find that the ground is solid rock.

    Wouldn't that change the plan, the budget, and the schedule? Even the politicians were expecting dirt.

    Engineers, especially on projects that occupy an area of this size, have a general sense of what kind of soil exists in the area. However, they don't know exactly what kind of 
    soil will occupy every square meter of space along the entire route. A lot of time they need to make educated guesses and create some wiggle room for error.

    Now generally the differences in soil conditions are limited to only a few options (maybe you get a silty clay instead of a sandy clay), but variation within those options can still be difficult to handle on the fly.
    Different soils have different characteristics and, often, a slightly different en-situ condition won't kill a project.

    But it's akin to baking a cake and expecting a specific consistency of the batter, only to find out midway that the flour you're using has a slightly different texture than you're used to. 
    While you can still bake the cake, you might need to adjust other ingredients or baking time to ensure it turns out just right. 
    Similarly, encountering unexpected soil conditions may require adjustments in construction methods or materials, which can impact the project's timeline and cost.
    
    Since we have the location for each project, we can figure out an estimate for what type of soil is most likely to be in that area. Using the SoilGrid API, I generate a 
    soil type probability for each city in the dataset. This will allow us to visualize if a specific soil type correlates with higher construction costs.
    ''')
    ######## Plotting cost variation country ######## 
    avg_cost_by_duration = df_engineered.groupby('duration')['cost_km_2023'].mean().reset_index()
    count_by_duration = df_engineered.groupby('duration').size().reset_index(name='count')
    merged_df = avg_cost_by_duration.merge(count_by_duration, on='duration')

    fig = px.scatter(df_streamlit,
         x="duration",
         y='cost_km_2023',
         color = 'soil_type',
         color_continuous_scale=px.colors.sequential.Viridis
         )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        geo_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=0, b=0, l=0, r=0),
        legend=dict(
            x=0.5,
            y=-.2,
            orientation='h',
            xanchor='center',
            yanchor='top'
        ))
    st.plotly_chart(fig,use_container_width=True)
    ####### END Plotting cost variation country ######## 
    
    st.write('''
    The plot created above relies on the information data provided by the dataset. By using the city, 
    country pairs I was able to generate a latitude and longitude for each data point. Using this lat,lng data I generated information about the underlying in-situ soil conditions using the SoilGrids API.
    I then further refined the output of the API to simplify the output into several categories of soil. The [original values](https://www.isric.org/explore/wrb) were a probability of a specific soil type broken down into 32 major soil groups categorized by the World Reference Base for Soil Resources
    For the purposes of this analysis, the 7 soil groups represent enough variation in soil properties to be useful.

    and interestingly we can see somewhat of a trend in which soils sit at the bottom of the price ranges for each duration and which sit at the top. This however is not a significant pattern, as there isn't a clear delineator we can point to. But it's another piece of the puzzle.
    ''')
    st.caption('***Climate***')
    st.write('''
    As you can imagine, it's more difficult to do manual labor when the weather is uncomfortable. In the searing heat you might need to take more frequent breaks and in the freezing cold, you might be bundled up so tight that you don't know where your gloves end and your hand starts. 
    Of course, this is an example of vernacular thinking, but it makes some sense at least in theory. Let's see if the data show that as well.
    ''')
    fig = px.scatter_3d(df_streamlit,
                        x="precipitation_type",
                        y='temperature_category',
                        z='cost_km_2023',  # Using cost as a 3rd dimension
                        color='cost_km_2023',
                        color_continuous_scale=px.colors.sequential.Viridis,
                        category_orders={
                            'precipitation_type': ['Low', 'Moderate', 'High', 'Very High'],
                            'temperature_category': ['Cold/Cool', 'Mild/Moderate', 'Warm/Hot']
                        })

    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=1.75, y=1.75, z=1.75)  # Adjust these values as needed
        ),
        coloraxis_colorbar=dict(
            x=0.35,  # Adjust the x position
            y=1,  # Adjust the y position
            yanchor='top',
            lenmode='pixels',
            len=340,
            thickness=10,
            orientation = 'h'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        geo_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=0, b=0, l=0, r=0),
    )

    st.plotly_chart(fig, use_container_width=True)
    ####### END Plotting cost variation country ######## 

    st.write('''
    Something that I haven't mentioned yet is that many of these variable might correlate with each other (multicollinearity). Take, for instance, the coexistence of extreme heat and very arid climates.
    This combination is found in only a few regions globally, and even fewer of those regions have dense populations warranting a transportation system.
    Thus, while these individual plots provide insights, they shouldn't be viewed in isolation. No single feature offers a comprehensive understanding to conclusively determine, say, the impact of weather on construction cost.
    From the plot mentioned, we might infer that weather could influence a project's overall cost. However, to make such a determination, this insight must be contextualized with other data points.


    This is the role of machine learning. It allows us to take a dataset with a vast, albeit finite, number of features and see if there's a pattern in the data.
    ''')

    st.write('_________')


    st.subheader('Machine Learning')
    st.write('''
    The process I described above is the data science equivalent of throwing things at a wall and see which ones stick. 
    Within the feature engineering process, I created a number of features that I hoped would help the model more accurately predict the price of a urban railway, but there is some error in that process. 
    Thankfully, in the iterative process of creating a model, there are ways to determine which features work and which features don't. Without going into each details (which are available in the [full analysis](https://github.com/smileshey/TransitCostEstimator)), the final set of features looks like this:    
    ''')
    df_user
    st.write('''
    Within these features, there are some features we didn't discuss, such as:
    - City & country density (how many people occupy each square km)
    - City size (Total city population)
    - Country income class (How affluent each country is per capita)
    - Affordability (How affordable the city is for residents
    - Union Prevalence (How many people in the country are part of labor unions)
    - Poverty rates (How many people in the country live in poverty)
    ''')
    st.write('''
    But which features are the most important for the model?
    ''')

    sorted_idx = importances.argsort()[-10::]
    feature_names_array = np.array(feature_names)
    sorted_names = feature_names_array[sorted_idx]
    greyish_white = '#D0D0D0'

    fig = go.Figure(data=[
        go.Bar(y=sorted_names, 
            x=importances[sorted_idx], 
            orientation='h', 
            text=sorted_names,
    #            textfont=dict(color=greyish_white),
            textposition='outside',
            marker={'color': importances[sorted_idx],
                    'colorscale': 'Viridis_r'})
    ])
    fig.update_layout(
        title='Feature Importances',
        title_font=dict(color=greyish_white),
        yaxis_title='Features',
        yaxis_title_font=dict(color=greyish_white),  # Changes the y-axis title font color
        xaxis_title='Importance',
        xaxis_title_font=dict(color=greyish_white),  # Changes the x-axis title font color
        yaxis_showticklabels=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            range=[0, 300]  # Set your desired maximum value here
        )
    )

    st.plotly_chart(fig, use_container_width=True)



    st.write('''
    Understandably, the length components are very important to the model with overall track length being the most important. 
    Feature importances indicate the relative significance of each feature in predicting the target variable in a machine learning model. They help in understanding which features contribute most to the model's predictions, either positively or negatively. Importances are derived from the model's internal mechanics, such as how often a feature is used to split data in tree-based models. Evaluating feature importances aids in feature selection, model interpretability, and insights into the underlying data relationships.

    While feature importances rank the predictive power of features in a model, they don't specify the direction or nature of their impact. 
    In contrast, SHAP values provide a detailed decomposition of feature effects, quantifying both magnitude and direction of each feature's contribution for individual predictions.
    ''')
    ##### SHAP plot #####
    df_plot_melted['shap_abs'] = df_plot_melted['SHAP'].abs()
    max_abs_values_by_feature = df_plot_melted.groupby('Feature')['shap_abs'].max().reset_index(name='max_shap_abs')
    df_plot_melted = df_plot_melted.merge(max_abs_values_by_feature, on='Feature')
    df_plot_melted = df_plot_melted.sort_values(by='max_shap_abs', ascending=True)
    fig = px.scatter(df_plot_melted, x='SHAP', y='Feature', color='SHAP', size='difference', 
                    color_continuous_scale='viridis', height=500, width=800)

    fig.update_layout(
        xaxis_title="SHAP Value (Impact on Model Output)",
        xaxis=dict(showgrid=True, gridcolor='WhiteSmoke', zerolinecolor='Gainsboro'),
        yaxis=dict(showgrid=True, gridcolor='WhiteSmoke', zerolinecolor='Gainsboro'),
        paper_bgcolor='rgba(0,0,0,0)',
        geo_bgcolor='rgba(0,0,0,0)',
        boxgap=1
    )
    fig.update_traces(marker=dict(line=dict(width=.15)))
    st.plotly_chart(fig,use_container_width=True)
    #####end plot#######.
    st.write('''
    Each point on the above plot indicates how much the output of the 
    model changed when the underlying data for each feature was altered. By adjusting each value for each feature, we can construct an array of values that
    tell us which features contribute the most to the model, in comparison to the baseline model prediction.

    This type of plot is called a [SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html) 
    plot and is a method to explain the output of machine learning models. This is a method to select which features are most valuable. Including unneccessary features into the model
    introduces noise that the model might mistake for a meaningful trend. 
    
    In SHAP, once the final model is completed, it can then calculate how important each feature is to the model. 
    It then takes each row and calculates the model output without a feature (for instance 'length') and then compares this result to the results of the model with all of the features. 
    It repeats this process for each row (project) and calculates a SHAP value for each datapoint. In the above plot, blobs are centered around 0 and a blob further away from 0 describes the magnitude of the effect of removing that feature for that datapoint.
    The further away a point is from 0, or generally how much variance there is for the feature, the more important that feature is to the result of the model.
    In the plot above, both the 'tunnel' and 'length' components were the most important features in the dataset.

    The purpose of adding each feature was to attempt to paint the clearest picture I could of the conditions within each city without telling the model how much the project is expected to cost. 
    By giving the model many features which are no directly related to the cost of a project, my hope is that I can capture enough information to accurately predict the cost given some top-down details about the area. Additionally, by grouping together several features that aren't directly connected to a specific site, 
    I can create a model that generalizes to cities and countries that we don't have data for.
    ''')
    st.write('_________')

    st.subheader('Model Results')
    st.write('''
    As discussed previously, the modelling processing is iterative. It involves creating a model to evaluate the model's results and then revisiting the data to understand which data points the model struggles to identify correctly (outliers). The purpose of revisiting the data is not to outright remove any 
    datapoints that the model deems an outlier, but to understand why they're an outlier. This process required me to verify the details of each project that could be erroneous and, if warranted, altering the data to reflect the most recent updates. In some situations it was advantageous to remove the datapoint, but this was a last resort.

    The summary table you see below outlines this process and shows the benefit of this process. 
    ''')
    combined_metrics
    st.write('''
    The above table describes the results of each model in the process. From the modelling process, I created two models:
    1. A specialized model intended for professionals, such as engineers, familiar with particular locales.
    2. A general-use model, designed for anyone keen on gauging transit project expenses.

    Within this summary, we've been working with the user-friendly model. While both models are similarly accurate, the user model is simpler to use, as it replaces many of the integer oriented features with a descriptor. 
    Both models predict the cost of a transit project within roughly 500M USD and achieve an R-squared value of ~.9.
    ''')

    ##### ERROR BAND PLOT###########
    lambda_at_grade = lambdas_dict['at_grade_transformed']
    lambda_elevated = lambdas_dict['elevated_transformed']
    lambda_tunnel = lambdas_dict['tunnel_transformed']

    predictions['length'] = (inv_boxcox(predictions['at_grade_transformed'], lambda_at_grade) + 
                            inv_boxcox(predictions['elevated_transformed'], lambda_elevated) + 
                            inv_boxcox(predictions['tunnel_transformed'], lambda_tunnel))    
    predictions['error'] = predictions['cost_real_2023_transformed'] - predictions['prediction_label']
    # Create bins
    bin_size = 10
    bins = np.arange(0, predictions['length'].max() + bin_size, bin_size)
    predictions['length_bin'] = pd.cut(predictions['length'], bins, labels=bins[:-1] + bin_size/1, right=False)

    # Group by bins and calculate mean error and standard deviation for each bin
    bin_means = predictions.groupby('length_bin')['error'].mean()
    bin_stds = predictions.groupby('length_bin')['error'].std()
    bin_counts = predictions.groupby('length_bin').size()

    predictions['percentage_error'] = ((predictions['prediction_label'] - predictions['cost_real_2023_transformed']) / predictions['cost_real_2023_transformed']) * 100

    # Group by bins and calculate mean percentage error and its standard deviation for each bin
    bin_means_percentage_error = predictions.groupby('length_bin')['percentage_error'].mean()
    bin_stds_percentage_error = predictions.groupby('length_bin')['percentage_error'].std()

    fig = go.Figure([
        go.Scatter(
            name='Mean Percentage Error',
            x=bin_means_percentage_error.index,  # Binned train line lengths
            y=bin_means_percentage_error.values,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Upper Bound',
            x=bin_means_percentage_error.index,
            y=bin_means_percentage_error.values + bin_stds_percentage_error.values,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=bin_means_percentage_error.index,
            y=bin_means_percentage_error.values - bin_stds_percentage_error.values,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])

    fig.add_trace(
        go.Bar(
            x=bin_counts.index,
            y=bin_counts.values,
            name='Count',
            marker_color='rgba(255, 182, 193, 0.1)',  # You can change this color as per your preference
            yaxis='y2'
        )
    )

    fig.update_layout(
        xaxis_title='Train Line Length (km)',
        yaxis_title='Prediction Error (%)',
        yaxis=dict(
            range=[-50, 50]  # Adjust the range as per the actual percentage errors in your data
        ),
        yaxis2=dict(
                overlaying='y', 
                side='right',
                position=1,  # Adjust as needed
                range=[0, max(bin_counts.values)/.3],  # Adjust range to half the maximum count or as desired
                showticklabels=False  # To hide the tick labels of y2 axis
            ),
        title='Percentage Prediction Error by Train Line Length',
        hovermode="x"
    )

    # Add a horizontal line at y=0 to indicate the baseline (no error)
    fig.add_shape(
        type='line',
        line=dict(dash='dash'),
        x0=min(bin_means_percentage_error.index),
        x1=max(bin_means_percentage_error.index),
        y0=0,
        y1=0
    )

    st.plotly_chart(fig, use_container_width=True)

 
    ##### END ERROR BAND PLOT###########
    st.write('''
    The error band plot above shows how the model's predictions change as the length of the line increases. From the 1700+ datapoints in the dataset, we set aside ~300 datapoints to use as a measuring stick for the model.
    These datapoints are represented in the above plot and show the mean error (blue) as a function of the total length. 
    
    We can see that the blue line stays relatively consistent until the length of the project reaches about 20km. After which, two things occur:

    1) The number of datapoints (predidictions) gets smaller. Meaning that a single outlier can unduly affect the average.
    2) The prediction error begins to deviate away from 0, indicating that the model, on average, produces less accurate predictions.
        
    With additional data, it's likely that the mean error would be more uniform accross all track lengths, however it is also expected that larger projects are prone to more cost overruns.  Additional data would also likely help reduce the magnitude of the variance (dark grey band) which indicates the distribution of errors at each length.
        ''')


    st.write('_________')
    st.subheader('Model Validity')
    st.write('''
    The model performance described above demonstrates the model's accuracy on all of the data but it does not confirm if the model's predictions 
    should be trusted. On the next sheet I'll describe the predictions made by the model, demonstrating the model's effectiveness on various segments of the data, 
    and illustrating that it fulfills several important assumptions.
    ''')

elif menu == 'Evaluating the Model':
    st.title('Evaluating the Model')
    st.write('_________')
    st.subheader('What are Predictions?')
    st.write('''
    A prediction, in the machine learning sense, is less hand-wavy and ephemeral than the word sounds. In the process of creating a model, we train an algorithm with a bunch of data. We tell the algorithm which feature is the target (dependent variable), which features it should use to find patterns (independent variables), and the algorithm then learns from this data, identifying intricate relationships between the features. 
    The goal of the model is to find an equation that best connects the dependent variable with the independent variables. 
    
    As an example let's devise fictitious and simplified version of the algorithm used to create the model for this project. Let's say the dataset is only 3 points:

    1. Length = 10, At Grade = 10, tunnel = 0, cost = 10
    2. Length = 10, At Grade = 0, tunnel = 10, cost = 20
    3. Length = 10, At Grade = 5, tunnel = 5, cost = 15

    The model would then look at this data and say that 1km of at grade track is the equivalent of 1 cost unit and 1km of tunneled track is equal to 2 cost units. This equation would look like:

        cost = 1(AG) + 2(T)

    and then we could tell our model to generate a "prediction" for a 4th datapoint using the same equation:

    4. Length = 15, At Grade = 5, tunnel = 10, cost = ?
    
    and the model would predict that:

        cost = 1(5) + 2(10), which would return 25 cost units
    
    The purpose of creating a model that can identify trends within a dataset is so that we can apply that model to new data (data that we haven't calculated the cost for) and make a prediction of the cost for that new data. 
    
    Once trained, the model can take new, unseen data and make an informed guess or "prediction" about the target based on the patterns it 
    recognized during training. It's not a vague guess or intuition; it's a calculated output based on patterns present in the data    
    ''')
    st.write('_________')
    st.subheader('What do the Model Predictions Look Like?')
    st.write('''
    The predictions provided by this model will simply be a number (cost) that was calculated using the same features present in the dataset. This is useful if you're looking to identify the cost of a transit project, but it's not worthwhile to evaluate by itself. It's more valuable to look at how the model does given data it hasn't seen before and evaluate the difference between the known cost of the project and the cost calculated by the model. 
    
    This difference between the actual and the predicted value is called a Residual and it's a very important concept for evaluating a machine learning model.

    ''')
    ### Residuals Calculation
    lambda_at_grade = lambdas_dict['at_grade_transformed']
    lambda_elevated = lambdas_dict['elevated_transformed']
    lambda_tunnel = lambdas_dict['tunnel_transformed']

    predictions['length'] = (inv_boxcox(predictions['at_grade_transformed'], lambda_at_grade) + 
                            inv_boxcox(predictions['elevated_transformed'], lambda_elevated) + 
                            inv_boxcox(predictions['tunnel_transformed'], lambda_tunnel)) 
    predictions['Residuals'] = predictions['cost_real_2023_transformed'] - predictions['prediction_label']
    mean_res = np.mean(predictions['Residuals'])
    std_res = np.std(predictions['Residuals'])
    predictions['Standardized_Residuals'] = (predictions['Residuals'] - mean_res) / std_res

    ### Plot of Residuals
    scatter = go.Scatter(
        x=predictions['prediction_label'],
        y=predictions['Standardized_Residuals'],
        mode='markers',
        marker=dict(
            color=predictions['length'],
            colorscale='viridis',
            size=6,
            colorbar=dict(title='Length')
        ),
        name='Scatter',
        yaxis='y2'
    )

    # Create the histogram using plotly.graph_objects
    histogram = go.Histogram(
        y=predictions['Standardized_Residuals'], 
        name='Histogram',
        marker_color='lightseagreen',
        opacity=0.7,
        xaxis='x2',
        yaxis='y',
        nbinsy=40
    )
    fig = go.Figure([scatter, histogram])
    fig.update_layout(
        yaxis=dict(title='Standardized Residuals', side="left", showticklabels=False),
        yaxis2=dict(title='Standardized Residuals', side="left", showticklabels=True),
        xaxis=dict(domain=[0, 0.85],title='Prediction Label'),
        xaxis2=dict(domain=[0.85, 1], showticklabels=False),
        barmode='overlay',
        showlegend = False
    )
    st.plotly_chart(fig, use_container_width=True)
    ### End plot of residuals

    st.write('_________')
    st.subheader('What Makes the Model\'s Predictions Valid?')
    st.write('''

    To make the model valid, we're looking to satisfy 3 assumptions:

    1. The standardized residuals are normally distributed around 0.
    2. Residuals have consistent spread across the predicted values.
    3. The residuals for the independent variables are independent of each other.


    With the plot above, we can see that the residuals generated from this model are generally scattered around 0 (assumption #1) and there aren't any discernible trends across the predicted values that would indicate the model is systematically overestimating or underestimating certain values (assumption #2).
    There are some [outliers](https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm#:~:text=Definition%20of%20outliers,what%20will%20be%20considered%20abnormal.), but those outliers were deemed to be genuine datapoints. One tool to evaluate an outlier is to use the Standardized Residuals, instead of the base Residuals. 
 
    This metric has the same purpose as the previously discussed 'Residual', but it's standardized against the standard deviation of all of the residuals in the dataset and
    it's the metric we'll be using to create density plots of the important features to check autocorrelation (assumption #3)
    ''')
    ### Plot of standardized residuals
    scatter = go.Scatter(
    x=predictions['prediction_label'],
    y=predictions['Standardized_Residuals'],
    mode='markers',
    marker=dict(
        color=predictions['length'],
        colorscale='viridis',
        size=6,
        colorbar=dict(title='Length')
    ),
    name='Scatter',
    yaxis='y2'
    )

    histogram = go.Histogram(
        y=predictions['Standardized_Residuals'], 
        name='Histogram',
        marker_color='lightseagreen',
        opacity=0.7,
        xaxis='x2',
        yaxis='y',  # Histogram will refer to y (primary y-axis)
        nbinsy=40
    )

    fig = go.Figure([scatter, histogram])

    fig.update_layout(
        xaxis=dict(domain=[0, 0.85],title='Prediction Label'),
        yaxis=dict( showticklabels=False),
        yaxis2=dict(title='Standardized Residuals', side="left", showticklabels=True),

        xaxis2=dict(domain=[0.85, 1], showticklabels=False),
        barmode='overlay',
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    ### END Plot of standardized residuals

    st.write('''
    The above plot shows the same scatterplot as before, except this time using the standardized residuals. Notice the difference in the axis labels. a standardized residual is the number of standard deviations that a given observed value deviates from its predicted value. 
    
    This allows for a comparison of residuals across different scales or units, and helps in identifying outliers or points that have larger deviations than expected under a given model. When visualizing standardized residuals, values far from zero may indicate potential issues with the model or specific data points. 
    
    Below we'll look at sub-sets of standardized residuals for each prediction made by the model to illustrate which situations the model performs well and where it may struggle. 
    Since 'length' was the most important feature for the model, let's see how it performed on different lengths of track.
    ''')
    #### Predictions from model
    q1 = predictions['length'].quantile(0.25)
    q2 = predictions['length'].quantile(0.5)
    q3 = predictions['length'].quantile(0.75)
    bins = [0, q1, q2, q3, float('inf')]
    labels = ["short", "medium", "medium-long", "long"]
    predictions['length_category'] = pd.cut(predictions['length'], bins=bins, labels=labels, right=False)
    
    all_projects = predictions['Standardized_Residuals']
    short_projects = predictions[predictions['length_category'] == 'short']['Standardized_Residuals']
    medium_projects = predictions[predictions['length_category'] == 'medium']['Standardized_Residuals']
    medium_long_projects = predictions[predictions['length_category'] == 'medium-long']['Standardized_Residuals']
    long_projects = predictions[predictions['length_category'] == 'long']['Standardized_Residuals']

    # Generate colors from viridis colormap
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    hex_colors = [plt.colors.rgb2hex(color) for color in colors]

    # Create distplot
    fig = ff.create_distplot([all_projects, short_projects, medium_projects, medium_long_projects, long_projects], 
                         group_labels=['All','Short', 'Medium', 'Medium-Long', 'Long'], 
                         bin_size=.5, 
                         curve_type='normal',
                         histnorm='probability density',
                         show_hist=False)

    # Set colors for density plots
    for i, trace in enumerate(fig.data[:5]):  # Assuming there are 4 density plots
        trace['line']['color'] = hex_colors[i]

    # Set colors for rug plots
    num_density_plots = 5
    for i, trace in enumerate(fig.data[-5:]):  # Assuming there are 4 rug plots
        if 'marker' in trace:
            trace['marker']['color'] = hex_colors[i]

    # Customize the layout
    fig.update_layout(
        title = 'Model Predictions Per Track Length',
        paper_bgcolor='rgba(0,0,0,0)',
        geo_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=0, l=0, r=0),
        legend=dict(
            x=0.9,
            y=1,
            orientation='v',
            xanchor='center',
            yanchor='top',
        ))

    st.plotly_chart(fig, use_container_width=True)
    ### END PLOT Model Results by Length
    
    st.write('''
    The plot above shows the density of the predictions, as they pertain to the length of the track used for the project (km). 
    Using the [interquartile ranges](https://www.scribbr.com/statistics/interquartile-range/#:~:text=The%20interquartile%20range%20(IQR)%20contains,half%20of%20a%20data%20set.) for the length feature, I've created 4 bins:
    - Short (<10km | <25th quartile)
    - Medium (<17km | 25-50th quartile)
    - Medium-Long (<29km | -50-75th quartile)
    - Long (>29km | >75th quartile)

    The density plot tells us how dense the predictions are for each length category. The X-axis are the residuals and the y-axis represents the density. Since a residual of 0 indicates that the model correctly predicted the cost of the project, this type of distribution would be the ideal, but not always realistic, outcome for the model.
    
    From this plot we can see that the residuals, for each length category, are centered around 0 and [somewhat normally distributed](https://www.mathsisfun.com/data/standard-normal-distribution.html) (Minus the 'Long' group). However the distribution for the shortest length category is much more dense around 0. This implies that the model is much better at predicting the cost of a project when the project is under 10km in length. 
    
    In the previously discussed SHAP plot, we showed that length and tunnel length were the most important features. Let's repeat this for the tunnel length.
    ''')
    ### dist plot for length of tunnel
    bins = [0, 1, 90, float('inf')]
    labels = ["no tunnel", "mixed", "subway"]
    lambda_tunnel = lambdas_dict['tunnel_transformed']
    predictions['tunnel']= inv_boxcox(predictions['tunnel_transformed'], lambda_tunnel)-1
    predictions['tunnel_per'] = (predictions['tunnel']/predictions['length'])*100
    predictions['tunnel_category'] = pd.cut(predictions['tunnel_per'], bins=bins, labels=labels, right=False)

    all_projects = predictions['Standardized_Residuals']
    no_tunnel_projects = predictions[predictions['tunnel_category'] == 'no tunnel']['Standardized_Residuals']
    mixed_projects = predictions[predictions['tunnel_category'] == 'mixed']['Standardized_Residuals']
    subway_projects = predictions[predictions['tunnel_category'] == 'subway']['Standardized_Residuals']

    # Generate colors from viridis colormap
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    hex_colors = [plt.colors.rgb2hex(color) for color in colors]

    # Create distplot with the new categories
    fig = ff.create_distplot([all_projects,no_tunnel_projects, mixed_projects, subway_projects], 
                            group_labels=['All','No Tunnel', 'Mixed', 'Subway'], 
                            bin_size=.5, 
                            curve_type='normal',
                            histnorm='probability density',
                            show_hist=False)

    # Set colors for density plots
    for i, trace in enumerate(fig.data[:4]):  # Now assuming there are 3 density plots
        trace['line']['color'] = hex_colors[i]

    # Set colors for rug plots
    for i, trace in enumerate(fig.data[-4:]):  # Now assuming there are 3 rug plots
        if 'marker' in trace:
            trace['marker']['color'] = hex_colors[i]

    # Customize the layout
    fig.update_layout(
        title = 'Model Predictions Per Tunnel Length',
        paper_bgcolor='rgba(0,0,0,0)',
        geo_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=0, l=0, r=0),
        legend=dict(
            x=0.9,
            y=1,
            orientation='v',
            xanchor='center',
            yanchor='top',
        ))

    st.plotly_chart(fig, use_container_width=True)
    ### end length of tunnel dist plot
    st.write('''
    Again we see a create distribution for all 3 sub-categories of tunnel-length. 
    The model is best at predicting the cost of at-grade projects and pretty good at 
    predicting the cost of subway projects (completely underground). 
    However the predictions from the model when the project is a mixed type project are 
    less precise. When the project is split between some combination of at-grade/elevated and 
    underground, the predictions are less normally distributed. 

    Additionally, since the density plots show a relatively normal distribution for the data and its 
    important subsets, this indicates that model is lacking significant Homoscedasticity and the residuals are equally spread across all independent variables (assumption #3).
    ''')
    st.write('_________')
    st.subheader('Auxillary Features')
    st.write('''
    Since we've shown that the assumptions for a machine learning model were met within this analysis, let's evaluate the auxillary features that weren't discussed above. 
    Features like climate, soil type, and socioeconomic conditions weren't as important to the model as the length components, but they still added value. Let's plot the standardized residuals to show how the model performs with each auxillary feature set.
    ''')
    predictions_socio = predictions

    fig = px.scatter(
        predictions_socio,
        color='city_size',
        x='prediction_label',
        y='Standardized_Residuals', 
        title="Residuals vs. Predictions by City Population",
        marginal_y="histogram")
    st.plotly_chart(fig, use_container_width=True)

    st.write('''
    The first plot shows the standardized residuals of the model, aggregated by city population. 
    The resulting plot is slightly skewed to the upside, however not significantly enough to consider this a pattern given that the overall residuals trend is more normally distributed. 
    
    We can continue this analysis by aggregating the residuals by region & train types.
    ''')

    fig = px.box(predictions_socio, x='region', y='Standardized_Residuals', color='train_type',
                title='Distribution of Standardized Residuals by Sub-region',
                labels={'standardized_residuals': 'Standardized Residuals', 'sub_region': 'Sub-region', 'train_type': 'Train Type'})

    fig.update_layout(xaxis_title='Sub-region',
                    yaxis_title='Standardized Residuals')

    st.plotly_chart(fig, use_container_width=True)

    st.write('''
    Once again the residuals, as aggregated by both region and train type, are roughly centered on zero. This is positive for the integrity of the model as it shows the model has limited bias.
    
    Lastly, the soil type parameter standardized residuals provide a similar result as the previous plot.
    ''')

    fig = px.scatter(predictions_socio, color='soil_type', x='prediction_label', y='Standardized_Residuals', 
                    title="Standardized Residuals vs. Predictions by Soil Type", marginal_y="histogram")

    fig.update_layout(
        width=800,  # Set the width of the plot
        height=600,  # Set the height of the plot
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-.5,
            xanchor="center",
            x=0.5
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write('_________')
    st.subheader('Your Turn')
    st.write('''
    Now that I've shown that the model is valid and described the model, how it works, and why it's neccessary, it's your turn to make predictions.
    In the sidebar, there's a tab that directs you to the modelling section of this analysis. In this section, you'll be able to use the model to generate predictions for a transit project of your own. 
    
    As a starting point, it helps to look up a project being completed in your city. By using the parameters from your local project, you can see how close the predictions from the model are the your governments prediction of the project's cost.
    ''')


elif menu == ':sparkles: **:rainbow[Project Cost Calculator]** :sparkles:':
    ### Existing Features
    cont_feats = ['end_year','at_grade_transformed','tunnel_transformed',
              'elevated_transformed', 'duration_transformed','stations_transformed',
              'tunnel_MRT_interaction','tunnel_asia_interaction','at_grade_MRT_interaction','at_grade_asia_interaction','stations_Streetcar_interaction',
              'stations_LightRail_interaction', 'stations_MRT_interaction','stations_tunnel_interaction',
              'stations_atgrade_interaction','stations_elevated_interaction','duration_tunnel_interaction',
              'duration_atgrade_interaction', 'duration_elevated_interaction',
              'extension_tunnel_interaction', 'extension_atgrade_interaction','extension_elevated_interaction']

    cat_feats = ['region', 'sub_region', 'train_type', 'soil_type', 'city_size',
            'country_income_class', 'precipitation_type','elevation_class',
             'poverty_rate','temperature_category', 'city_density_type','project_type']

    ### Feature categories
    feature_categories = {'region': ['Africa', 'Americas', 'Asia', 'Europe', 'Oceania'],
    'sub_region': ['Australia and New Zealand',
    'Eastern Asia',
    'Eastern Europe',
    'Latin America and the Caribbean',
    'Northern Africa',
    'Northern America',
    'Northern Europe',
    'South-eastern Asia',
    'Southern Asia',
    'Southern Europe',
    'Western Asia',
    'Western Europe'],
    'soil_type': ['Clay Dominant',
    'Cold Climates (Permafrost, Rock Outcrops)',
    'Environment Dependent (Wetlands, Volcanic Ash, Mineral Rich)',
    'Fertile/Agricultural (Grasslands, Food Bearing, Pasture)',
    'High Altitude/Wet (Mountain, Swampy)',
    'River Valleys/Deltas (River Sediments)',
    'Saline/Arid (Desert Soils, High Salt Content)'],
    'city_size': ['Large (5M-15M)',
    'Medium (1M-5M)',
    'Metropolis (>15M)',
    'Small (<1M)'],
    'train_type': ['Light Rail', 'MRT', 'Monorail/APM', 'Streetcar'],
    'country_income_class': ['high-income', 'low-income', 'middle-income'],
    'elevation_class': ['Coastal', 'High-land', 'Mid-land'],
    'precipitation_type': ['High', 'Low', 'Moderate'],
    'temperature_category': ['Cold', 'Hot', 'Mild'],
    'poverty_rate': ['High Poverty', 'Low Poverty', 'Moderate Poverty'],
    'city_density_type': ['High Density', 'Low Density', 'Medium Density'],
    'project_type': ['Extension', 'New']}

    #### Sorting Features for better UI
    # feature_categories['region'].sort()
    # feature_categories['sub_region'].sort()
    
    # gauge_width_order = ['standard', 'non-standard']
    # feature_categories['gauge_width'] = sorted(feature_categories['gauge_width'], key=lambda x: gauge_width_order.index(x))

    city_size_order = ['Small (<1M)','Medium (1M-5M)','Large (5M-15M)','Metropolis (>15M)']
    feature_categories['city_size'] = sorted(feature_categories['city_size'], key=lambda x: city_size_order.index(x))

    precipitation_order = ['Low', 'Moderate', 'High', 'Very High']
    feature_categories['precipitation_type'] = sorted(feature_categories['precipitation_type'], key=lambda x: precipitation_order.index(x))  

    poverty_order = ['High Poverty', 'Moderate Poverty','Low Poverty']
    feature_categories['poverty_rate'] = sorted(feature_categories['poverty_rate'], key=lambda x: poverty_order.index(x))

    city_density_order = ['High Density', 'Medium Density','Low Density']
    feature_categories['city_density_type'] = sorted(feature_categories['city_density_type'], key=lambda x: city_density_order.index(x))

    # country_density_order = ['High Density', 'Medium Density','Low Density']
    # feature_categories['country_density_type'] = sorted(feature_categories['country_density_type'], key=lambda x: country_density_order.index(x))

    ### Creating the user interface
    st.header('Generating Your Own Predictions')
    st.write('---------------------------')
    lambda_at_grade = lambdas_dict['at_grade_transformed']
    lambda_elevated = lambdas_dict['elevated_transformed']
    lambda_tunnel = lambdas_dict['tunnel_transformed']

    predictions['length'] = (inv_boxcox(predictions['at_grade_transformed'], lambda_at_grade) + 
                            inv_boxcox(predictions['elevated_transformed'], lambda_elevated) + 
                            inv_boxcox(predictions['tunnel_transformed'], lambda_tunnel))
    subset_predictions = predictions[(predictions['length'] >= 0) & (predictions['length'] <= 20)]
    subset_predictions['absolute_error'] = (subset_predictions['cost_real_2023_transformed'] - subset_predictions['prediction_label']).abs()
    subset_mae = subset_predictions['absolute_error'].mean()    
    formatted_subset_mae = "{:.0f}".format(subset_mae)

    # subset_predictions['error'] = subset_predictions['cost_real_2023_transformed'] - subset_predictions['prediction_label']
    # subset_mean_error = abs(subset_predictions['error']).mean()
    # formatted_subset_mean_error = "{:.1f}".format(subset_mean_error)

    st.write(f'''
    In this section, you can estimate the construction cost for your own project. While not exact, the predictions offer a reasonable approximation. Keep in mind, the model might not fully account for exceptionally unique projects, projects set to take place far into the future, and it does not cover rolling stock or financial costs like loans or interest.

    The model’s Mean Absolute Error (MAE) is ±$470M USD across all lengths. A more precise error adjusted for your project's length will appear beneath the predicted value in the sidebar
    ''')
    st.write('---------------------------')
    st.subheader("0. Choose Your Units")

    # continuous features
    feature_ranges_km = {
        'length': (.5, 20.0),  # Specify as floats
        'tunnel': (0.0, 20.0),  # Specify as floats
        'elevated': (0.0, 20.0),  # Specify as floats
        'at_grade': (0.0, 20.0),  # Specify as floats
        'stations': (0.0, 25.0),  # Specify as floats, if you want to allow tenths for stations as well
        'duration': (1.0, 25.0)  # Specify as floats
    }

    currency_conversion_rates = {
        'USD': 1,  # Base rate for conversion, U.S. dollar
        'EUR': 0.92,  # Euro
        'JPY': 147.46,  # Japanese yen
        'GBP': 0.80,  # Pound sterling
        'AUD': 1.52,  # Australian dollar
        'CAD': 1.34,  # Canadian dollar
        'CHF': 0.86,  # Swiss franc
        'CNY': 7.1,  # Renminbi (Chinese yuan)
        'HKD': 7.80,  # Hong Kong dollar
        'NZD': 1.60,  # New Zealand dollar
        'SEK': 10.40,  # Swedish krona
        'KRW': 1330.00,  # South Korean won
        'SGD': 1.35,  # Singapore dollar
        'NOK': 10.50,  # Norwegian krone
        'MXN': 19.00,  # Mexican peso
        'INR': 79.85,  # Indian rupee
        'RUB': 90.00,  # Russian ruble
        'ZAR': 18.8,  # South African rand
        'TRY': 30.37,  # Turkish lira
        'BRL': 5.00,  # Brazilian real
    }

    cont_input_values = {}
    cat_input_values = {}

    def convert_ranges_to_miles(feature_ranges_km):
        feature_ranges_miles = {}
        for key, value in feature_ranges_km.items():
            feature_ranges_miles[key] = (value[0] * 0.621371, value[1] * 0.621371 if key == 'length' else value[1])
        return feature_ranges_miles

    col1, col2 = st.columns(2)

    # Place unit selection radio button in the first column
    with col1:
        unit = st.selectbox("Length Unit:", ('Kilometers', 'Miles'))

    # Place currency selection selectbox in the second column
    with col2:
        selected_currency = st.selectbox('Choose the currency for the results:', options=list(currency_conversion_rates.keys()))


    # Update feature ranges based on the selected unit
    feature_ranges = feature_ranges_km if unit == 'Kilometers' else convert_ranges_to_miles(feature_ranges_km)
    def miles_to_km(value_in_miles):
        return value_in_miles * 1.60934
    st.write('---------------------------')

    st.subheader("1. Describe the Type of Railway Being Constructed")
    # Single column for 'length' slider
    length_col = st.container()

    with length_col:
        if 'length' not in st.session_state:
            st.session_state.length = max(feature_ranges['length'][0], .5)  # Ensure at least 1 unit to prevent 0, as float
        # Correct the step type mismatch error by making min_value and max_value floats and allowing for unit selection
        slider_format = "%.1f " + unit
        st.session_state.length = st.slider('Select Total Length', min_value=float(feature_ranges['length'][0]), max_value=float(feature_ranges['length'][1]), step=0.1, format=slider_format)
        # Convert length from miles to kilometers if necessary
        cont_input_values = {'length': miles_to_km(st.session_state.length) if unit == 'Miles' else st.session_state.length}


    slider_format = "%.1f km" if unit == 'Kilometers' else "%.1f mi"
    if st.session_state.length > 0.0:
        st.write("Of the Total Length, What Portion of the Track is Underground, At Grade, or Elevated?")
        st.session_state.length_set = True

    if st.session_state.length_set:
        cols = st.columns(3)
        # Add step=0.1 to all sliders to allow for tenths of a kilometer increments
        cont_input_values['tunnel'] = cols[0].slider('Underground Track Length', min_value=0.0, max_value=st.session_state.length, step=0.1, format=slider_format)
        available_length = st.session_state.length - cont_input_values['tunnel']

        if available_length > 0:
            cont_input_values['at_grade'] = cols[1].slider('Street Level Track Length', min_value=0.0, max_value=available_length, step=0.1, format=slider_format)
            available_length -= cont_input_values['at_grade']
        else:
            cont_input_values['at_grade'] = 0

        if available_length > 0:
            cont_input_values['elevated'] = cols[2].slider('Elevated Track Length', min_value=0.0, max_value=available_length, step=0.1, format=slider_format)
        else:
            cont_input_values['elevated'] = 0


        if 'train_type' in feature_categories:
            cols = st.columns(1)
            cat_input_values['train_type'] = cols[0].radio('What kind of train is it?', options=feature_categories['train_type'], horizontal=True)
            del feature_categories['train_type']

        if 'project_type' in feature_categories:
            project_type_question = 'Is This Project an Extension of an Existing Line?'
            project_type_options = ['No', 'Yes']

            user_response = st.radio(project_type_question, project_type_options, horizontal=True)
            project_type_mapping = {'Yes': 'Extension', 'No': 'New'}

            cat_input_values['project_type'] = project_type_mapping[user_response]
            del feature_categories['project_type']

        cols = st.columns(2)
        cont_input_values['duration'] = cols[0].slider('How Long will the Project take to Build?', min_value=1, max_value=25, format="%d Years")
        cont_input_values['stations'] = cols[1].slider('How Many Stations Will be Built?', min_value=0, max_value=25, format="%d stations")
        cont_input_values['start_year'] = 2023
        cont_input_values['end_year'] = cont_input_values['start_year'] + cont_input_values['duration']
        st.write('---------------------------')

    st.subheader("2. Describe the Project Area")
    cols = st.columns(2)
        
    # Streamlit dropdown for region selection
    if 'region' in feature_categories and 'sub_region' in feature_categories and 'city_size' in feature_categories and 'soil_type' in feature_categories:
            cols = st.columns(2)

            cat_input_values['region'] = cols[0].selectbox('What Region is the Project in?', options=feature_categories['region'])
            
            # Define subregion choices based on the selected region
            if cat_input_values['region'] == 'Asia':
                sub_region_choices = ['Eastern Asia', 'Central Asia', 'Southern Asia', 'Western Asia', 'South-eastern Asia']
            elif cat_input_values['region'] == 'Europe':
                sub_region_choices = ['Southern Europe', 'Western Europe', 'Eastern Europe', 'Northern Europe']
            elif cat_input_values['region'] == 'Americas':
                sub_region_choices = ['Northern America', 'Latin America and the Caribbean']
            elif cat_input_values['region'] == 'Africa':
                sub_region_choices = ['Northern Africa','Sub-Saharan Africa']
            elif cat_input_values['region'] == 'Oceania':
                sub_region_choices = ['Australia and New Zealand']
            else:
                sub_region_choices = feature_categories['sub_region']

            cat_input_values['sub_region'] = cols[1].selectbox('What Sub-Region is the Project in?', options=sub_region_choices)
            cat_input_values['city_size'] = cols[0].selectbox('What\'s the population of the city?', options=feature_categories['city_size'])
            cat_input_values['soil_type'] = cols[1].selectbox('What kind of soil is this city built on?', options=feature_categories['soil_type'])

            del feature_categories['region']
            del feature_categories['sub_region']
            del feature_categories['city_size']
            del feature_categories['soil_type']


    if 'city_density_type' in feature_categories:
        cols = st.columns(1)
        cat_input_values['city_density_type'] = cols[0].radio('How Densely Populated is the City?', options=feature_categories['city_density_type'],horizontal = True)
        del feature_categories['city_density_type']
    
    st.write('---------------------------')

    if 'precipitation_type' in feature_categories and 'temperature_category' in feature_categories and 'elevation_class' in feature_categories:
        st.subheader("3. Describe the Type of Climate the Project is in")
        cols = st.columns(1)

        cat_input_values['precipitation_type'] = cols[0].selectbox('How would you describe the precipitation in the region?', options=feature_categories['precipitation_type'])
        cat_input_values['temperature_category'] = cols[0].radio('How would you describe the climate there?', options=feature_categories['temperature_category'],horizontal = True)
        cat_input_values['elevation_class'] = cols[0].radio('What\'s the elevation like at the project site?', options=feature_categories['elevation_class'],horizontal = True)

        # Removing them from the dictionary to avoid duplication in the following loop
        del feature_categories['precipitation_type']
        del feature_categories['temperature_category']
        del feature_categories['elevation_class']


    st.write('---------------------------')
    st.subheader("4. Describe the Socioeconomic Conditions of the Project Area")

    if 'country_income_class' in feature_categories and 'poverty_rate' in feature_categories:
        cat_input_values['poverty_rate'] = st.radio('How much poverty is present in this country?', options=feature_categories['poverty_rate'],horizontal = True)
        cat_input_values['country_income_class'] = st.radio('How Wealthy is the Country?', options=feature_categories['country_income_class'],horizontal = True)

        del feature_categories['country_income_class']
        del feature_categories['poverty_rate']


    st.write('---------------------------')

    st.markdown("<div style='text-align: left; font-size: 23px;color: orange'>"
                "👈 Click the 'Make Prediction' button once you are finished"
                "</div>", unsafe_allow_html=True)


    for idx, (feat, options) in enumerate(feature_categories.items()):
        
        if idx % 3 == 0:
            cols = st.columns(3)
        cat_input_values[feat] = cols[idx % 3].radio(f'Select {feat}', options=options)

    # Start the summary in the sidebar
    st.sidebar.header("Summary of Your Selections:")
    input_values = {**cont_input_values, **cat_input_values}

    markdown_lines = []

    # Iterate over the input values and append styled markdown to the list
    paragraph_keys = ["train_type", "length", "tunnel", "elevated", "at_grade", "duration", "stations", "sub_region", "city_density", "city_size",'elevation_class']

    components = []

    if float(input_values['tunnel']) > 0:
        components.append(f"<span style='color:orange;'>{round(float(input_values['tunnel']), 1)} km</span> of tunneled track")
    if float(input_values['elevated']) > 0:
        components.append(f"<span style='color:orange;'>{round(float(input_values['elevated']), 1)} km</span> of elevated track")
    if float(input_values['at_grade']) > 0:
        components.append(f"<span style='color:orange;'>{round(float(input_values['at_grade']), 1)} km</span> of at-grade track")

    # Format the components list with appropriate conjunctions
    if len(components) == 0:
        component_str = ' '
    elif len(components) == 1:
        component_str = components[0]
    elif len(components) == 2:
        component_str = f"{components[0]} and {components[1]}"
    else:
        component_str = ', '.join(components[:-1]) + f", and {components[-1]}"

    if input_values['temperature_category'] == 'Very Hot or Extreme Heat':
        temperature_str = "Very Hot"
    else:
        temperature_str = input_values['temperature_category']

    if input_values['project_type'] == 'Extension':
        project_str = "an Extension of an existing "
    else:
        project_str = 'a New '

    # Simplify precipitation type
    if input_values['precipitation_type'] == 'Arid':
        precipitation_str = "very little rainfall"
    elif input_values['precipitation_type'] == 'Semi-Arid':
        precipitation_str = "little rainfall"
    elif input_values['precipitation_type'] == 'Very-Arid':
        precipitation_str = "almost no rainfall"
    else:
        # Extracting the first word (like High, Low, Moderate) for other categories
        precipitation_str = input_values['precipitation_type'].split('-')[0].lower() + " rainfall"

    if input_values['poverty_rate'] == 'Mostly Impoverished':
        poverty_str = "A lot of poverty"
    elif input_values['poverty_rate'] == 'Very Impoverished':
        poverty_str = "Abundant poverty"
    else:
        poverty_str = input_values['poverty_rate']

    paragraph = (f"<span style='font-size: 12.5px;'>"
                f"You selected a <b> <span style='color:orange;'>{project_str}</span></b><b><span style='color:orange;'>{input_values['train_type']} line </span></b> with a track length of "   
                f"<span style='color:orange;'>{input_values['length']} km</span>, including {component_str}. "
                f"The project duration is <span style='color:orange;'>{input_values['duration']} Years</span> and "
                f"<span style='color:orange;'>{input_values['stations']}</span> stations will be built.<br><br> "  # Two line breaks after stations
                f"The line will be located in a city within  <b><span style='color:orange;'>{input_values['sub_region']}</span></b> that has a population of "
                f"<b><span style='color:orange;'>{input_values['city_size']}</span></b> and is "
                f"<b><span style='color:orange;'>{input_values['city_density_type']}</span></b>. "
                f"This <span style='color:orange;'>{input_values['elevation_class']} </span>city experiences "
                f"<b><span style='color:orange;'>{precipitation_str}</span></b> and the temperature is often "
                f"<b><span style='color:orange;'>{temperature_str}</span></b>. The underlying soil is "
                f"<b><span style='color:orange;'>{input_values['soil_type']}</span></b>.<br><br> "
                f"The surrounding country is typically <b><span style='color:orange;'>{input_values['country_income_class']}</span> and has <span style='color:orange;'>{input_values['poverty_rate']}</span></b>. "
                f"</span>")


    markdown_lines = [paragraph]
    # Join the lines with line breaks to form a multi-line markdown string
    markdown_content = "<br>".join(markdown_lines)
    
    # Display the markdown content in the sidebar
    st.sidebar.markdown(markdown_content, unsafe_allow_html=True)

    ### Creating Logged versions of the user inputted features
    features_to_transform = ['tunnel', 'at_grade', 'elevated', 'duration', 'stations']
    for feature in features_to_transform:
        original_feature_name = feature  # This is the user input feature name
        transformed_feature_name = f'{feature}_transformed'  # This is the transformed feature name used in your model
        if original_feature_name in cont_input_values and transformed_feature_name in lambdas_dict:
            lambda_value = lambdas_dict[transformed_feature_name]
            # Apply the Box-Cox transformation to the user input
            cont_input_values[transformed_feature_name] = boxcox(cont_input_values[original_feature_name] + 1, lambda_value)
            del cont_input_values[original_feature_name]  # Optionally remove the original feature key if not needed anymore
    

    # for term in interaction_terms:
    #     cont_input_values[term] = 0

    # cont_input_values['tunnel_MRT_interaction'] = 0
    # cont_input_values['at_grade_MRT_interaction'] = 0
    # cont_input_values['tunnel_asia_interaction'] = 0
    # cont_input_values['at_grade_asia_interaction'] = 0


    cont_feats = ['end_year','at_grade_transformed','tunnel_transformed',
              'elevated_transformed', 'duration_transformed','stations_transformed',
              'tunnel_MRT_interaction','tunnel_asia_interaction','at_grade_MRT_interaction','at_grade_asia_interaction','stations_Streetcar_interaction',
              'stations_LightRail_interaction', 'stations_MRT_interaction','stations_tunnel_interaction',
              'stations_atgrade_interaction','stations_elevated_interaction','duration_tunnel_interaction',
              'duration_atgrade_interaction', 'duration_elevated_interaction',
              'extension_tunnel_interaction', 'extension_atgrade_interaction','extension_elevated_interaction']

    interaction_terms = [feat for feat in cont_feats if '_interaction' in feat]
    for term in interaction_terms:
        cont_input_values[term] = 0
   
   
    #### Generating values for interaction terms
    if 'train_type' in cat_input_values and cat_input_values['train_type'] == 'MRT':
        cont_input_values['tunnel_MRT_interaction'] = cont_input_values.get('tunnel_transformed', 0)
        cont_input_values['at_grade_MRT_interaction'] = cont_input_values.get('at_grade_transformed', 0)
        cont_input_values['stations_MRT_interaction'] = cont_input_values.get('stations_transformed', 0)

    if 'train_type' in cat_input_values and cat_input_values['train_type'] == 'Light Rail':
        cont_input_values['stations_LightRail_interaction'] = cont_input_values.get('stations_transformed', 0)
    if 'train_type' in cat_input_values and cat_input_values['train_type'] == 'Streetcar':
            cont_input_values['stations_Streetcar_interaction'] = cont_input_values.get('stations_transformed', 0)
    if 'train_type' in cat_input_values and cat_input_values['train_type'] == 'Monorail/APM':
            cont_input_values['stations_Monorail_interaction'] = cont_input_values.get('stations_transformed', 0)

    if 'region' in cat_input_values and cat_input_values['region'] == 'Asia':
        cont_input_values['tunnel_asia_interaction'] = cont_input_values.get('tunnel_transformed', 0)
        cont_input_values['at_grade_asia_interaction'] = cont_input_values.get('at_grade_transformed', 0)
    
    if 'project_type' in cat_input_values and cat_input_values['project_type'] == 'Extension':
        cont_input_values['extension_tunnel_interaction'] = cont_input_values.get('tunnel_transformed', 0)
        cont_input_values['extension_atgrade_interaction'] = cont_input_values.get('at_grade_transformed', 0)
        cont_input_values['extension_elevated_interaction'] = cont_input_values.get('elevated_transformed', 0)


    cont_input_values['duration_tunnel_interaction'] = cont_input_values['duration_transformed'] * cont_input_values['tunnel_transformed']
    cont_input_values['duration_atgrade_interaction'] = cont_input_values['duration_transformed'] * cont_input_values['at_grade_transformed']
    cont_input_values['duration_elevated_interaction'] = cont_input_values['duration_transformed'] * cont_input_values['elevated_transformed']
    
    ### Predictions Button
    col1, col2, col3 = st.sidebar.columns([1,2,1])
    with col2:
        # Make Predictions
        if st.button('Make Prediction', key="actualButton", help="Click to make a prediction",type='primary',use_container_width=False):
            input_values = {**cont_input_values, **cat_input_values}
            input_values.pop('length', None)
            model_feature_order = df_user.drop(columns=['cost_real_2023_transformed']).columns
        
            data_for_prediction = [input_values[key] for key in model_feature_order]
            
            # Convert the list to a DataFrame & make predictions
            df_for_prediction = pd.DataFrame([data_for_prediction], columns=model_feature_order)
            prediction = model.predict(df_for_prediction)
        
            
            # Calculate Error for Prediction
            user_length = st.session_state.length
            user_length = max(0, min(user_length, 25))
            subset_predictions = predictions[predictions['length'] <= user_length]
            subset_predictions['absolute_error'] = (subset_predictions['cost_real_2023_transformed'] - subset_predictions['prediction_label']).abs()
            subset_mae = subset_predictions['absolute_error'].mean()
            formatted_subset_mae = "{:.0f}".format(subset_mae)

            # Format the prediction output
            lambda_prediction = lambdas_dict['cost_real_2023_transformed']
            predicted_transformed_value = model.predict(df_for_prediction)
            predicted_value = inv_boxcox(predicted_transformed_value[0], lambda_prediction)
            conversion_rate = currency_conversion_rates[selected_currency]
            predicted_value_in_selected_currency = predicted_value * conversion_rate

            # Format the converted prediction for display
            if predicted_value_in_selected_currency >= 1_000_000:  # Greater than or equal to 1 trillion in the selected currency
                display_value_converted = f"{predicted_value_in_selected_currency/1_000_000:.2f}T {selected_currency}"
            elif predicted_value_in_selected_currency >= 1000:  # Greater than or equal to 1 billion but less than 1 trillion in the selected currency
                display_value_converted = f"{predicted_value_in_selected_currency/1000:.2f}B {selected_currency}"
            else:  # Less than 1 billion in the selected currency
                display_value_converted = f"{predicted_value_in_selected_currency:.2f} Million {selected_currency}"
            # Displaying the prediction and additional information in the sidebar
            st.sidebar.markdown(f"<div style='text-align: center; font-size: 30px;'>Predicted Cost</div>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<div style='text-align: center; font-size: 25px; color: orange;'>{display_value_converted}</div>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<div style='text-align: center; font-size: 12px;'>±${formatted_subset_mae}M USD (2023)</div>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<div style='text-align: center; font-size: 20px;'>   </div>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<div style='text-align: center; font-size: 20px;'> To build this project today</div>", unsafe_allow_html=True)


        ### END CODE


        #### TO DO LIST FROM REDDIT COMMENTS
            # -  XXX Currency Converter
            # - XXX More clear UI for predictions (change name of model sidebar name, add link to predictions at top of first page)
            # - XXX Maybe highlight the model name in yellow to show to click there
            # - global average line on bar plot 2nd section not labelled
            # - summary of model results includes many user models results
        ####