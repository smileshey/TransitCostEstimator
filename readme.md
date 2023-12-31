# A Predictive Approach to Passenger Rail Construction Cost Estimation
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd3Z0YWpuYzk3bnlhcGtzdmdtZmNhd2tqbWxmN3AzYmJ2eHVkdzUzeSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/N26oFFY8In8JxBfQrk/source.gif" width="1000" height="100" alt="Alt Text" style="object-fit: cover;">

## Problem Statement

Efficient public transit systems play a pivotal role in enhancing mobility, reducing carbon footprints, and fostering sustainable city growth. In many American cities, there is an ongoing and concerted effort to improve multi-modal transit and build better transit systems that supplement, or replace, the predominantly car-oriented infrastructure. Often these transit system improvements are subject to scrutiny, as urban rail projects require an extensive up-front investment of public money.
Within many transit agencies financial constraints exist and officials are often hesitant to allocate significant public funds for long-term projects. This hesitancy is intensified by the potential for unexpected expenses that could jeopardize an entire project. There's a pressing need for a transparent tool that provides accurate cost assessments for urban rail projects that doesn't rely on the participation of government officials.

This analysis aims to provide community members, local officials, and advocates with realistic construction cost estimates for passenger rail projects, tailored to the specific constraints of individual communities. Such a tool could help optimize public fund allocation and bridge the gap between political decision-making and community needs.

The objective of this analysis was to examine existing data on the construction costs of train lines to develop a predictive model capable of estimating the total cost of such projects. The endeavor produced two primary outputs:

1. A specialized model intended for professionals, such as engineers, familiar with particular locales.
2. A general-use model, designed for anyone keen on gauging transit project expenses.

By doing so, I hope to contribute to the creation of more efficient, timely, and cost-effective transit projects that better serve the needs of urban populations globally. There are several important metrics to track and within this sheet I will outline the data that will be used in the analysis.


## The Data

The basis for this analysis will be the data collected and organized by the [Transit Costs Project](https://transitcosts.com/) (TCP) which is affiliated with NYU Marron Institute of Urban Management. The Transit Costs Project has provided their own analysis of the data, which can be found on their [analysis page](https://transitcosts.com/new-data/). I intend to build upon their analysis to build a tool that provide a baseline cost estimate to help estimate the overall cost for a project, given some information about the project area.

The final dataset used in this project will be a modified version of the dataset discussed above. Within the TCP's dataset, there were approximately 150 rows with missing values. The original research team, for the purposes of compiling a trustworthy dataset, left several items blank if they could not verify their true values from official sources. I opted to backfill these datapoints by using a variety of techniques that, I feel, provided me with a suitable approximation. It's worth noting that many of these techniques are imperfect and should be viewed as a potential sources of errors. I'll discuss the techniques used in the the data cleaning section of this analysis.

The Transit Project data includes several important features that will be used in my analysis, however those features are primarily related to the physical attributes of the railway themselves. In addition to this data, I intend to use the provided location for each project to produce several relevant features that pertain to the specific site conditions. Below, I'll outline the existing features and their purpose.

### The Features

| Feature                     | Unit            | Description                                                                                   |
|-----------------------------|-----------------|-----------------------------------------------------------------------------------------------|
| ID                          | -               | A unique identifier for each record in the dataset.                                           |
| Country                     | -               | The country where the transit project is located.                                             |
| City                        | -               | The city where the transit project is located.                                                |
| Line                        | -               | The name or identifier of the transit line within the city.                                   |
| Phase                       | -               | The phase of the transit project (e.g., Phase 1, Phase 2, etc.).                              |
| Start year                  | Year            | The year in which the transit project construction started.                                   |
| End year                    | Year            | The year in which the transit project construction was completed.                             |
| RR?                         | -               | A binary indicator (Yes/No) for whether the transit line is a rapid transit or not.           |
| Length                      | Kilometers/Miles | The total length of the transit line.                                                         |
| TunnelPer                   | Percentage (%)   | The percentage of the transit line that runs underground in tunnels.                          |
| Tunnel                      | Kilometers/Miles | The length of the transit line that runs underground in tunnels.                              |
| Elevated                    | Kilometers/Miles | The length of the transit line that is elevated above ground level.                           |
| Atgrade                     | Kilometers/Miles | The length of the transit line that is at ground level (at-grade).                            |
| Stations                    | Count           | The total number of stations on the transit line.                                             |
| Platform Length    | Meters          | The average length of platforms at stations.                                                  |
| Source1                     | -               | The source or reference from which the data was obtained.                                    |
| Cost                        | Currency        | The cost of the transit project in the original currency.                                     |
| Currency                    | -               | The currency in which the cost is specified.                                                  |
| Year                        | Year            | The year in which the cost value was recorded.                                                |
| PPP rate                    | -               | The Purchasing Power Parity (PPP) rate for converting the cost to a common currency.         |
| Real cost                   | Currency        | The adjusted cost of the transit project, considering the PPP rate and inflation.            |
| Cost/km         | Millions/km        | The cost of the transit project per kilometer.                                               |
| Cheap?                      | -               | A binary indicator (Yes/No) for whether the transit project is considered cheap or not.      |
| Clength                     | Millions        | The cost of the transit project per kilometer for the length of the transit line.            |
| Ctunnel                     | Millions        | The cost of the transit project per kilometer for the tunnel portion.                         |
| Anglo?                      | -               | A binary indicator (Yes/No) for whether the transit project is located in an Anglophone country. |
| Inflation Index             | -               | The inflation index for adjusting the cost to real value.                                     |
| Real cost    | 2021 dollars    | The adjusted cost of the transit project in 2021 dollars.                                    |
| Cost/km      | Millions        | The cost of the transit project per kilometer in 2021 dollars.                               |
| Source2                     | -               | Additional source or reference for the data.                                                  |
| Reference                   | -               | Any additional reference information related to the transit project.                         |


## Results

The purpose of this analysis was to evaluate available data pertaining to the construction of Metro systems around the world and to produce a model that can estimate the cost for future Metro projects, given some data about the project. The resulting model is an successful first iteration towards achieving this goal.

The finalized user model achieves both goals set at the outset of this analysis as it is both 1) relatively accurate and 2) easily understandable.

The accuracy achieved by the model with a mean absolute error of 480.95M USD and an R-squared of .896 is sufficient for the purposes of creating a user focused model that can estimate the potential cost for a project in a given area. Addtionally, the model is far more accurate for projects that have a length of less than 25km, which are more commonly constructed. 

This model, which is demoed below, is [deployed and available for anyone to use on Streamlit](https://buildmoretrains.streamlit.app/)

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd3Z6N250OWk4NHlyZWk3MmY0eTA5bDFudzFxaGRnMThldHdkenU3cCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/nWZYYOectRB9Sgf1dN/giphy.gif">
</p>



### Using the Model

The basis for the model shown above is to provide users with a quick and simple(r) way to create a cost estimate. Instead of needing to input specific values for a city's population or exact figures for poverty rates in a specific region, the deployed model requires generalized descriptions of the area that are easier to input without needing to do significant research.

The features used in this user model are described below:

| Feature               | Description                                         |
|-----------------------|-----------------------------------------------------|
| length                | Total length of the rail line (kilometers) |
| tunnel                | Length of track that is tunneled (km)         |
| elevated              | Length of track that is elevated (km)        |
| at_grade              | Length of track that is at ground level (km)        |
| stations              | Number of stations along the route                  |
| anglo?                | Is the area predominantly Anglo-Saxon? (yes/no)                             |
| cost_real_2021        | Estimated cost for construction in 2021             |
| duration              | Time taken for construction (years)       |
| region                | Geographical region (e.g., North America, Asia)     |
| sub_region            | Subdivision within the main region                  |
| soil_type             | Type of soil the rail is built on                   |
| gauge_width           | Width between the two rails (e.g., standard, narrow, monorail) |
| city_size             | Population of the city       |
| train_type            | Type of train (subway, tram, monorail)                |
| country_income_class  | Income classification of the country                |
| elevation_class       | Altitude classification                             |
| precipitation_type    | Normal Expected precipitation in the region          |
| temperature_category  | Normal expected climate                           |
| affordability         | Relative affordability of the project               |
| union_prevalence      | Presence and influence of worker unions             |
| poverty_rate          | Percentage of population below the poverty line     |
| city_density_type     | Population density classification of the city       |
| country_density_type  | Population density classification of the country    |

By inputting the features described above into the model deployed on Streamlit, a cost estimate will be provided by the model. This cost estimate is in 2021 USD and provides you with a rough idea for how expensive it would be to build the described project in the described city.


### Resources

Within the analysis I've summarized above, I relied heavily on several external sources. Each source I list below was invaluable and I'm gratful for their contributions to the world.

1. [The Transit Project](https://transitcosts.com/about/)
2. [Pedestrian Observations](https://pedestrianobservations.com/)
3. [PyCaret](https://pycaret.gitbook.io/docs/)
4. [Urban Rail](https://www.urbanrail.net/)
5. [Soil Grids](https://www.isric.org/explore/soilgrids)
6. [Plotly](https://plotly.com/python/)
7. [OpenMeteo](https://open-meteo.com/)
8. [Streamlit](https://www.gradio.app/)


### Open List of Potential Improvements

After submitting this project to various subreddits for input, several potentially improvements have become apparent. I'll keep a running tally of those issues below: 
1. More data. 1000 data points isn't enough to provide reliable results.
2. Stations are weighted equally for subways and trams, even though a tram station is less complex to  build.
3. Adjust inflation rates to 2023, or dynamically adjust them to a year set by the user.
4. PPP rates and inflation rates often vary considerably over the course of a project, need to devise a way to find a weighted average of both rates to prevent skewed results.

