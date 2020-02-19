# Seattle Real Estate Analysis

## Purpose
Build a model to predict the probability that a building on any particular real estate parcel will be demolished in order to redevelop the land.

An obvious business case for this model would be for a developer to identify properties that they could redevelop. This model is limited in that it does not consider if a particular property is currently available for purchase, nor is it capable of determining the value of potential new buildings.

The question I was working with throughout the project was from the perspective of someone looking to move into a new neighborhood:

`What is the probability that I will be moving into a construction zone for the near future?`

## Data Sources
All data is publically available from 
- [King County GIS](https://gis-kingcounty.opendata.arcgis.com/search?tags=property_OpenData)
- [King County Department of Assessments](https://info.kingcounty.gov/assessor/DataDownload/default.aspx)
- [Seattle Department of Construction and Inspections](https://data.seattle.gov/Permitting/Building-Permits/76t5-zqzr)

I read all of the information into a Pandas DataFrame, but by the time I merged all of the tables and created category columns, the data was too large for my personal laptop. I could have limited the data to reduce size, but instead decided to use this as an opportunity to learn Apache Spark.

Each row is a parcel as defined by its PIN (Parcel ID Number), and I limited the scope to the City of Seattle (as opposed to all of King County).

The rows include information about the actual property (physical features, size, views, etc.), the building (square footage, number of bedrooms/bathrooms, year built, etc.), and assessment information (value of land, value of property).

The only feature engineering I did was calculating the percentage of the assessed value that was in the property compared to the property and improvements and the assessed value per square foot of building. In both cases, my prediction was that a low value building on an expensive piece of real estate would result in an increased probability that the property would flip. This was correct, and my ROC results improved by 4 points after adding these features.

## Methodology
The only information available on the public websites was for the current structure, i.e. I was not able to access information about building that had been torn down in the past a redeveloped. Therefore, I had an unbalanced dataset with only houses with demolition permits applied for or issued, but not actually executed. At the time that I built the model, about 1% of the homes in Seattle met this criteria.

I built a random forest model to predict the probability that a specific parcel would be redeveloped. I am interested in each property in the city, so I could not simply create a train/test split. Instead, I used a variation of K-Fold cross validation.
For training purposes, I assigned a probability of 1 for parcels for which a demo permit had been applied, and a 0 for all others. I split all of the parcels into ten different folds, and would train the model ten times, each time with one fold being left out to make the actual prediction. Because the folds were randomly assigned, in each case the distribution of 'ones' and 'zeroes' were similar to that of the full dataset.

## Results

It was a soft classifier, and the predicted probabilities ranged from .8% to 11.6%, with a mean of 10.2%.

Below is the distribution of probabilities, separated by whether or not the property is scheduled for redevelopment.

![Histogram](https://github.com/frischmuth/Seattle-Real-Estate-Analysis/blob/master/Histogram.png "Histogram")

Below is the ROC Curve. The area under the curve is 0.73.
![ROC](https://github.com/frischmuth/Seattle-Real-Estate-Analysis/blob/master/ROC.png "ROC")

## Next Steps

- Perform addition feature engineering and also feature analysis to further determine what parcel charactaristics are most indicative of redevelopment.
- Create an interface where a user could search for individual properties to see how likely that property or any of the neighboring properties are to be redeveloped. Continuing that, it would also be interesting to create a heat map of the city to see where construction is happening.
- As time elapses, the model could be given new information based on new demolition permits people have applied for. This could be used both for validation purposes and to improve the model results.
- Access historical data of previous buildings that is not available online.

