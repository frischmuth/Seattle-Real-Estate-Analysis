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



This model is built to predict the probability that an area of the city will likely be an active construction zone for the near future.

Original Proposed Methodology:
Find houses that have been torn down and compare them to current houses. Higher similarity would indicate higher likelihood of redevelopment.
The available city data only includes current structures, so this will not work.

New Proposed Methodology (10/15, probably already changed):
Use open building permits to identify houses that are currently undergoing the review process for demolition.
Feature engineer to pull neighbor information, neighbors defined as sharing same Major parcel ID value.
Feature engineer to try and find characteristics that might make a property appealing - value, lot size, age, how it compares to neighbors, etc
Look at if there has already been redevelopment (compare mean year of construction to recent construction) - percentage built within recent timeframe could be added as a feature
Look at open demolition permits
Find way to compare parcel groups?
Do Stratified K Fold Validation
Assign probabilities that a parcel might be redeveloped
Calculate construction score as some combination of neighbors' probabilities

