# Seattle-Real-Estate-Analysis

This model is built to predict the probability that an area of the city will likely be an active construction zone for the near future.

Original Proposed Methodology:
Find houses that have been torn down and compare them to current houses. Higher similarity would indicate higher likelihood of redevelopment.
The available city data only includes current structures, so this will not work.

New Proposed Methodology:
Use open building permits to identify houses that are currently undergoing the review process for demolition.
Feature engineer to pull neighbor information, neighbors defined as sharing same Major parcel ID value.
Feature engineer to try and find characteristics that might make a property appealing - value, lot size, age, how it compares to neighbors, etc
Look at if there has already been redevelopment (compare mean year of construction to recent construction) - percentage built within recent timeframe could be added as a feature
Look at open demolition permits
Find way to compare parcel groups?
Do Stratified K Fold Validation
Assign probabilities that a parcel might be redeveloped
Calculate construction score as some combination of neighbors' probabilities

