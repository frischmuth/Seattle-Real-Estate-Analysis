The city of Seattle has various departments that provide data relevant to this project.

GIS Open Data

Parcels for King County with Address with Property Information / parcel address area
https://gis-kingcounty.opendata.arcgis.com/datasets/parcels-for-king-county-with-address-with-property-information-parcel-address-area/data?geometry=-137.543%2C43.508%2C-107.462%2C48.828
Download Full Dataset Spreadsheet
Parcels_for_King_County_with_Address_with_Property_Information__parcel_address_area.csv
This dataset is the bridge between Parcel ID Number (PIN), which is a combination of a Major and Minor ID, the Address (parsed out by address number, prefixes, street name, and suffixes)

Assessments Data Download
https://info.kingcounty.gov/assessor/DataDownload/default.aspx
https://aqua.kingcounty.gov/extranet/assessor/Parcel.zip
Parcel.zip
This dataset has parcel information that can be used to compare properties, including location, zoning, water, sewer, views, noise, waterfront status, historic status, etc. Has PIN as the unique ID. See Parcel.doc as readme (or eventually import table into this file).

https://aqua.kingcounty.gov/extranet/assessor/Tax%20Data.zip
Tax Data.zip
This dataset has 9.6M tax records with an AcctNbr as unique ID. Still needing to find a link from this to either Parcel ID or address. See Tax Data.doc as readme.


Building Permits
https://data.seattle.gov/Permitting/Building-Permits/76t5-zqzr
Building_permits.csv
This dataset has all of the city-issued permits gonig back to (...), including demolition and construction. Use this to find lots that have been redeveloped and might be in the future.

Land Use Permits
https://data.seattle.gov/Permitting/Land-Use-Permits/ht3q-kdvx
TBD, but I think it is applications for people who want to develop a parcel. Could be joined on OriginalAddress1 column. Description column could be parsed for information on what the proposed use is.




