data_scraping.ipynb: 

	Collect restaurant data from Yelp API. 
	The result is stored in restaurant_yelp_unique_refined.csv.
	Some adjustments have been made to the 'restaurant_yelp_unique_refined.csv' for data refinement.
	
	
model.ipynb;
	
	Use the data collected from the questionnaire to build the model.
	A model.h5 will be generated. The model file is then uploaded to the AWS S3.
	Analysis on the model training results is done at the end.
	
	
Upload_firebase_data.ipynb:

	Upload the restaurant data 'restaurant_yelp_unique_refined.csv' to the Realtime Database.
	Upload the user data collected in the questionnaire for testing purpose.
	
	
data:
	Contains the data for running the jupyter notebook

logs: 
	Contains the log file for model attempts 3,4,7,9,10,13,28


