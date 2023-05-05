import pandas as pd
import numpy as np
import folium 
import requests
import json
import time
import seaborn as sns
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import random
from tqdm import tqdm
from iteration_utilities import deepflatten

URL = "https://listings.offcampusliving.cornell.edu/api/listingById?listingId="
DROP_COLS = [
    'Unnamed: 0',
    'PaymentStatus',
    'LocationConfirmed',
    'Active',
    'active',
    'LastUpdated',
    'lastUpdated', 
    'description', 
    'LongDescription',
    'status',
    'dateLastUpdated', 
    'SatisfiesApplicableCode',
    'ListingComplete',
    'safetyRatingAddress',
    'SafetyRatings',
    'safetyRatings',
    'SafetyRatings.BuildingAddress',
    'SafetyRatings.CategoryScores',
    'categoryScores',
    'categoryScoresAsPercentage',
    'SafetyRatings.CategoryScoresAsPercentage',
    'SafetyRatings.DateLastUpdated', 
    'categoryScores', 
    'SafetyRatings.OverallSafetyRatingAsPercentage', 
    'overallSafetyRatingAsPercentage',
    'OverallSafetyRatingAsPercentage',
    'categoryScoresAsPercentage', 
    'SafetyRatings.CategoryScoresAsPercentage', 
    'SafetyRatings.SprinklerType',
    'sprinklerType',
    'fireExtinguisherCertificateExpiration',
    'SafetyRatings.FireExtinguisherCertificateExpiration',
    'SafetyRatings.NotificationSystem',
    'SafetyRatings.CertificateExpirationDate',
    'certificateExpirationDate',
    'notificationSystem', 
    'hasValidCertificateOfOccupancy',
    'SafetyRatings.HasValidCertificateOfOccupancy',
    'accountId',
    'AccountId',
    'contactPhone', 
    'ContactPhone', 
    'contactEmail', 
    'ContactEmail', 
    'paymentStatus',
    'lastUpdated', 
    'firstName',
    'lastName',
    'emailAddress',
    'phoneNumber',
    'PhoneNumber',
    'password', 
    'Password',
    'accountType',
    'AccountType',
    'lastLogin', 
    'isStudent', 
    'fraudulent', 
    'Coordinates.lat',
    'Coordinates.lng',
    'dateTimeListed',
    'SafetyRatings.SatisfiesApplicableCode'
]
 
def extract(url, ids): 
    s = requests.Session()
    df = pd.DataFrame()
    for i in tqdm(ids):
        time.sleep(random.uniform(0.05, 0.25))
        r = s.get(f"""https://listings.offcampusliving.cornell.edu/api/listingById?listingId={i}""")
        json = r.json()
        if not json is None: 
            new_df = pd.json_normalize(json, max_level = 1)
            df = pd.concat([df, new_df], ignore_index=True)
    return df
    
def find_valid_ids(url, ids): 
    s = requests.Session()
    df = pd.DataFrame()
    valid_ids = []
    for i in tqdm(ids):
        time.sleep(random.uniform(0.05, 0.1))
        r = s.get(f"""https://listings.offcampusliving.cornell.edu/api/listingById?listingId={i}""")
        json = r.json()
        if not json is None: 
            valid_ids.append(i)
    pd.Series(valid_ids).to_csv("data/valid_ids.csv", index=False)
    return valid_ids

def download_photos(df):
    # works for 2023 data only
    photo_urls = []
    for i, listing_id in tqdm(enumerate(df.listing_id.unique())):
        for j in range(len(df.iloc[i].listing_photos)):
            if len(df.iloc[i].listing_photos) > 0: 
                time.sleep(random.uniform(0.1, 0.2))
                url = df.iloc[i].listing_photos[j]['PhotoUrl']
                img_data = requests.get(url).content
                with open(f'''photos/{listing_id}_{j}.{url.split(".")[-1]}''', 'wb') as handler:
                    handler.write(img_data)
    return photo_urls
                    
def load_data(filepath="data/raw_scrape_oct_25_2021.csv"):  
    """
    param filepath: str of filename (including '.csv')
    rtype: pd.DataFrame
    """
    df = pd.read_csv(filepath)
    return df

# Using haversine function, google map data to compute a distance to McGraw Clocktower (a proxy for campus)
# for each of these listings, we want to compute the haversine dist from the listing to the clocktower. 

def haversine_distance(pt1, pt2):
    """
    Using haversine function and google map data, computes a distance to
    McGraw Clocktower 

    type pt1: tuple 
    type pt2: tuple
    rtype: float (m)
    """
    lat1,lon1 = pt1
    lat2,lon2 = pt2
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # haversine formula
    a = sin((lat2 - lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1)/2)**2
    c = 2 * asin(sqrt(a))
    return float(6371000*c)

# McGraw Clocktower (lat,long): 42.44764589643703, -76.48496728637923
def add_distance_to_campus(df, lat=42.44764589643703, long=-76.48496728637923): 
    """
    Uses haversine_distance to calculate and record distance_to_campus (in meters) for each listing in DataFrame
    """
    distances = []
    for index, row in df.iterrows(): 
        listing_latitude = row["gmap_latitude"]
        listing_longitude = row["gmap_longitude"]
        distances.append((haversine_distance((listing_latitude,listing_longitude),((lat,long)))))
    df["distance_to_campus"] = distances
    df["distance_to_campus"] = df["distance_to_campus"].astype(float)
    return df

def add_price_per_room(df):
    """
    """
    df["rent_amount"] = df["rent_amount"].astype(float)
    df["price_per_room"] = df.apply(lambda row: row.rent_amount / max(1, row.num_bedrooms) if row.rent_type.lower() == "price per unit" else row.rent_amount, axis=1)
    return df 

def add_price_per_unit(df):
    """
    """
    df["price_per_unit"] = df.apply(lambda row: row.rent_amount * max(1, row.num_bedrooms) if row.rent_type.lower() == "price per person" else row.rent_amount, axis=1)
    return df

def convert_to_snake_case(column_name):
    """
    Returns: snake_case version of a given column name
    
    param column_name: 
    type column_name: 
    rtype: str
    """
    return "".join(["_"+value.lower() if value.isupper() else value for value in column_name]).lstrip("_")


def flatten_amenities(df):
    """
    Adds binary columns for every possible amenity, sets to 1 for a given unit if unit includes that amenity
    """
    potential_amenities = {'Air Conditioning' : 'air_conditioning'
                           ,'Electricity Included' : 'electricity_included'
                           ,'Electronic Payments Accepted ': 'electronic_payments'
                           ,'Furnished' : 'furniture_included'
                           ,'Heat Included': 'heat_included'
                           ,'Internet Included' : 'internet_included'
                           ,'Kitchen': 'kitchen_available'
                           ,'Laundry Facilities' : 'laundry_available'
                           ,'Near Bus Route' : 'near_bus'
                           ,'Off-Street Parking Available' : 'osparking_available'
                           ,'Off-Street Parking Included' : 'osparking_included'
                           ,'Permitted Street Parking Available' : 'psparking_available'
                           ,'Water Included' : 'water_included'} 
    amenities_list = [i for i in potential_amenities.keys()]
    for i in amenities_list: 
        df[potential_amenities[i]] = df["amenities"].apply(lambda x: 1 if i in x else 0)  
    df = df.drop(columns = "amenities", inplace=False)
    return df


def flatten_listing_types(df):
    """
    Adds binary columns for every possible listing type, sets to 1 for a given unit if unit is of that listing type  
    """
    potential_listing_types = {'Graduate' : 'graduate',
                 'Staff/Faculty' : 'faculty',
                 'Undergraduate' : 'ugrad', 
                 'Sabbatic Housing' : 'sabbatical',
                 'Fall Sublet' : 'fa_sublet',
                 'Spring Sublet' : 'sp_sublet', 
                 'Summer Sublet' : 'sum_sublet',
                 'Short-Term Housing' : 'short_term'}
    listing_types_list = [i for i in potential_listing_types.keys()]
    for i in listing_types_list:
        df[potential_listing_types[i]] = df["listing_types"].apply(lambda x: 1 if i in x else 0)
    df = df.drop(columns = "listing_types", inplace=False)
    return df 

def get_column_rename_mapping(): 
    """
    """
    rename_mapping = {"air_conditioning":"amenity_air_conditioning"
                      ,"bathrooms":"num_bathrooms"
                      ,"bedrooms":"num_bedrooms"
                      ,"create_date":"created_at"
                      ,"date_available":"available_at"
                      ,"electronic_payments":"electronic_payments_allowed"
                      ,"faculty":"market_faculty"
                      ,"graduate":"market_graduate"
                      ,"kitchen_available":"amenity_kitchen_available"
                      ,"laundry_available":"amenity_laundry_available"
                      ,"length_available":"months_available_for"
                      ,"listing_address":"address"
                      ,"listing_city":"city"
                      ,"listing_expiration_date":"expires_at"
                      ,"listing_zip":"zip"
                      ,"near_bus":"is_near_bus"
                      ,"water_included":"includes_water"
                      ,"osparking_available":"amenity_offstreet_parking_available"
                      ,"osparking_included":"includes_offstreet_parking"
                      ,"electricity_included":"includes_electricity"
                      ,"furniture_included":"includes_furniture"
                      ,"heat_included":"includes_heat"
                      ,"internet_included":"includes_internet"
                      ,"pets":"pets_allowed"
                      ,"price_per_room":"price_per_room"
                      ,"price_per_unit":"price_per_unit"
                      ,"sabbatical":"market_sabbatical"
                      ,"fa_sublet":"market_fall_sublet"
                      ,"sp_sublet":"market_spring_sublet"
                      ,"sum_sublet":"market_summer_sublet"
                      ,"ugrad":"market_ugrad"
                      ,"graduate":"market_grad"
                      ,"short_term":"market_short_term"
                      ,"water_included":"includes_water"
                      ,"safety_ratings._overall_safety_rating":"overall_safety_rating"
                      ,"safety_ratings._has_fire_resistant_construction_type":"has_fire_resistant_construction_type"
                      ,"safety_ratings._meets_minimum_requirements":"meets_minimum_requirements"
                      ,"safety_ratings._exceeds_requirements":"exceeds_requirements"
                     }
    return rename_mapping 

# Setting datatypes
def set_dtypes(df): 
    """
    Sets the datatype of each column in the given DataFrame to ensure standardization 
    """
    # Floats
    df["price_per_unit"] = df["price_per_unit"].astype(float)
    df["price_per_room"] = df["price_per_room"].astype(float)
    df["gmap_latitude"] = df['gmap_latitude'].astype(float)
    df["gmap_longitude"] = df['gmap_longitude'].astype(float)
    df["num_bedrooms"] = df["num_bedrooms"].astype(float)
    df["num_bathrooms"] = df["num_bathrooms"].astype(float)
    # Datetimes
    df["created_at"] =  pd.to_datetime(df["created_at"])
    df["available_at"] =  pd.to_datetime(df["available_at"])
    # Dummy Variables (bools as ints) 
    df['meets_minimum_requirements'] = df['meets_minimum_requirements'].astype(int)
    df['has_fire_resistant_construction_type'] = df['has_fire_resistant_construction_type'].astype(int)
    df['exceeds_requirements'] = df['exceeds_requirements'].astype(int)
    df["pets_allowed"] = df["pets_allowed"].astype(int)
    df['amenity_air_conditioning'] = df['amenity_air_conditioning'].astype(int)
    df['includes_electricity'] = df['includes_electricity'].astype(int)
    df['includes_furniture'] = df['includes_furniture'].astype(int)
    df['includes_heat'] = df['includes_heat'].astype(int)
    df['includes_internet'] = df['includes_internet'].astype(int)
    df['amenity_kitchen_available'] = df['amenity_kitchen_available'].astype(int)
    df['amenity_laundry_available'] = df['amenity_laundry_available'].astype(int)
    df['includes_water'] = df['includes_water'].astype(int)
    df['is_near_bus'] = df['is_near_bus'].astype(int)
    df['amenity_offstreet_parking_available'] = df['amenity_offstreet_parking_available'].astype(int)
    df['includes_offstreet_parking'] = df['includes_offstreet_parking'].astype(int)
    df['market_faculty'] = df['market_faculty'].astype(int)
    df['market_ugrad'] = df['market_ugrad'].astype(int)
    df['market_sabbatical'] = df['market_sabbatical'].astype(int)
    df['market_fall_sublet'] = df['market_fall_sublet'].astype(int)
    df['market_spring_sublet'] = df['market_spring_sublet'].astype(int)
    df['market_summer_sublet'] = df['market_summer_sublet'].astype(int)
    df['market_short_term'] = df['market_short_term'].astype(int)
    df['market_grad'] = df['market_grad'].astype(int)
    df['is_single_room'] = df['is_single_room'].astype(int)
    # Regular ints
    df['zip'] = df['zip'].astype(int)
    df['months_available_for'] = df['months_available_for'].astype(int)
    df['listing_id'] = df['listing_id'].astype(int)
    return df 

def remap_to_int(df):
    """
    rtype: DataFrame
    """
    df["pets_allowed"] = df["pets_allowed"].apply(lambda x: 1 if x=="Yes" else 0).astype(int)
    df["num_bedrooms"] = df["num_bedrooms"].apply(lambda x: 0 if x=="studio" else x)
    df["num_bedrooms"] = df["num_bedrooms"].astype(float)
    df["has_gender_preference"] = df["gender_preference"].apply(lambda x: 0 if x=="No Preference" else 1)
    df["has_female_gender_preference"] = df["gender_preference"].apply(lambda x: 1 if x=="Female" else 0)
    df["has_male_gender_preference"] = df["gender_preference"].apply(lambda x: 1 if x=="Male" else 0)
    df["is_single_room"] = df["housing_type"].apply(lambda x: 1 if x=="Room to Rent" else 0)
    df['meets_minimum_requirements'] = df['meets_minimum_requirements'].fillna(0)
    df['has_fire_resistant_construction_type'] = df['has_fire_resistant_construction_type'].fillna(0)
    df['exceeds_requirements'] = df['exceeds_requirements'].fillna(0)
    return df

def add_uuid(df):
    df['uuid'] = df.listing_id.astype(str)+'-'+df.available_at.dt.year.astype(str)+'-'+df.available_at.dt.month.astype(str)
    return df 

def add_numerical_dates(df):
    df['year'] = df.available_at.dt.year.astype(int)
    df['month'] = df.available_at.dt.month.astype(int)
    return df

def remove_errant_rows(df): 
    """
    Removes errant values from the DataFrame
    """
    # Entries with 'test' unit number (presumably for use by site admins)
    df = df.loc[df.unit_number != 'test']
    # Entries without a listing price
    df = df.loc[df.price_per_room > 10]
    # Entries without valid latitude/longitude
    df = df.loc[df["gmap_latitude"].notna()]
    df = df.loc[df['gmap_longitude'].notna()]
    df = df.loc[df['months_available_for'].notna()]
    return df

def remove_implausible_rows(df): 
    """
    Removes implausible values from the DataFrame
    """
    # Entries outside 5 kilo|meter radius from Clocktower
    df = df.loc[df.distance_to_campus <= 5000]
    df = df.loc[df.num_bathrooms >= 0]
    df = df.loc[df.num_bedrooms >= 0]
    return df 

def transform(data):
    """
    Transforms a given df 
    
    param data: dataframe to transform
    type data: pandas.Dataframe
    rtype: pandas.Dataframe
    """
    df = (
        data
        .drop(columns = [col for col in data if col in DROP_COLS])
        .rename(mapper=convert_to_snake_case, axis=1)
        .rename(columns=get_column_rename_mapping())
        .pipe(remap_to_int)
        .pipe(flatten_amenities)
        .pipe(flatten_listing_types)
        .pipe(add_distance_to_campus)
        .pipe(add_price_per_room)
        .pipe(add_price_per_unit)
        .rename(columns=get_column_rename_mapping())
        .pipe(remove_errant_rows)
        .pipe(set_dtypes)
        .pipe(add_numerical_dates)
        .pipe(remove_implausible_rows)
        .pipe(add_uuid)
    )
    return df 

def convert_meters_to_feet(value, unit_from="meters", unit_to="feet", 
                           output_type=float, prefix='', factor_power=None):
    """
    A function to convert (prefix)meters to feet of output_type
   
    param value: the measurement in (prefix)meters to convert
    type value: float
    param unit_from (optional): one of "meters", "feet". Defaults to "meters"
    type unit_from: str
    param unit_to (optional): one of "meters", "feet". Defaults to "feet"
    type unit_to: str
    param prefix (optional): a valid International System of Units metric prefix.
        (i.e. "kilo"), see wikipedia.org/wiki/Metric_prefix. Case Insensitive. Defaults to ''
    type prefix: str 
    param output_type (optional): desired output type. Defaults to float
    type output_type: one of str, int, float
    param factor_power (optional): the base 10 exponent of desired prefix. Defaults to None 
    type factor_power: int
 
    rtype: output_type 
    """
    possible_prefixes={"yotta":24,"zetta":21,"exa":18,"peta":15,"tera":12,"giga":9,"mega":6,"kilo":3,"hecto":2,"deca":1,
                       "":0,"deci":-1,"centi":-2,"milli":-3,"micro":-6,"nano":-9,"pico":-12,"femto":-15,"atto":-18,"zepto":-21,"yocto":-24}
    possible_conversion_factors = {"meters_to_feet":3.28, "feet_to_meters":0.3048, "feet_to_feet":1, "meters_to_metters":1}
    conversion_factors_key = unit_from+"_to_"+unit_to
    if factor_power is None:
        factor_power = possible_prefixes[prefix.lower()]
    value_scaled_to_meter = value*(10**factor_power)
    conversion_factor = possible_conversion_factors[conversion_factors_key]
    output = value_scaled_to_meter*conversion_factor
    return output_type(output)

def apply_filter(df): 
    """
    """
    df = df.loc[df.market_spring_sublet == 0]
    df = df.loc[df.market_fall_sublet == 0]
    df = df.loc[df.market_summer_sublet == 0]
    df = df.loc[df.market_short_term == 0]
    df = df.loc[df.num_bedrooms <= 20]
    df = df.loc[df.price_per_unit <= 15000]
    df = df.loc[df.distance_to_campus <= 2500]
    return df 