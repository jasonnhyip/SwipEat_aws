import requests
import pandas as pd

YELP_API_KEY = "2SqvLuEwHaCfU-vvsp3xdbW1_H4CP51EieMf5FLjKdX6J8nAPSradgt5M5-w2gPLWQG5VYJKknOaQ6VmA48ik9vvZuHIbuVJm15iq11gkdUmRl30fsnDqPglpraVYnYx"


def search(latitude, longitude, radius = 1000, price = "1,2,3,4", yelpCategories = ["food"], offset = 0):
    """
    Request data from Business Search Endpoint
    Args:
        latitude (double)
        longitude (double)
        radius (int): search radius in meter
        price (str): list of price levels, e.g. "1,2,3,4"
        yelpCategories (list of str): restaurant categories
        offset (int): offset for the number of restaurants. Used when retrieving 
                      total number of restaurants that is larger than 50
    Returns:
        (dict): The JSON response from the request.
    """
    url = 'https://api.yelp.com/v3/businesses/search'

    headers = {
        'Authorization': 'Bearer {}'.format(YELP_API_KEY),
    }
    categories = ",".join(yelpCategories);
    print("search params: ", radius, price, categories)
    
    url_params = {
        'term': 'restaurants',
        'latitude': latitude,
        'longitude': longitude,
        'limit': 50,
        'radius': radius,
        'price': price,
        'categories': categories,
        'offset': offset
    }

    return requests.request('GET', url, headers=headers, params=url_params).json()


def search_nearby_restaurant(latitude, longitude, radius, price, yelpCategories):
  """
    Request data from Business Search Endpoint
    Args:
        latitude (double)
        longitude (double)
        radius (int): search radius in meter
        price (str): list of price levels, e.g. "1,2,3,4"
        yelpCategories (list of str): restaurant categories

    Returns:
        (list of dict): list of JSON response from the request.
    """
  # print("start yelp request")
  responses = []
  responses.append(search(latitude, longitude, radius=radius, price=price, yelpCategories=yelpCategories))
  
  if 'total' in responses[-1]:
    total = responses[-1]['total']
    offset = 50

    while offset < total and offset <= 50:
      responses.append(search(latitude, longitude, radius=radius, price=price, yelpCategories=yelpCategories, offset=offset))
      offset = offset + 50

  return responses


def decode_resp(responses):
  """
    Decode list of responses to dataframe
    Args:
        responses(list of dict): responses obtained from search
    Returns:
        (list of dict): list of JSON response from the request.
    """
  # print("decode responses")
  restaurant_list = []
  for resp in responses:
    for restaurant in resp['businesses']:
      restaurant_list.append(restaurant)

  labels_restaurant = ['id', 'alias', 'name', 'image_url', 'is_closed', 'url', 'review_count', 'categories',
                     'rating', 'coordinates', 'transactions', 'location', 'phone', 'display_phone', 'distance', 'price']

  return pd.DataFrame.from_records(restaurant_list, columns=labels_restaurant)