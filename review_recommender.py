# Recommendation system to calculate euclidean distance between pairs of products who have been reviewed by the same user.
# pick up original data here: https://s3.amazonaws.com/demo-datasets/beer_reviews.tar.gz
# original code from Greg Lamp: http://nbviewer.ipython.org/gist/glamp/20a18d52c539b87de2af


import pandas as pd
import numpy as np
import pylab as pl
df = pd.read_csv("beer_reviews.csv")
from sklearn.metrics.pairwise import euclidean_distances

#  fill in run requirements here:
REVIEW_FEATURES = ['review_overall', 'review_aroma', 'review_palate', 'review_taste']
products = ["Dale's Pale Ale", "Sierra Nevada Pale Ale", "Michelob Ultra",
         "Natural Light", "Bud Light", "Fat Tire Amber Ale", "Coors Light",
         "Blue Moon Belgian White", "Guinness Draught"]  # our favorite beers!


def get_product_reviews(product, common_users):
    mask = (df.review_profilename.isin(common_users)) & (df.beer_name==product)
    reviews = df[mask].sort('review_profilename')
    reviews = reviews[reviews.review_profilename.duplicated()==False]
    return reviews



def calculate_similarity(product1, product2):
    # find common reviewers
    product_1_reviewers = df[df.beer_name==product1].review_profilename.unique()
    product_2_reviewers = df[df.beer_name==product2].review_profilename.unique()
    common_reviewers = set(product_1_reviewers).intersection(product_2_reviewers)

    # get reviews
    product_1_reviews = get_product_reviews(product1, common_reviewers)
    product_2_reviews = get_product_reviews(product2, common_reviewers)
    dists = []
    for f in REVIEW_FEATURES:
        dists.append(euclidean_distances(product_1_reviews[f], product_2_reviews[f])[0][0])
    
    return dists




simple_distances = []
for product1 in products:
    print "starting distance computations for ", product1
    for product2 in products:
        if product1 != product2:
            row = [product1, product2] + calculate_similarity(product1, product2)
            simple_distances.append(row)

output_names = []
for f in REVIEW_FEATURES:
    output_name = f + "_distance"
    output_names.append([output_name])

cols = ["product1", "product2"] + output_names
simple_distances = pd.DataFrame(simple_distances, columns=cols)

#print "Tail of Results File:"
#simple_distances.tail()
product= "Coors Light"
p1 = simple_distances[simple_distances.product1==product]
print "Results for ", product
print p1
