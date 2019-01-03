#************************************************
#Assignment5 - Big data in business and industry
#************************************************

library(sparklyr)
library(tidytext)
library(tidyverse)
library(dplyr)
library(ggplot2)

data("stop_words")

#spark_install("2.1.0")
sc<-spark_connect(master = "local")
class(sc)

#whole dataset. NOT WORK unnest_tokens
#reviews_tbl<-copy_to(sc,name = 'reviews',read_csv("7282_1.csv"))
#glimpse(reviews_tbl)
#str(reviews_tbl)

#reviews filter out-of-scale
reviewsdata<-read_csv("7282_1.csv")
reviews_cl<-reviewsdata %>% 
  filter(reviews.rating<=5) %>%
  unnest_tokens(word, reviews.text) %>%
  anti_join(stop_words) 

#copy to spark
reviews_rt<-copy_to(sc,name = 'reviewsrt',reviews_cl)
#TODO: Find error. code NOT WORKING
#reviews_rt<-reviews_tbl %>% 
#  unnest_tokens(word, reviews_text) %>%
#  anti_join(stop_words) %>% filter(reviews_rating<=5)


#create function to calculate rate of new review based on historical data
#create function: this function already calculates de weighted mean
findRate<-function(newreview) {
  myreview<-tibble(review=newreview)
  mywords<-myreview %>%
    unnest_tokens(word,review) %>%
    anti_join(stop_words)
  rateValue<-reviews_rt %>%
    filter(!is.na(reviews_rating)) %>%
    right_join(mywords, copy=TRUE) %>%
    summarise(mean(reviews_rating))
}
#myRate<-findRate(newreview)
#myRate
a_review<-"I enjoyed my stay: friendly staff and the breakfast was great"
b_review<-"We were disappointed on the 2-bed room that was promised to us. Had the worst sleep ever."
c_review<-"There was no toilet paper available"
aRate<-findRate(a_review)
bRate<-findRate(b_review)
cRate<-findRate(c_review)
aRate #rate for a review
bRate #rate for b review
cRate #rate for c review