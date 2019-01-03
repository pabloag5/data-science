#************************************************
#Assignment4 - Big data in business and industry
#************************************************

#importing libraries
library(readxl)
library(tidyverse)
library(tidytext)
library(ggplot2)

#********** First part ****************************************

#Question 1
#importing dataset
fridgedata<-read_excel("fridge.xlsx")
head(fridgedata)
str(fridgedata)
fdPrescriptive<-fridgedata %>% select(
  'Person',
  'Prescription 1', 
  'Prescription 2',
  'Prescription 3') %>%
  rename(pres1='Prescription 1',
         pres2='Prescription 2',
         pres3='Prescription 3') %>%
  gather(namePres,prescriptive,-Person) %>% 
  arrange(Person) %>%
  filter(prescriptive!='NA')
fdPrescriptive

fdP_tidy<-fdPrescriptive %>% unnest_tokens(word, prescriptive)
fdP_tidy %>% count(word,sort=T)
#excluding stop words
data(stop_words)
fdP_non_sw<-fdP_tidy %>% anti_join(stop_words)
# another way of excluding stop words is 
# fdP_tidy %>% anti_join(get_stopwords())

fdP_non_sw %>% count(word,sort=T) %>%slice(3L)

cat(
  sprintf("The 3rd word most often ocurring is:\n ' %s",
          fdP_non_sw %>% 
            count(word,sort=T) %>%
            slice(3L) %>%
            select(word)
  ),"'"
)

#********** Second part ****************************************

#importing Hotel reviews dataset
hotel_rw<-read.csv('7282_1.csv')
glimpse(hotel_rw)
data(stop_words)

#Question 2
unique(hotel_rw$city)
unique(hotel_rw$country)
NYC_hotel_rw<- hotel_rw %>% 
  filter(city=="New York") %>%
  select(city, reviews.text) %>%
  mutate(reviews.text=as.character(reviews.text)) %>%
  unnest_tokens(word, reviews.text) %>%
  anti_join(stop_words) %>%
  count(word, sort = T)
cat(
  sprintf("The 5th most commonly ocurring word is:\n ' %s",
          NYC_hotel_rw[5,"word"]),"'"
)

#Question 3
hotel_rw_sentences<-hotel_rw %>%
  unnest_tokens(sentences,reviews.text, token="sentences") %>%
  count(sentences,sort=T)
cat(
sprintf("The 7th most written sentence is:\n ' %s",
      hotel_rw_sentences[7,"sentences"]),"'"
)

#Question 4
NYC_hotel_rw_sent<-NYC_hotel_rw %>% 
  inner_join(get_sentiments("bing")) %>%
  group_by(sentiment) %>%
  summarise(number_rws=n()) %>%
  ungroup() %>%
  mutate(percent_sent=number_rws/sum(number_rws))
cat(  
sprintf(
  "The percentage of positive words is: %.2f", 
        NYC_hotel_rw_sent[
          NYC_hotel_rw_sent$sentiment=="positive",
          "percent_sent"]
  )
)