# multi-label-bookgenre
This project is my thesis research to pursue my undergraduate degree in Computer Science major.
The purpose of my thesis is to analyzes the application of multi-label classification practice using the Convolutional Neural Network model to classify the fiction book genre information based on the cover. The genre categories that used in this project are fantasy, literature (general fiction), romance, mystery, and horror.

On this project, I'm using VGG16 as the model's architecture. I also collected both the training and testing dataset through web scraping using Python and BeautifulSoup package. I scraped all the fiction book's cover, ISBN, and title from the FictionDB website. 
If you happen to see my previous repository about scraping book dataset from Goodreads, maybe you already know that I've tried scraping from the FictionDB first, then tried the Goodreads later on. But in the end, I use the dataset from FictionDB in my thesis because of the label issue from the Goodreads' one.

On the predict.py file you can see that I tried to predict a batch of data from a folder and print the plot pictures all at once. 
These are the a few example of the predict_batch results:


![results](https://user-images.githubusercontent.com/32532031/174702755-d2ce94ab-9d02-479a-8450-1cbcdbc336b9.png)


If you have any suggestion about this code, you can pull a request or contribute to my repository. And please kindly give your feedback if you found this code helpful for you ^^
