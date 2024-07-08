from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("flavors_of_cacao.csv")

nullBeanType = dataset['Bean Type'].isnull().sum()
print("There are " + str(nullBeanType) + " reviews made in 2013 in the dataset.")

dataset = dataset.dropna()

print("There are " + str(dataset.shape[0]) + " tuples in this dataset")

encoder = LabelEncoder()
oneHotData = encoder.fit_transform(dataset["Company  (Maker-if known)"])
print("There are " + str(max(oneHotData)) + " unique company names are there in the dataset")

numReviews2013 = dataset[dataset['Review Date'] == 2013].shape[0]
print("There are " + str(numReviews2013) + " reviews made in 2013 in the dataset.")

plt.hist(dataset['Rating'], bins=20, edgecolor="black")
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Chocolate Ratings')
plt.show()

dataset['Cocoa Percent'] = dataset['Cocoa Percent'].str.replace('%', '').astype(float)

plt.scatter(dataset['Cocoa Percent'], dataset['Rating'], alpha=0.1)
plt.xlabel('Cocoa Percent')
plt.ylabel('Rating')
plt.title('Cocoa Percent vs. Rating')
plt.show()

scaler = StandardScaler()
dataset['Normalized Rating'] = scaler.fit_transform(dataset[['Rating']])

print("Normalized Ratings:")
print(dataset['Normalized Rating'].head(15))