import pandas as pd
from urllib.parse import urlparse,urlencode
import ipaddress
import re
import urllib.request
from datetime import datetime
from bs4 import BeautifulSoup
import whois
import requests
from socket import *

df_phish = pd.read_csv("D:/Project_and_Case_Study_1/Phishtank.csv")
print(df_phish.head())
print("Phishing dataset shape = ", df_phish.shape)

# Collecting 5,000 random Phishing URLs from dataset
phish_URL = df_phish.sample(n = 5000, random_state = 1).copy()
phish_URL = phish_URL.reset_index(drop = True)
print(phish_URL.head())
print("Phishing dataset shape = ", phish_URL.shape)

#.......Legitimate URls......
df_legit = pd.read_csv("D:/Project_and_Case_Study_1/Benign_list_big_final.csv") 
df_legit.columns = ['URLs']
print(df_legit.head())

# Collecting 5,000 random Legitimate URLs from dataset
legit_URL = df_legit.sample(n = 5000, random_state = 1).copy()
legit_URL = legit_URL.reset_index(drop = True)
print(legit_URL.head())
print("Legitimate dataset shape = ", legit_URL.shape)

#...Feature Extraction...
def URLs(url):
    return url

# Domain of the URL 
def getDomain(url):
    domain = urlparse(url).netloc
    if re.match(r"^www.", domain):
        domain = domain.replace("www.","")
    return domain

# Checks for IP address in URL 
def havingIP(url):
    try:
        ipaddress.ip_address(url)
        ip = 1
    except:
        ip = 0
    return ip

# Checks the presence of @ in URL 
def haveAtSign(url):
    if "@" in url:
        at = 1
    else:
        at = 0
    return at

# Finding the length of URL and categorizing
def getLength(url):
    if len(url) < 54:
        length = 0
    else: 
        length = 1
    return length

# Gives number of '/' in URL (URL_Depth)
def getDepth(url):
    s = urlparse(url).path.split('/')
    depth = 0
    for j in range(len(s)):
        if len(s[j]) != 0:
            depth = depth + 1
    return depth

# Checking for redirection '//' in the url 
def redirection(url):
    pos = url.rfind('//')
    if pos > 6:
        if pos > 7:
            return 1
        else:
            return 0
    else:
        return 0
    
# Existence of “HTTPS” Token in the Domain Part of the URL 
def httpDomain(url):
    domain = urlparse(url).netloc
    if 'https' in domain:
        return 1
    else:
        return 0

#listing shortening services
shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                      r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                      r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                      r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                      r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                      r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                      r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                      r"tr\.im|link\.zip\.net"
                      
# Checking for Shortening Services in URL 
def tinyURL(url):
    match = re.search(shortening_services, url)
    if match:
        return 1
    else:
        return 0
    
# Checking for Prefix or Suffix Separated by (-) in the Domain 
def prefixSuffix(url):
    if '-' in urlparse(url).netloc:
        return 1
    else:
        return 0

# Web traffic 
def web_traffic(url):
  try:
    #Filling the whitespaces in the URL if any
    url = urllib.parse.quote(url)
    rank = BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + url).read(), "xml").find(
        "REACH")['RANK']
    rank = int(rank)
  except TypeError:
        return 1
  if rank < 100000:
    return 1
  else:
    return 0

# Survival time of domain: The difference between termination time and creation time (Domain_Age)  
def domainAge(domain_name):
  creation_date = domain_name.creation_date
  expiration_date = domain_name.expiration_date
  if (isinstance(creation_date, str) or isinstance(expiration_date, str)):
    try:
      creation_date = datetime.strptime(creation_date, '%Y-%m-%d')
      expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d")
    except:
      return 1
  if ((expiration_date is None) or (creation_date is None)):
      return 1
  elif ((type(expiration_date) is list) or (type(creation_date) is list)):
      return 1
  else:
    ageofdomain = abs((expiration_date - creation_date).days)
    if ((ageofdomain / 30) < 6):
      age = 1
    else:
      age = 0
  return age

# End time of domain: The difference between termination time and current time (Domain_End) 
def domainEnd(domain_name):
  expiration_date = domain_name.expiration_date
  if isinstance(expiration_date, str):
    try:
      expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d")
    except:
      return 1
  if (expiration_date is None):
      return 1
  elif (type(expiration_date) is list):
      return 1
  else:
    today = datetime.now()
    end = abs((expiration_date - today).days)
    if ((end / 30) < 6):
      end = 0
    else:
      end = 1
  return end

# IFrame Redirection (iFrame)
def iframe(response):
  if response == "":
      return 1
  else:
      if re.findall(r"[<iframe>|<frameBorder>]", response.text):
          return 0
      else:
          return 1

# Checks the effect of mouse over on status bar 
def mouseOver(response): 
  if response == "" :
    return 1
  else:
    if re.findall("<script>.+onmouseover.+</script>", response.text):
      return 1
    else:
      return 0

# Checks the status of the right click attribute 
def rightClick(response):
  if response == "":
    return 1
  else:
    if re.findall(r"event.button ?== ?2", response.text):
      return 0
    else:
      return 1

# Checks the number of forwardings  
def forwarding(response):
  if response == "":
    return 1
  else:
    if len(response.history) <= 2:
      return 0
    else:
      return 1

#.....Computing URL Features.....

# Function to extract features
def featureExtraction(url,label):

  features = []
  #Address bar based features (10)
  features.append(URLs(url))
  features.append(getDomain(url))
  features.append(havingIP(url))
  features.append(haveAtSign(url))
  features.append(getLength(url))
  features.append(getDepth(url))
  features.append(redirection(url))
  features.append(httpDomain(url))
  features.append(tinyURL(url))
  features.append(prefixSuffix(url))
  
  # Domain based features (4)
  dns = 0
  try:
    domain_name = whois.whois(urlparse(url).netloc)
  except:
    dns = 1

  features.append(dns)
  features.append(web_traffic(url))
  features.append(1 if dns == 1 else domainAge(domain_name))
  features.append(1 if dns == 1 else domainEnd(domain_name))
  
  # HTML & Javascript based features (4)
  try:
    response = requests.get(url)
  except:
    response = ""
  features.append(iframe(response))
  features.append(mouseOver(response))
  features.append(rightClick(response))
  features.append(forwarding(response))
  features.append(label)
  
  return features


# Feature extraction on legitimate URLs
legi_features = []
label = 0
for i in range(0, 5000):
    url = legit_URL['URLs'][i]
    legi_features.append(featureExtraction(url, label))
    
# Converting the list to dataframe
feature_names = ['URLs', 'Domain', 'Have_IP', 'Have_At', 'URL_Length', 'URL_Depth','Redirection', 
                      'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 'Web_Traffic', 
                      'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over','Right_Click', 'Web_Forwards', 'Label']

legitimate = pd.DataFrame(legi_features, columns = feature_names)
print(legitimate.head())

# Storing the extracted legitimate URLs features to csv file
legitimate.to_csv('D:/Project_and_Case_Study_1/legitimate.csv', index = False)

#...Phishing URls...
# Extracting the feautres & storing them in a list
phish_features = []
label = 1
for i in range(0, 5000):
  url = phish_URL['url'][i]
  phish_features.append(featureExtraction(url,label))
  
# Converting the list to dataframe
feature_names = ['URLs', 'Domain', 'Have_IP', 'Have_At', 'URL_Length', 'URL_Depth','Redirection', 
                      'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 'Web_Traffic', 
                      'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over','Right_Click', 'Web_Forwards', 'Label']

phishing = pd.DataFrame(phish_features, columns= feature_names)
print(phishing.head())

# Storing the extracted Phishing URLs features to csv file
phishing.to_csv('phishing.csv', index = False)

# Final Dataset

# Concatenating the dataframes into one 
urldata = pd.concat([legitimate, phishing]).reset_index(drop = True)
print(urldata.head())
print("Final datset shape = ", urldata.shape)

# Storing the data in CSV file
urldata.to_csv('D:/Project_and_Case_Study_1/URL_Feature_Extracted_data.csv', index = False)