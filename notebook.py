import keras_ocr
import json
import cv2
import urllib
import glob
import numpy as np
import requests
import time

start_time = time.time()

pipeline = keras_ocr.pipeline.Pipeline()

details_dic = { "ads_urls":[
    			'https://www.lankapropertyweb.com/pics/5398417/5398417_1671010175_051.jpeg',
    			'https://www.lankapropertyweb.com/pics/5398417/5398417_1671010174_724.jpg'],
		"other_ads_urls1":[],
		"other_ads_urls2":['https://www.lankapropertyweb.com/pics/5398417/5398417_1671010174_724.jpg'],
		"other_ads_urls3":['https://www.lankapropertyweb.com/pics/5398417/5398417_1671010175_051.jpeg']
    		}

urls = details_dic["ads_urls"]

otherurls  = []

for key in details_dic:
    if key != "ads_urls":
        for value in details_dic[key]:
            otherurls.append(value)




images = [keras_ocr.tools.read(url) for url in urls]
prediction_groups = pipeline.recognize(images)

texts = {}
for i, group in enumerate(prediction_groups):
    texts[f"name{i+1}"] = [text for text, box in group]


counts = [len(values) for values in texts.values()]
max_count = max(counts)
print(texts)
print(max_count)



def orb_sim(img1, img2):
    # SIFT is no longer available in cv2 so using ORB
    orb = cv2.ORB_create()

    # detect keypoints and descriptors
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)

    # define the bruteforce matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # perform matches.
    matches = bf.match(desc_a, desc_b)
    # Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)


img_folder = 'images/*'

list1 = []
list2 = []
lowres=False
visibility=False


for url in urls:
    with urllib.request.urlopen(url) as url:
        s = url.read()
        arr = np.asarray(bytearray(s), dtype=np.uint8)
        img1= cv2.imdecode(arr, -1)

        height, width = img1.shape[:2]
        if height<380 and width<500:
             lowres=True
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # Compute the average pixel value
        mean_val = cv2.mean(gray)[0]

        # Define the threshold value
        threshold = 128

        # Check if the image is too dark
        if mean_val < threshold:
             visibility=True
             
           
    for path in glob.glob(img_folder):
        # print(path)
        img2 = cv2.imread(path, cv2.COLOR_BGR2RGB)
        # img3 = resize(img2, (img1.shape[0], img1.shape[1]), anti_aliasing=True, preserve_range=True)
        # img3 = resize(img2, img1.size, anti_aliasing=True, preserve_range=True)

        orb_sim_out = orb_sim(img1, img2)
        list1.append(orb_sim_out)

    for otherurl in otherurls:
        # print(path)
        response = requests.get(otherurl)
        arr = np.asarray(bytearray(response.content), dtype=np.uint8)
        img2 = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

        # img2 = cv2.imread(path, cv2.COLOR_BGR2RGB)
        # img3 = resize(img2, (img1.shape[0], img1.shape[1]), anti_aliasing=True, preserve_range=True)
        # img3 = resize(img2, img1.size, anti_aliasing=True, preserve_range=True)

        orb_sim_out = orb_sim(img1, img2)
        list2.append(orb_sim_out)

print(list1)
print(len(list1))
print("Max score is : ", max(list1)*100)
isSpam=False
if max(list1)>0.7:
	isSpam=True

print(list2)
print(len(list2))
print("Max score is : ", max(list2)*100)
isDuplicate=False
if max(list2)>0.7:
	isDuplicate=True


if max_count>list1.index(max(list1)):
	print("Max index is list1 : ", list1.index(max(list1)))
else:
	print("Max index is list1 : ", list1.index(max(list1))/len(urls))

print("Max index is list2 : ", list2.index(max(list2)))

if list2.index(max(list2))>len(otherurls):
	print("Max index is list2 url  : ", otherurls[list2.index(max(list2)) % len(otherurls)])
	value = otherurls[list2.index(max(list2)) % len(otherurls)]
else:
	print("Max index is list2 url  : ", otherurls[list2.index(max(list2))])
	value = otherurls[list2.index(max(list2))]

def find_key(value):
    for key in details_dic:
        if key == "ads_urls":
            continue
        if value in details_dic[key]:
            return key
    return None

key = find_key(value)

print("similer ad is present id :", key)

end_time = time.time()

total_time = end_time - start_time

print("Total time taken: {:.2f} seconds".format(total_time))



x = {"Text_Content":max_count,"IsSpam":isSpam,"IsDuplicate":isDuplicate,"Similer_Ad_details":[key,value],"Low_Res":lowres,"darkness":visibility}
y = json.dumps(x)
print(y)

