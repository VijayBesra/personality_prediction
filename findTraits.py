from cgitb import text
import requests

def findTraits(givenText):
    url = "https://big-five-personality-insights.p.rapidapi.com/api/big5"
    payload = [
        {
            "id" : "1",
            "language" : "en",
            "text" : givenText
        }
    ]

    headers = {
        "content-type": "application/json",
	    "X-RapidAPI-Key": "6d8a86c49amshd48ce186913176ap1ed2b4jsn877b89cc3c65",
	    "X-RapidAPI-Host": "big-five-personality-insights.p.rapidapi.com"
    }

    return requests.request("POST",url,json=payload,headers=headers).text