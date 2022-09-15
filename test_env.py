import requests, json
# 这里地址要和上面的服务端地址及端口好一致
url = 'http://172.17.16.17:2345/'
 
# r = requests.post(url, data=json.dumps(data))
# print(r.json())
test_farmer_string = {
"self_cards": "5,10,5",
"self_out": "8,9,10,J,Q,K,A,5,6,7,8,9,7,Q",
"oppo_last_move": "6",
"oppo_out": "7,8,9,10,J,Q,K,J,X,2,2,2,A,A",
"self_win_card_num": 1,
"oppo_win_card_num": 0,
"oppo_left_cards":6,
"bomb_num": 0,
"history":{
"self": ["8,9,10,J,Q,K,A", '5,6,7,8,9', '7', 'Q', "", ""] ,
"oppo": ["7,8,9,10,J,Q,K", "", "", 'J','X', '2,2,2,A,A', '6']
}
}
    
r = requests.get(url, data=json.dumps(test_farmer_string)) # 发送到服务端
print(r.json())

test_landlord_string = {
    "self_cards": "5,10,5,6,6,6",
    "self_out": "7,8,9,10,J,Q,K,J,X,2,2,2,A,A",
    "oppo_last_move": "",
    "oppo_out": "8,9,10,J,Q,K,A,5,6,7,8,9,7,Q",
    "self_win_card_num": 0,
    "oppo_win_card_num": 1,
    "oppo_left_cards":3,
    "bomb_num": 0,
    "history":{
        "self": ["7,8,9,10,J,Q,K", "", "", 'J','X', '2,2,2,A,A'],
        "oppo": ["8,9,10,J,Q,K,A", '5,6,7,8,9', '7', 'Q', "", ""]
    }
    }

r = requests.get(url, data=json.dumps(test_landlord_string)) # 发送到服务端
print(r.json())