import json
import hashlib
import time
import datetime

def timeStamp(s):
    return time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f").timetuple())

if __name__=="__main__":
    hash = hashlib.sha1()
    users = ['Harmke', 'Jos']
    event_data = [
        [
            {
                'label' : 'snoring',
                't_start' : timeStamp("2016-05-04 23:41:52,117"),
                't_end' : timeStamp("2016-05-04 23:43:12,013")
            },
            {
                'label': 'traffic',
                't_start': timeStamp("2016-05-04 22:23:52,117"),
                't_end': timeStamp("2016-05-04 22:43:01,013")
            }
        ],
        [
            {
                'label': 'snoring',
                't_start': timeStamp("2016-05-05 23:41:52,117"),
                't_end': timeStamp("2016-05-05 23:43:12,013")
            },
        ]
    ]



    for user, events in zip(users, event_data):
        hash.update(str(time.time()) + user)

        sleep_dat = {}
        sleep_dat['id'] = hash.hexdigest()
        sleep_dat['user'] = user
        sleep_dat['events'] = events

        json_dat = json.dumps(sleep_dat)
        with open(user + '.txt', 'w') as outfile:
            json.dump(sleep_dat, outfile)

