import csv

                #file name                                  #accTime        #use wipers
ScnScriptList=[
               #
               # NOT READY NOT READY NOT READY NOT READY NOT READY
               #['exec /home/gt/incity2/sunny.txt',          [5],    0],
               #['exec /home/gt/incity2/dayrain.txt',        [6],    1],
               #['exec /home/gt/incity2/night.txt',          [5],    0],
               #['exec /home/gt/incity2/nightrain.txt',      [5],    1],#
               #
               # offroad 37kmh READY
               #['exec /home/gt/offroad/dayrain.txt',        [5],    1]#,# acc ok
               #['exec /home/gt/offroad/sunny.txt',          [3.8],  0]#,# acc ok
               #['exec /home/gt/offroad/night.txt',          [3.8],  0]#,# acc ok
               #['exec /home/gt/offroad/nightrain.txt',      [5],    1]#,# acc ok
               #
               # Incity 3 50kmh READY
               #['exec /home/gt/incity3/dayrain.txt',        [6.6],    1],
               #['exec /home/gt/incity3/sunny.txt',          [4.5],    0]#,
               #['exec /home/gt/incity3/night.txt',          [4.5],    0],
               #['exec /home/gt/incity3/nightrain.txt',      [6.6],    1],
               #
               # NOT READY NOT READY NOT READY NOT READY NOT READY
               #['exec /home/gt/incity/dayrain.txt',         [5],    1],
               #['exec /home/gt/incity/sunny.txt',           [5],    0],
               #['exec /home/gt/incity/night.txt',           [5],    0],
               #['exec /home/gt/incity/nightrain.txt',       [5],    1],
               #
               # countryroad 60kmh READY Done
               #['exec /home/gt/countryroad/dayrain.txt',    [8.9],    1],
               #['exec /home/gt/countryroad/sunny.txt',      [5.2],    0],
               #['exec /home/gt/countryroad/night.txt',      [5.2],    0],
               #['exec /home/gt/countryroad/nightrain.txt',  [5.8],    1],
               #
               # Highway 80kmh READY Done
               #['exec /home/gt/highway/dayrain.txt',        [10.6],    1]#,
               #['exec /home/gt/highway/sunny.txt',          [7.3],    0]
               #['exec /home/gt/highway/night.txt',          [7.3],    0]#,
               #['exec /home/gt/highway/nightrain.txt',      [10.6],    1]
               #
               ]

def CSVlogSave(data,pathtocsv,firstrow):
    with open(pathtocsv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(firstrow)
        for each in data:
            writer.writerow(each)