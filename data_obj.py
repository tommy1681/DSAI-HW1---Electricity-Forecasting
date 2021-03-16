import datetime
import time
import csv
class data_obj:
    def __init__(self,d):
        self.d = d 
    
    def to_csv(self,path):
        oneDay = datetime.timedelta(days=1)
        startDay = datetime.date(2021,3,23)

        with open(path, 'w', newline='',encoding="utf-8") as student_file:
            writer = csv.writer(student_file)
            writer.writerow(["date","operating_reserve(MW)"])
            for i in self.d[0]:

                timeArray = startDay.strftime("%Y%m%d")
                writer.writerow([timeArray,i])
                startDay = startDay + oneDay 
        print("預測資料寫檔完畢")


