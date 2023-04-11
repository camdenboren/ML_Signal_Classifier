import calendar
from cgi import test
import collections
import datetime
from pymongo import MongoClient

from pip import main

def delete_signal():
    while(True):
        user_choice = input("Make a selection: \n1. Select Start/End date to delete\n2. Return to main menu")
        if(user_choice == "1"):
            deleteFromDatabase()
        elif(user_choice == "2"):
            return

def deleteFromDatabase():
    startInput = str(input("Start date that you want to delete(mm/dd/yyyy): ")) #Start date
    startSplitData = startInput.split("/")
    # startsplitdata month
    while(True):
        if (int(startSplitData[0]) <= 0 or int(startSplitData[0]) > 12):
            startSplitData[0] = str(input("enter a month in the range of 1 - 12 for start date: "))
        else:
            break
    #startsplitdata year
    while(True):
        if (int(startSplitData[2]) <= 0):
            startSplitData[2] = str(input("Enter a year greater than 0: "))
        else:
            break
    
    #startsplitdata day
    startEndOfMonth = calendar.monthrange(int(startSplitData[2]), int(startSplitData[0]))[1]
    while(True):
        if (int(startSplitData[1]) > startEndOfMonth or int(startSplitData[1]) <= 0):
            startSplitData[1] = str(input("Enter a date that is between 1 and %i for Start date: " % startEndOfMonth))
        else:
            break
    endInput = str(input("end date that you want to delete(mm/dd/yyyy): ")) #End date
    endSplitData = endInput.split("/")
    #endsplitdata year because year has to be >=
    while(True):
        yeardiff = int(endSplitData[2]) - int(startSplitData[2])
        if(yeardiff < 0):
            endSplitData[2] = str(input("Enter a year that is greater than or equal to %s: " % startSplitData[2]))
        else:
            break
    #endsplitdata month
    while(True):
            if(yeardiff == 0):
                if(int(endSplitData[0]) < int(startSplitData[0]) or int(endSplitData[0]) > 12):
                    endSplitData[0] = str(input("enter a month in the range of %s - 12 for end date: " % startSplitData[0]))
                else:
                    break
            elif (int(endSplitData[0]) <= 0 or int(endSplitData[0]) > 12):
                endSplitData[0] = str(input("enter a month in the range of 1 - 12 for end date: "))
            else:
                break
    monthdiff = int(endSplitData[0]) - int(startSplitData[0])
    #endsplitdata day
    endEndOfMonth = calendar.monthrange(int(endSplitData[2]), int(endSplitData[0]))[1]
    while(True):
        if (yeardiff == 0 and monthdiff == 0):
            if(int(endSplitData[1]) < int(startSplitData[1]) or int(endSplitData[1]) > int(endEndOfMonth)):
                endSplitData[1] = str(input("Enter a date that is between %s and %i for End date: " % startSplitData[1], endEndOfMonth))
            else:
                break
        elif (int(endSplitData[1]) > endEndOfMonth or int(endSplitData[1]) <= 0):
            endSplitData[1] = str(input("Enter a date that is between 1 and %i for End date: " % endEndOfMonth))
        else:
            break
        #TODO make sure to create a catch that will not allow the program to run if dates don't match up 

    #0-2 0 = month, 1 = day, 2 = year 
    #saving date information in [gregorian] format
    startDT = datetime.date(int(startSplitData[2]), int(startSplitData[0]), int(startSplitData[1]))
    endDT = datetime.date(int(endSplitData[2]), int(endSplitData[0]), int(endSplitData[1]))

    #If the years are different, then this whole function would need to run based on year, month, day
    #if not, then just month and day.
    difyear = (endDT.year - startDT.year + 1) or 1
    difmonth = (endDT.month - startDT.month + 1) or 1
    difday = (endDT.day - startDT.day + 1) or 1


    print("Start date: " + str(startDT))
    print("End date: " + str(endDT))
    #TODO Need to make sure to check that the given date is not out of the month range
    #Case for for same [year] and [month]
    if (difyear == 1 and difmonth == 1):
        dayloop(startDT, endDT)
    #Case for for same [year]
    elif(difyear == 1):
        monthloop(startDT, endDT, False)
    #case for [year], [month], and [days] being different
    else:
        yearloop(startDT, endDT)

def deletionProcess(day):
    client = MongoClient('143.244.155.10', 8080)
    db = client.test #this is the database name
    # _collection = db['test_store_signals'] #this is the collection name
    _collection = db['test_training_signals']

    # convert to have 0 in front if less than 10
    if (day.day < 10):
        usedDay = "0" + str(day.day)
    else:
        usedDay = day.day
    if (day.month < 10):
        usedMonth = "0" + str(day.month)
    else:
        usedMonth = day.month

    convertedDay = "%s/%s/%i" % (usedMonth, usedDay, day.year)
    date = "^" + str(convertedDay) #this should be the [date] that we are deleting
    choiceOfDay = {"date-time": {"$regex": str(date)}} #will delete something that starts with [date]
    deleted = _collection.delete_many(choiceOfDay) #delete all cases that match [choiceOfDay]
    print(deleted.deleted_count, " documents deleted") #print cases that have been deleted

def yearloop(startDT, endDT):
    difyear = (endDT.year - startDT.year + 1)
    changedYear = False

    for y in range(difyear):
        decemberRange = calendar.monthrange(startDT.year + y, 12)
        if (changedYear and int(startDT.year + y) == int(endDT.year)):
            #print("case 20") years are equal to eachother so we are in the last year
            monthloop(datetime.date(startDT.year+y, 1, 1), endDT, True)
        elif (changedYear):
            #print("case 21") past the first year but not in the last year so we start on the January 1st and end on December 31st
            monthloop(datetime.date(startDT.year+y, 1, 1), datetime.date(startDT.year+y, 12, decemberRange[1]), True)
        else:
            #print("case 22") First run of the loop
            monthloop(startDT, datetime.date(startDT.year, 12, decemberRange[1]), True)
        changedYear = True

#loop that will occur for when month is more than 1
def monthloop(startDT, endDT, yearBool):
    changedMonth = False
    difmonth = (endDT.month - startDT.month + 1) #find the dif between the months
    # print(difmonth)
    for m in range(difmonth):
        endOfMonth = calendar.monthrange(startDT.year, startDT.month + m)
        #print("current month range: " + str(endOfMonth[1]))
        if (changedMonth and (int(startDT.month + m) == int(endDT.month))):
            #print("case 10") this case is for if we are at the end of the month. I don't remember why I added the bool in this case
            dayloop(datetime.date(startDT.year, startDT.month + m, 1), endDT)
        elif (changedMonth):
            #print("case 11") Case for if we changed the month but are not at the end so [startmonth + m] < [endmonth]
            dayloop(datetime.date(startDT.year, startDT.month + m, 1), datetime.date(startDT.year, startDT.month + m, endOfMonth[1]))
        elif (yearBool and (int(startDT.month) == 12)):
            #print("case 12") odd case for if the individual started on december and moved to next year
            dayloop(datetime.date(startDT.year, startDT.month, startDT.day), endDT)
        elif (yearBool and (int(startDT.month) == int(endDT.month))):
            #print("case 13") again odd case if we only work with a single month such as january from the [yearloop]
            dayloop(datetime.date(startDT.year, startDT.month, 1), endDT)
        else:
            #print("case 14") first loop of the month
            dayloop(startDT, datetime.date(startDT.year, startDT.month, endOfMonth[1]))
        changedMonth = True
    
#loop that will only occur for days in the same month
def dayloop(currentDT, currentEndDT):
    # print("current DT: " + str(currentDT))
    # print("Current end DT: " + str(currentEndDT))
    findRange = currentEndDT.day - currentDT.day
    for date in range(findRange+1):
        print(datetime.date(currentDT.year, currentDT.month, currentDT.day + date))
        deletionProcess(datetime.date(currentDT.year, currentDT.month, currentDT.day + date))
        #TODO Would call deltionProcess here