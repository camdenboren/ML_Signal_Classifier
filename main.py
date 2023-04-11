from classify_signal import classify_signal
from store_signal import store_signal
from train_new_model import train_new_model
from delete_signal import delete_signal



def main():
    while(True):        
        user_choice = input("Please select an option:\n1. Classify signal\n2. Store signal in DB\n3. Train new model\n4. Delete Signal Data\n5. Exit Program\n")

        e = 0
        try:
            user_choice = int(user_choice)
        except:
            pass
        else:
            if(user_choice == 1):
                e = classify_signal()
            elif(user_choice == 2):
                e = store_signal()
            elif(user_choice == 3):
                e = train_new_model()
            elif(user_choice == 4):
                e = delete_signal()
            elif(user_choice == 5):
                break
            else:
                print("Invald option...")

        if e != 0:
            print_errors(e)
            break

    print("Exiting Application")

    
'''
Error reporting Idea
Functions return int values that correspond to the error that is triggered
Error Table

0 = No Error
1 = ?
2 = could not connect to the SDR
3 = Not storing Data in Database, error with signal collection?
'''
def print_errors(e):
    print("Error: " + str(e))  
                

main()