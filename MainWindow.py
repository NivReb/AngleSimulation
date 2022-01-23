import random
import calendar


def d_to_e():
    d_to_e_string = input('Please enter a string: ')
    first_d_index = d_to_e_string.find('d')
    from_first_d_to_end = d_to_e_string[first_d_index+1:]
    from_start_to_first_d = d_to_e_string[:first_d_index+1]
    e_string = from_first_d_to_end.replace('d', 'e')
    print(from_start_to_first_d + e_string)


def to_lower(char):
    return char.lower()


def missing_word(word):
    word_len = len(word)
    print('_ ' * word_len)


def palindrom():
    string = input('Enter a word: ')
    no_spaces_string = string.replace(" ", "")
    print(no_spaces_string)
    lower_case_string = to_lower(no_spaces_string)
    print(lower_case_string)
    string_len = len(lower_case_string)  # = 8
    if (string_len % 2) == 0:
        print(lower_case_string[:string_len/2])
        print(lower_case_string[-1:-string_len/2])
        if lower_case_string[:string_len/2] == lower_case_string[string_len:string_len/2:-1]:
            print('OK')
        else:
            print('NOT')
    else:
        print(lower_case_string[:string_len // 2])
        print(lower_case_string[string_len:string_len//2:-1])
        if lower_case_string[:string_len//2] == lower_case_string[string_len:string_len//2:-1]:
            print('OK')
        else:
            print('NOT')



def temperature_converter():
    temp = input('Insert the temperature you would like to convert: ')
    temp_format = to_lower(temp)
    if 'f' in temp_format:
        temp_val = float(temp_format.replace('f', ""))
        print((5*temp_val-160)/9, 'C')
    elif 'c' in temp_format:
        temp_val = float(temp_format.replace('c', ""))
        print((9*temp_val+(32*5))/5, 'F')
    else:
        print('Temperature units must be specified')


def day_in_the_date():
    date = input('Enter a date: ')
    day = int(date[0:2].lstrip("0"))

    month = int(date[3:5].lstrip("0"))

    year = int(date[len(date)-4:len(date)])

    print(calendar.day_name[calendar.weekday(year, month, day)])

def Error(char):
    if len(char) > 1:
        if char.isalpha():
            print("E1")
        else:
            print("E3")
    else:
        if char.isalpha():
            print(char)
        else:
            print("E2")




HANGMAN_ASCII_ART = 'Welcome to the game Hangman ' \
                    '' \
""" 
     _    _                                         
    | |  | |                                        
    | |__| | __ _ _ __   __ _ _ __ ___   __ _ _ __  
    |  __  |/ _` | '_ \ / _` | '_ ` _ \ / _` | '_ \ 
    | |  | | (_| | | | | (_| | | | | | | (_| | | | |
    |_|  |_|\__,_|_| |_|\__, |_| |_| |_|\__,_|_| |_|
                         __/ |                      
                        |___/"""

MAX_TRIES = 6

print(HANGMAN_ASCII_ART, '\n', MAX_TRIES)


#hangman_word = input('Please enter a word: ')
#missing_word(hangman_word)

guessed_letter = input('Guess a letter: ')
Error(to_lower(guessed_letter))


'''
print("""x-------x""")
print("""x-------x
|
|
|
|
|""")
print("""x-------x
|       |
|       0
|
|
|""")
print("""x-------x
|       |
|       0
|       |
|
|""")
print("""x-------x
|       |
|       0
|      /|/
|
|""")
print("""x-------x
|       |
|       0
|      /|/
|      /
|""")
print("""x-------x
|       |
|       0
|      /|/
|      / /
|""")
'''