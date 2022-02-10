import random
import calendar
import math


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


def last_early(my_str):
    my_str = to_lower(my_str)
    if my_str[-1] in my_str[:len(my_str)-1]:
        return True
    else:
        return False


def distance(num1, num2, num3):
    dis_first_two = abs(num1 - num2)
    dis_first_last = abs(num1 - num3)
    if dis_first_two == 1 or dis_first_last == 1:
        if dis_first_two >= 2 or dis_first_last >= 2:
            return True
        else:
            return False
    else:
        return False


def fix_age(age):
    age = 0
    return age


def filter_teens(a=13, b=13, c=13):
    non_valid_ages = [13, 14, 17, 18, 19]
    if a in non_valid_ages:
        a = fix_age(a)
    if b in non_valid_ages:
        b = fix_age(b)
    if c in non_valid_ages:
        c = fix_age(c)
    age_sum = a + b + c
    return age_sum


def chocolate_maker(small, big, x):
    big_len = 5
    if x - (small + big * big_len) <= 0:
        return True
    else:
        return False


def func(num1, num2):
    """
    Calculate the sum of the two input values
    :param num1: first value
    :param num2: second value
    :return: The sum of two values
    """
    return num1 + num2


def shift_left(my_list):
    if len(my_list) == 3:
        a, b, c = my_list
        shift_list = [b, c, a]
        return shift_list


def format_list(my_list):
    """
    Acceptes Acceptes even parameter list. create new list with param only at even indeces and adds the last param.
    return the list as one string with each param seperate with ',' and the last param as 'and' befor him.
    :param my_list: Acceptes even parameter list
    :return:
    """
    if len(my_list) % 2 == 0:
        even_places_list = my_list[::2]
        last = "and " + my_list[-1]
        formatted = even_places_list + [last]
        return ", ".join(formatted) # Method who can creat one string from list and seperate its contennt with specified seperator.
    else:
        return "List len is not even"


def extend_list_x(list_x, list_y):
    for i in range(0, len(list_y)):
        list_x.insert(i, list_y[i])

    return list_x


def are_lists_equal(list1, list2):
    list1.sort()
    list2.sort()
    if list1 == list2:
        return True
    else:
        return False


def longest(my_list):
    my_list.sort(key=len)
    print(my_list)
    return my_list[-1]

def squared_numbers(start, stop):
    squared_list = []
    while start <= stop:
        squared_list.append(start**2)
        start += 1
    return squared_list

def is_greater(my_list, n):
    greater_list = []
    for num in my_list:
        if num > n:
            greater_list.append(num)
    return greater_list

def numbers_letters_count(my_str):
    num_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    num = 0
    letter = 0
    for char in my_str:
        if char in num_list:
            num += 1
        else:
            letter += 1
    list = [num, letter]
    return list

def seven_boom(end_number):
    list = []
    for i in range(end_number+1):
        num_to_char = str(i)
        if i % 7 == 0:
            list.append('BOOM')
        elif '7' in num_to_char:
            list.append('BOOM')
        else:
            list.append(i)
    return list

def sequence_del(my_str):
    """
    delete duplicate letters in my_str. cant deal with duplicate spaces.
    :param my_str: string
    :return: currect string
    """
    aa = " ".join(my_str.split())
    result = "".join(dict.fromkeys(aa))
    return result


def price(tup):
    return float(tup[-1])


def sort_prices(list_of_tuples):
    sort_list = []
    return sorted(list_of_tuples, key=price, reverse=True)


def mult_tuple(tuple1, tuple2):
    pair_tuples = []
    for num1 in tuple1:
        for num2 in tuple2:
            pair_tuples.append((num1, num2))
            pair_tuples.append((num2, num1))
    return pair_tuples


def sort_anagrams(list_of_strings):
    anagram_list = []
    anagram_local_list = []
    for word_1 in list_of_strings:
        anagram_local_list.append(word_1)
        for word_2 in list_of_strings:
            if word_1 != word_2 and (sorted(word_1) == sorted(word_2)):
                anagram_local_list.append(word_2)

        anagram_list.append(anagram_local_list)
        anagram_local_list = []
    return anagram_list

list_of_words = ['deltas', 'retainers', 'desalt', 'pants', 'slated', 'generating', 'ternaries', 'smelters', 'termless', 'salted', 'staled', 'greatening', 'lasted', 'resmelts']
print(sort_anagrams(list_of_words))
#Hangman project:

def Error(char):
    """
    Function check if the user input character is valid.
    If the input value compose of more than 1 letter from the abc Error is E1.
    If the input value compose of more than 1 latter and contains none abc char Error is E3.
    If the input value compose of 1 letter that is not one of abc char Error is E2.
    :param char: user input letter
    :return: Error value, or char value if char is validate
    """
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


def is_valid_input(char):
    """
    FUnction check if user input char is validate
    :param char: input letter
    :return: If the input is valid(True) or not(False)
    """
    if len(char) > 1:
        return False
    if not char.isalpha():
        return False
    if len(char) == 1 and char.isalpha():
        return True


def check_valid_input(letter_guessed, old_letter_guessed):
    """

    :param letter_guessed: player current guess
    :param old_letter_guessed: player old guesses
    :return: boolean with the validity of the letter and if the player all ready guessed it before
    """
    if is_valid_input(letter_guessed) and letter_guessed.lower() not in old_letter_guessed:
        return True
    else:
        return False


def try_update_letter_guessed(letter_guessed, old_letters_guessed):
    if check_valid_input(letter_guessed, old_letters_guessed):
        old_letters_guessed.append(letter_guessed)
        print(old_letters_guessed)
        return True
    else:
        print('X')
        old_letters_guessed.sort()
        print(*old_letters_guessed, sep=' -> ')
        return False



def main():
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

    old_letters = ['a', 'p', 'c', 'f']
    print(try_update_letter_guessed('A', old_letters))
    print(try_update_letter_guessed('s', old_letters))
    print(try_update_letter_guessed('$', old_letters))
    print(try_update_letter_guessed('d', old_letters))


#if __name__ == "__main__":
#    main()

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