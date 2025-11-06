with open("Dataset/input.txt", "r") as file: #loading the file
    content = file.read()
    print(len(content.splitlines())) #counting the number of lines in the file
    
chars = sorted(list(set(content))) #getting the unique characters in the file
print(' '.join(chars)) #printing the unique characters
print("Total characters:", len(chars)) #printing the number of unique characters 