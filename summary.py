import os

class_names = ["circinatum", "garryana", "glabrum", "kelloggii", "macrophyllum","negundo"]

def save_and_summary( results ):
    if not os.path.exists("./train&test"):
        os.makedirs("./train&test")

    # completeName = os.path.join(".\train&test\AllResults.txt")         
    AllResults = open("./train&test/AllResults.txt" , 'w') 
    # completeName = os.path.join(".\train&test\Summary.txt")         
    Summary = open("./train&test/Summary.txt" , 'w') 
    
    num = dict()
    den = dict()
    for name in class_names:
        num[name] = 0
        den[name] = 0
    for result in results:
        AllResults.write(' -> '.join(result)+'\n')
        den[result[0]] += 1
        if result[0] == result[1]:
            num[result[0]] += 1
    n = 0
    d = 0
    Summary.write("Summary\n")
    for name in class_names:
        if den[name] != 0:
            line = name+" average is equal "+ str(num[name]/den[name])
            Summary.write(line+"\n")        
            print(line)
        n += num[name]
        d += den[name]
    if d != 0:
        line = "Average for whole dataset is equal "+ str(n/d)
        Summary.write(line+"\n")        
        print(line)

    AllResults.close()
    Summary.close()

if __name__ == "__main__":
    results = list()
    save_and_summary(results)  