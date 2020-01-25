import os

# list of all spiecies
class_names = ["circinatum", "garryana", "glabrum", "kelloggii", "macrophyllum","negundo"]

# function for save results and summary
def save_and_summary( results ):
    # create files to save results
    if not os.path.exists("./train&test"):
        os.makedirs("./train&test")
    AllResults = open("./train&test/AllResults.txt" , 'w')     
    Summary = open("./train&test/Summary.txt" , 'w') 
    # create 2 dictionaries for computing correct preditions
    num = dict()
    den = dict()
    for name in class_names:
        num[name] = 0
        den[name] = 0
    for result in results:
        # write all preditions to AllResults file
        AllResults.write(' -> '.join(result)+'\n')
        # compute correctness for each spiecie
        den[result[0]] += 1
        if result[0] == result[1]:
            num[result[0]] += 1
    n = 0
    d = 0
    # Summary
    Summary.write("Summary\n")
    for name in class_names:
        # write results for each class
        if den[name] != 0:
            line = name+" average is equal "+ str(num[name]/den[name])
            Summary.write(line+"\n")        
            print(line)
        n += num[name]
        d += den[name]
    if d != 0:
        # write result for whole data set
        line = "Average for whole dataset is equal "+ str(n/d)
        Summary.write(line+"\n")        
        print(line)
    # close files
    AllResults.close()
    Summary.close()

if __name__ == "__main__":
    results = list()
    save_and_summary(results)  