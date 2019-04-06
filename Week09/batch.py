import sys
import driver
import time

# Load Data from files
start_file = open(sys.argv[1], "rU")
finish_file = open(sys.argv[2], "rU")

start = []
finish = []

for line in start_file:
    start.append(line)

for line in finish_file:
    finish.append(line)

start_file.close()
finish_file.close()


# Run in batch
sucesses = 0
errors = 0
total = len(start)
total_time = 0
for i in range(total):
    timer_start = time.time()
    driver.main(start[i])
    elapsed = time.time() - timer_start
    file = open("output.txt", "r")
    result = file.readline()
    if result == finish[i][:-1]:
        print "Validated in " + str(elapsed) + " s"
        if elapsed > 60:
            print "WARNING"
        if elapsed > 90:
            print "ERROR"

        sucesses += 1
    else:
        print "Not Validated  in " + str(elapsed) + " s"
        errors += 1
    total_time += elapsed
    file.close()

print "RESULT (sucesses/errors): " + str(sucesses) + " " + str(errors) + " - of " + str(total) + " tests\nTotal time: " + str(total_time) + " s"
