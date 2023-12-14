import numpy as np


def f_success(num_time_steps, i):
    return np.exp((num_time_steps - i) * np.log(0.5) / num_time_steps)


def f_failure(num_time_steps, i):
    return -np.exp((num_time_steps - i) * np.log(0.5) / num_time_steps) + 1


def main():
    labels = np.load("labels.npy").reshape(-1)
    
    continuous_labels = np.zeros_like(labels)
    
    index = 0
    
    with open("text.txt") as input_file:
        for line in input_file:
            tokens = line.split(": ")
            
            if tokens[0] == "Saved":
                num_time_steps = int(tokens[1])
                stop = num_time_steps + 1
                for i in range(1, stop):
                    continuous_labels[index] = f_success(num_time_steps, i)
                    index += 1
            
            if tokens[0] == "Dead":
                num_time_steps = int(tokens[1])
                stop = num_time_steps + 1
                for i in range(1, stop):
                    continuous_labels[index] = f_failure(num_time_steps, i)
                    index += 1
    
    # testing
    # prints end of 8, all of 9, all of 10, beginning of 11
    for i in range(13059, 13471):
        print(continuous_labels[i])
    
    continuous_labels = continuous_labels[:, np.newaxis]
    
    np.save("continuous_labels.npy", continuous_labels)


if __name__ == "__main__":
    main()
