import multiprocessing

# Define four unique algorithms for demonstration
# Each takes a value and a weight
def algorithm_1(value):
    return (value + 1)

def algorithm_2(value):
    return (value * 2) 

def algorithm_3(value):
    return (value - 1)

def algorithm_4(value):
    return (value ** 2) 

# Worker function that processes data with a given algorithm and considers the weight
def worker(worker_id, data, algorithm, conn):
    response = "okay"
    for value in data:
        # Apply the algorithm 
        result = algorithm(value)
        # Send result to parent
        conn.send(result)
        print(f"Worker {worker_id} sent result: {result}")
        # Receive new response from parent
        response = conn.recv()
    conn.close()

if __name__ == "__main__":
    data_array = [1, 2, 3, 4, 5]  # The shared data array
    algorithms = [algorithm_1, algorithm_2, algorithm_3, algorithm_4]  # The list of algorithms
    processes = []
    parent_connections = []
    child_connections = []

    for i, algorithm in enumerate(algorithms):
        parent_conn, child_conn = multiprocessing.Pipe()
        parent_connections.append(parent_conn)
        child_connections.append(child_conn)
        
        process = multiprocessing.Process(target=worker, args=(i, data_array, algorithm, child_conn))
        processes.append(process)
        process.start()

    # 2d array to store results
    # final[][]
    


    # Example loop for 5 iterations assuming 5 elements in the data array
    for _ in range(len(data_array)):
        results = []
        # Collect results from children
        for conn in parent_connections:
            results.append(conn.recv())
        print(f"Results: {results}")
        
        # append results to 2d array


        # Evaluate results and send back new weights (simplified example)
        # Here, you would insert logic to evaluate results and decide on weights
        # weights = [1, 1, 1, 1]  # Placeholder for new weights based on evaluation
        
        for i, conn in enumerate(parent_connections):
            # conn.send(weights[i])
            conn.send("okay")
    
    # Close parent connections and join processes
    for conn in parent_connections:
        conn.close()
    
    for process in processes:
        process.join()

    print("Processing complete.")
