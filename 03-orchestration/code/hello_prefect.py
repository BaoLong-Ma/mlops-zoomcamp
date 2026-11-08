from prefect import task, flow 

# this is a test, trying using prefect

@task
def say_hello(name : str) -> str :
    message = f"Hello {name} !"
    print(message)
    return message

@task 
def exclaim(message : str) -> str : 
    result = message + " (exclaimed!)"
    print(result)
    return result

@flow
def hello_flow(name :str) -> str :
    message = say_hello(name)
    result = exclaim(message)
    return result

if __name__ == "__main__" :
    hello_flow("Bao") 
