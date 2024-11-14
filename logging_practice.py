


import logging
import sys
from pathlib import Path
import datetime


import functools

# TODO: Figure out the main ideas that one might want to do when logging 
# log on different levels of abstraction
# print out specific parts of the code
# Before after.


"""
How to use a decorator to not only log stuff, but also hide that functionality away using functions.
Additionally show how to set up a logging object that would help create the messages.

- aim for comprehensiveness for documentation and debugging so that one can prioririze and search in this infractructure for the best information.
df probelsm - adding custom levels with PIPELINE + Algorithm + Modell + Different computations --> use interface data to push claude with more easier to interpret data. 
"""


def setup_logger(name='ml_experiment', log_dir='logs', level=logging.INFO):
    """
    Step-by-step logger setup with examples of what each component achieves
    """
    # STEP 1: Create the base logger
    logger = logging.getLogger(name)
    logger.setLevel(level)  # Base minimum level for ALL handlers
    
    # Example of why hierarchical naming is important:
    #   ml_experiment.training
    #   ml_experiment.evaluation
    #   ml_experiment.preprocessing
    
    # STEP 2: Setup Log Directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    # Important for: Organizing logs by experiment/run
    
    # STEP 3: Create Formatters
    # Detailed formatter for debugging and error analysis
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Example output:
    # "2024-11-14 10:23:45,123 - ml_experiment - ERROR - Model failed to converge"
    
    # Simple formatter for console viewing
    simple_formatter = logging.Formatter('%(levelname)s - %(message)s')
    # Example output:
    # "INFO - Starting training epoch 1"
    
    # STEP 4: Create File Handler (Debug and above)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(log_path / f'experiment_{timestamp}.log')
    fh.setLevel(logging.DEBUG)  # Captures everything
    fh.setFormatter(detailed_formatter)
    # Purpose: Complete historical record with timestamps
    # Example debug message:
    # "2024-11-14 10:23:45,123 - ml_experiment - DEBUG - Batch 1: loss=0.534"
    
    # STEP 5: Create Console Handler (Info and above)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)  # Less verbose for console
    ch.setFormatter(simple_formatter)
    # Purpose: Real-time monitoring without clutter
    # Example console output:
    # "INFO - Epoch 1/10: accuracy=0.856"
    
    # STEP 6: Create Error Handler (Error and above)
    eh = logging.FileHandler(log_path / f'errors_{timestamp}.log')
    eh.setLevel(logging.ERROR)  # Only serious issues
    eh.setFormatter(detailed_formatter)
    # Purpose: Separate error tracking
    # Example error log:
    # "2024-11-14 10:23:45,123 - ml_experiment - ERROR - Memory allocation failed"
    
    # STEP 7: Add all handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.addHandler(eh) # when an event is logged, we want to handle it in multipel ways not just logging --> how can we precisely in an event do we document     
    # define the output file, define the format, define which events we want.
    
    return logger



logger = setup_logger(level = logging.INFO)


def log_function(logger, level):
    def decorator(func):
        @functools.wraps(func) # emphasizes during statidc runtime the function definition helps to check it well --> typecheckign will happen runtime anyway, but i want the same behavior.
        def new_function(*args, **kwargs):
                logger.log(level, f"calling new_function outside level parmeter: {func.__name__}")
                x = func(*args, **kwargs)
                logger.log(level, f"finished a new action outside level parmeter: {func.__name__}")
                return x
        
        return new_function
    return decorator
    
   
        
    return new_function

@log_function(logger, logging.CRITICAL)
def function2():
     return 1 + 1


# logger vs logging are different
@log_function(logger, logging.INFO) # 1. log_fucntionn is callled --> create a decorator behavior--> decorator is callled with fucntion --> def now returns a new decorated fucntion
def print_hello_debug():
     print("hello")

if __name__ == '__main__': 
    


    logger.info('print action')
    print("action")
    logging.info('completed printing action')

    logger.log(10, "hello 2")



    print_hello_debug()
    function2()



