
def duration_printer(total_seconds: int):
    if total_seconds >= 3600:
        duration = f"{round(total_seconds / 3600, 2)}h"
    elif total_seconds >= 60:    
        duration = f"{round(total_seconds / 60, 2)}m"
    else:
        duration = f"{round(total_seconds, 2)}s"
    print(f"Model has been trained in {duration}")