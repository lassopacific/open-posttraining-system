## eta progress
import time


## helper function to calculate the remaining time ..
def progress(processed, total, start_time, show_eta=False, label="Progress"):

    progress = f"{label}: {processed}/{total}"
    pad_width = len(f"{label}: {total}/{total} | ETA: 00h 00m 00s")

    ## if ETA disabled or nothing has been processed yet.
    if not show_eta or processed <= 0:
        return progress.ljust(pad_width)

    elapsed = time.time() - start_time

    if elapsed <= 0:
        return progress.ljust(pad_width)

    remaining = max(total - processed, 0)

    ## if processed ---> already started
    if processed:
        avg_time = elapsed / processed  ## average time per process
        eta_seconds = avg_time * remaining
    else:  ## if processed ---> nothing ---> then eta_seconds will start from 0
        eta_seconds = 0

    eta_seconds = max(int(round(eta_seconds)), 0)

    ## this will give us the remaining time ---> in seconds
    ## divmod function returns quotient and remainder
    ## quotient as minutes ---> remainder as seconds
    minutes, rem_seconds = divmod(eta_seconds, 60)

    ## this one will give hours and remaining minutes
    hours, minutes = divmod(minutes, 60)

    if hours:
        eta = f"{hours}h {minutes:02d}m {rem_seconds:02d}s"

    elif minutes:
        eta = f"{minutes:02d}m {rem_seconds:02d}s"

    else:
        eta = f"{rem_seconds:02d}s"

    message = f"{progress} | ETA: {eta}"

    return message.ljust(pad_width)