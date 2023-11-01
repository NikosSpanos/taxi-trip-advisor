from datetime import datetime, timedelta

def end_date_calculation(dt_start: datetime, dt_interval:int) -> datetime:
    ending_date = dt_start - timedelta(days=dt_interval)
    return ending_date

def date_calculation(starting_date:datetime, date_interval: int) -> datetime:
    end_date = end_date_calculation(starting_date, date_interval)
    return end_date