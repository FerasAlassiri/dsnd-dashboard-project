# python-package/employee_events/query_base.py

# Import any dependencies needed to execute sql queries
from .sql_execution import QueryMixin
import pandas as pd  # (not strictly required but often handy for hints/type)

# Define a class called QueryBase
# Use inheritance to add methods
# for querying the employee_events database.
class QueryBase(QueryMixin):

    # Create a class attribute called `name`
    # set the attribute to an empty string
    name: str = ""

    # Define a `names` method that receives
    # no passed arguments
    def names(self) -> list[tuple]:
        # Return an empty list
        return []

    # Define an `event_counts` method
    # that receives an `id` argument
    # This method should return a pandas dataframe
    def event_counts(self, id: int) -> pd.DataFrame:
        # QUERY 1
        # Write an SQL query that groups by `event_date`
        # and sums the number of positive and negative events
        # Use f-string formatting to set the FROM {table}
        # to the `name` class attribute
        # Use f-string formatting to set the name
        # of id columns used for joining
        # order by the event_date column
        sql = f"""
            SELECT ee.event_date,
                   SUM(ee.positive_events) AS positive_events,
                   SUM(ee.negative_events) AS negative_events
            FROM {self.name}
            JOIN employee_events ee
                USING({self.name}_id)
            WHERE {self.name}.{self.name}_id = {id}
            GROUP BY ee.event_date
            ORDER BY ee.event_date
        """
        return self.pandas_query(sql)

    # Define a `notes` method that receives an id argument
    # This function should return a pandas dataframe
    def notes(self, id: int) -> pd.DataFrame:
        # QUERY 2
        # Write an SQL query that returns `note_date`, and `note`
        # from the `notes` table
        # Set the joined table names and id columns
        # with f-string formatting
        # so the query returns the notes
        # for the table name in the `name` class attribute
        sql = f"""
            SELECT n.note_date, n.note
            FROM {self.name}
            JOIN notes n
                USING({self.name}_id)
            WHERE {self.name}.{self.name}_id = {id}
            ORDER BY n.note_date DESC
        """
        return self.pandas_query(sql)
