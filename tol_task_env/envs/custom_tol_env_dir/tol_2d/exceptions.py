class RodEmptyException(Exception):
    message = 'You are trying to remove ball from empty rod with no balls present!'

    def __str__(self):
        return self.message


class RodAlreadyFullException(Exception):
    message = 'You are trying to add ball to the rod that is already full!'

    def __str__(self):
        return self.message
