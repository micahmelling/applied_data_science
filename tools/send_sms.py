import os

from twilio.rest import Client


account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
client = Client(account_sid, auth_token)


def main():
    message = client.messages.create(
        to='+16603516032',
        from_='+14804185557',
        body='Outwit. Outlast. Outplay.'
    )
