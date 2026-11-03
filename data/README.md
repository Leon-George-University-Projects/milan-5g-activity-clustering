# Dataset

Download the Milan Telecom Dataset from Harvard Dataverse:

<https://doi.org/10.7910/DVN/EGZHFV>

Place the daily telecom files in this directory using their original names:

```text
sms-call-internet-mi-*.txt
```

The raw data is intentionally excluded from Git because of its size. The
analysis expects tab-separated rows containing grid ID, timestamp, country
code, incoming and outgoing SMS, incoming and outgoing calls, and internet
activity.
