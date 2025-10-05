NASA data science project

instructions to run:

- sign up to urs.earthdata.nasa.gov
- add NASA GESDISC DATA ARCHIVE to your authorized applications
- create a file  "_netrc " in C:\Users\yourusername  
```
machine urs.earthdata.nasa.gov
    login username
    password password
```

```bash
pip install -r requirements.txt
```

- run scrape, shaper then: 
```bash
python server.py
```

