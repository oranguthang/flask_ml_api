# flask_ml_api
Example of API for working with pre-trained ML models

# Installation and run

    docker build -t flask-ml-api .  
    docker run -d -p 5000:5000 flask-ml-api

# Send a request

    curl --location 'http://127.0.0.1:5000/predict/arrhythmia' \
    --header 'Content-Type: application/json' \
    --data '[
        51.0,
        0.0,
        170.0,
        82.0,
        90.0,
        155.0,
        382.0,
        216.0,
        88.0,
        9.0,
        40.0,
        35.0,
        20.0,
        0,
        73.0,
        13.090909090909092,
        20.40320830386508,
        10.909090909090908,
        20.403208303865085,
        9.090909090909092,
        18.074592916326193,
        12.727272727272727,
        23.516338614209943,
        11.636363636363637,
        21.500105707985373,
        12.363636363636363,
        22.375311069447626,
        9.090909090909092,
        16.00908832791265,
        11.363636363636363,
        20.249803590517747,
        12.363636363636363,
        21.648430554073556,
        11.272727272727273,
        19.499184132116447,
        10.909090909090908,
        19.023430528979496,
        10.909090909090908,
        19.27409943657314,
        4.866666666666667,
        10.603890795363746,
        3.4,
        7.125482439807146,
        -1.2555555555555555,
        3.596217148307062,
        -4.2333333333333325,
        8.820855967535124,
        3.4000000000000004,
        7.780745465570763,
        0.5,
        1.1968709203585823,
        -2.1333333333333333,
        6.35373905665003,
        1.4333333333333331,
        6.639088792899219,
        3.922222222222222,
        12.584194231036188,
        2.522222222222222,
        8.672626156156188,
        3.5777777777777775,
        8.53094041969843,
        4.044444444444444,
        8.19071289801918
    ]'
