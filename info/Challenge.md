## SOURCE: 
    https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/?tab=details

### Summary
Data from several different space-based exoplanet surveying missions have enabled discovery of thousands of new planets outside our solar system, but most of these exoplanets were identified manually. With advances in artificial intelligence and machine learning (AI/ML), it is possible to automatically analyze large sets of data collected by these missions to identify exoplanets. Your challenge is to create an AI/ML model that is trained on one or more of the open-source exoplanet datasets offered by NASA and that can analyze new data to accurately identify exoplanets.

### Background

https://youtu.be/UTzp2Ghyej8

Exoplanetary identification is becoming an increasingly popular area of astronomical exploration. Several survey missions have been launched with the primary objective of identifying exoplanets.
Exoplanetary identification is becoming an increasingly popular area of astronomical exploration. Several survey missions have been launched with the primary objective of identifying exoplanets. Utilizing the “transit method” for exoplanet detection, scientists are able to detect a decrease in light when a planetary body passes between a star and the surveying satellite. Kepler is one of the more well-known transit-method satellites, and provided data for nearly a decade. Kepler was followed by its successor mission, K2, which utilized the same hardware and transit method, but maintained a different path for surveying. During both of these missions, much of the work to identify exoplanets was done manually by astrophysicists at NASA and research institutions that sponsored the missions. After the retirement of Kepler, the Transiting Exoplanet Survey Satellite (TESS), which has a similar mission of exoplanetary surveying, launched and has been collecting data since 2018.

For each of these missions (Kepler, K2, and TESS), publicly available datasets exist that include data for all confirmed exoplanets, planetary candidates, and false positives obtained by the mission (see Resources tab). For each data point, these spreadsheets also include variables such as the orbital period, transit duration, planetary radius, and much more. As this data has become public, many individuals have researched methods to automatically identify exoplanets using machine learning. But despite the availability of new technology and previous research in automated classification of exoplanetary data, much of this exoplanetary transit data is still analyzed manually. Promising research studies have shown great results can be achieved when data is automatically analyzed to identify exoplanets. Much of the research has proven that preprocessing of data, as well as the choice of model, can result in high-accuracy identification. Utilizing the Kepler, K2, TESS, and other NASA-created, open-source datasets can help lead to discoveries of new exoplanets hiding in the data these satellites have provided.

### Objectives
Your challenge is to create an artificial intelligence/machine learning model that is trained on one or more of NASA’s open-source exoplanet datasets, and not only analyzes data to identify new exoplanets, but includes a web interface to facilitate user interaction. A number of exoplanet datasets from NASA’s Kepler, K2, and TESS missions are available (see Resources tab). Feel free to utilize any open-source programming language, machine learning libraries, or software solutions that you think would fit into this project well. Think about the different ways that each data variable (e.g., orbital period, transit duration, planetary radius, etc.) might impact the final decision to classify the data point as a confirmed exoplanet, planetary candidate, or false positive. Processing, removing, or incorporating specific data in different ways could mean the difference between higher-accuracy and lower-accuracy models. Think about how scientists and researchers may interact with the project you create. Will you allow users to upload new data or manually enter data via the user interface? Will you utilize the data users provide to update your model? The choices are endless!

### Potential Considerations
You may (but are not required to) consider the following:

Your project could be aimed at researchers wanting to classify new data or novices in the field who want to interact with exoplanet data and do not know where to start.
Your interface could enable your tool to ingest new data and train the models as it does so.
Your interface could show statistics about the accuracy of the current model.
Your model could allow hyperparameter tweaking from the interface.


## Resources
NASA does not endorse any non-U.S. Government entity and is not responsible for information contained on non-U.S. Government websites. For non-U.S. Government websites, participants must comply with any data use parameters of that specific website.

#### NASA Data & Resources
"Kepler Objects of Interest (KOI)"[https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative]: This dataset is a comprehensive list of all confirmed exoplanets, planetary candidates, and false positives determined on all the transits captured by Kepler. Utilizing the variables in this labeled dataset could make for a solid method of performing supervised learning from different variables in the dataset. See column “Disposition Using Kepler Data” for classification.

TESS Objects of Interest (TOI): This dataset is a comprehensive list of all confirmed exoplanets, planetary candidates (PC), false positives (FP), ambiguous planetary candidates (APC), and known planets (KP, previously identified) identified by the TESS mission so far. See column “TFOWPG Disposition” for classification.

K2 Planets and Candidates: This dataset is a comprehensive list of all confirmed exoplanets, planetary candidates, and false positives determined on all the transits captured by the K2 mission. See the “Archive Disposition” column for classification.

Exoplanet Detection Using Machine Learning: This research article gives a great overview of exoplanetary detection methods as well as machine learning aimed at exoplanetary classification and a survey of the literature in the field as it stood in 2021.

Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification: This research article takes a good look at some of the machine learning techniques that have resulted in high accuracy using the datasets above. It also explores some of the pre-processing techniques that help achieve higher accuracy.

Space Agency Partner Resources
Canadian Space Agency (CSA)

Near - Earth Object Surveillance Satellite (NEOSSat) - Astronomy Data: The dataset includes the astronomical images from the Near-Earth Object Surveillance Satellite (NEOSSat). NEOSSat is the world's first space telescope dedicated to detecting and tracking asteroids, comets, satellites, and space debris.

Near - Earth Object Surveillance Satellite (NEOSSat): observing asteroids, space debris and exoplanets: This website provides more information on the Near - Earth Object Surveillance Satellite (NEOSSat). It covers topics such as asteroid and space debris tracking, exoplanet detection, and Canada’s contributions to space situational awareness.

James Webb Space Telescope (JWST) - Information: This page provides detailed information about the James Webb Space Telescope and highlights Canada’s contributions to the mission.