# Notes on the data processing stage
The `cmip26.py` script runs the data processing stage. It generates coarse
surface velocities and diagnosed forcings from the CM2.6 dataset. You may
configure certain parameters such as bounds (lat/lon) and CO2 level.

## Notes on the CM2.6 dataset
### Requester Pays
We use the CM2.6 dataset hosted on the Pangeo Cloud Datastore. Though public, it
is *not* freely available, due to the data being in a Requester Pays bucket on
Google Cloud Platform (GCP). Reading data from the bucket requires you to have
Google Cloud access credentials configured with billing access; bandwidth
charges are then charged to that account i.e. you.

A guide for configuring GCP credentials is over here on the Pangeo Cloud
Datastore: [Working with requester pays
data](https://catalog.pangeo.io/browse/master/ocean/GFDL_CM2_6/GFDL_CM2_6_control_ocean_surface/).
Alternatively, if you have a JSON credentials file (downloaded from GCP after
creating a service account), place it at
`~/.config/gcloud/application_default_credentials`.
