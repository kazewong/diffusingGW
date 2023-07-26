
from gwosc.timeline import get_segments
from gwosc.locate import get_urls

t0 = 1126259462

bulk_start_time = t0 - 12*60*60
bulk_end_time = t0 + 60*60*24*30*2
# define the instrument we want ('H1' for Hanford, 'L1' for Livingston)
ifo = 'H1'

# get time segments of available data within the specified 'bulk' time
segments = get_segments(f'{ifo}_DATA', bulk_start_time, bulk_end_time)

# get URLs of data files for the above segments
urls = get_urls(ifo, segments[0][0], segments[-1][-1], sample_rate=4096)

directory = '/mnt/home/wwong/ceph/Dataset/GW/DiffusionLikelihood/'

# Write all the url to a file
with open(directory+'urls.txt', 'w') as f:
    for url in urls:
        f.write("wget -O " + directory + url.split('/')[-1] + " " + url + "\n")

