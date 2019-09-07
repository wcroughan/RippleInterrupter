"""
Access the Rat Brain Atlas over the internet and show any coordinates that we
might be interested in.
"""

from urllib.request import urlopen
from PIL import Image
import io
import json
import logging

MODULE_IDENTIFIER = "[BrainAtlas] "
# Set the default coordinates to what you would use for hippocampus.
DEFAULT_ML_COORDINATE = 2.7
DEFAULT_AP_COORDINATE = -4.3
DEFAULT_DV_COORDINATE = 0
ENCODING_SCHEME = 'utf-8'

def fetchImage(image_url):
    """
    Fetch an image from url
    """

    try:
        url_content = urlopen(image_url)
        image_data = Image.open(io.BytesIO(url_content.read()))
    except Exception as err:
        logging.error(MODULE_IDENTIFIER + "Unable to read image from URL.")
        print(err)
        return

    return image_data

class WebAtlas(object):
    """
    Access interface to fetch BrainAtlas images from the internet and show any
    set of coordinates in Coronal, Sagittal or Horizontal Slices.

    TODO: We might have to add more features to make this useful. Caching
    images and urls locally would be helpful.
    """

    def __init__(self):
        self._url_base = "http://labs.gaidi.ca/rat-brain-atlas/api.php?"

    def fetchCoordinatesAndShowImage(self):
        # Open a dialog box to get the correct corrdinates and show the
        # corresponding image
        pass

    def getCoronalImage(self, ml, ap, dv, show=True):
        """
        Return an image of the Coronal slice at this corrdinate and where in
        the image does the specified coordinate lie.
        """
        url_data = self.queryServer(ml, ap, dv)
        coronal_image = fetchImage(url_data['coronal']['image_url'])
        coordinates = (url_data['coronal']['left'], url_data['coronal']['top'])
        if show:
            coronal_image.show()
        return (coronal_image, coordinates)

    def getSagittalImage(self, ml, ap, dv, show=True):
        """
        Return an image of the Sagittal slice at this corrdinate and where in
        the image does the specified coordinate lie.
        """
        url_data = self.queryServer(ml, ap, dv)
        sagittal_image = fetchImage(url_data['sagittal']['image_url'])
        coordinates = (url_data['coronal']['left'], url_data['coronal']['top'])
        if show:
            sagittal_image.show()
        return (sagittal_image, coordinates)

    def getHorizontalImage(self, ml, ap, dv, show=True):
        """
        Return an image of the Horizontal slice at this corrdinate and where in
        the image does the specified coordinate lie.
        """
        url_data = self.queryServer(ml, ap, dv)
        horizontal_image = fetchImage(url_data['horizontal']['image_url'])
        coordinates = (url_data['coronal']['left'], url_data['coronal']['top'])
        if show:
            horizontal_image.show()
        return (horizontal_image, coordinates)

    def queryServer(self, ml=DEFAULT_ML_COORDINATE, ap=DEFAULT_AP_COORDINATE, dv=DEFAULT_DV_COORDINATE):
        """
        Query the rat brain atlas server to get the correct images for the
        prescribed set of coordinates and return the image urls.
        """

        access_url = self._url_base + "ml=%2.1f&ap=%2.1f&dv=%2.1f"%(ml,ap,dv)
        try:
            url_response = urlopen(access_url)
        except Exception as err:
            logging.error(MODULE_IDENTIFIER + "Unable to read access URL. Check coordinates.")
            print(err)
            return

        # Decode the contents of the webpage
        try:
            url_content = url_response.read().decode(ENCODING_SCHEME)
            url_data_dict = json.loads(url_content)
        except Exception as err:
            logging.error(MODULE_IDENTIFIER + "Unable to parse URL content.")
            print(err)
            return

        # print(url_data_dict)
        return url_data_dict
