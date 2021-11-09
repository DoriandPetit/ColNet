import numpy as np
from skimage import color


def net_out2rgb(L,Lab_out):
    """Translates the net output back to an image.

    More specifically: unnormalizes both L and ab_out channels, stacks them
    into an image in LAB color space and converts back to RGB.

    Args:
        L: original L channel of an image
        ab_out: ab channel which was learnt by the network

    Retruns:
        3 channel RGB image
    """
    # Convert to numpy and unnnormalize
    L = L.numpy() * 100.0
    Lab_out = Lab_out.numpy() * 254.0 - 127.0

    #print("SHAPE : ",L.shape,ab_out.shape)
    print(L.shape,Lab_out.shape)
    #L = Lab_out[0].numpy() * 100.0
    #a_out = Lab_out[1].numpy() * 254.0 - 127.0
    #b_out = Lab_out[2].numpy() * 254.0 - 127.0
    #print("SHAPE : ",L.shape,ab_out.shape)


    # L and ab_out are tenosr i.e. are of shape of
    # Height x Width x Channels
    # We need to transpose axis back to HxWxC
    L = L.transpose((1, 2, 0))
    Lab_out = Lab_out.transpose((1, 2, 0))
    #print("SHAPE : ",L,ab_out)
    # Stack layers
    img_stack = np.dstack((L, Lab_out))

    # This line is CRUCIALL
    #   - torch requires tensors to be float32
    #   - thus all above (L, ab) are float32
    #   - scikit image floats are in range -1 to 1
    #   - http://scikit-image.org/docs/dev/user_guide/data_types.html
    #   - idk what's next, but converting image to float64 somehow
    #     does the job. scikit automagically converts those values.
    img_stack = img_stack.astype(np.float64)
    image = color.lab2rgb(img_stack)*256
    image = image.astype(np.uint8)


    return  image
