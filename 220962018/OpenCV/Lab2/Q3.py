import cv2


def resize_image(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    return resized_image


def crop_image(image, start_x, start_y, width, height):
    cropped_image = image[start_y:start_y + height, start_x:start_x + width]
    return cropped_image


def main():
    image = cv2.imread('./assets/source.jpg')
    resized_image = resize_image(image, width=800, height=800)
    cropped_image = crop_image(resized_image, start_x=100, start_y=100, width=400, height=300)  # Adjust as needed
    cv2.imshow('Resized Image', resized_image)
    cv2.imshow('Cropped Image', cropped_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
