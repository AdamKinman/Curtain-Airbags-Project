import horizontal_align
import process_masks_new
import masks_to_polygons


if __name__ == '__main__':
    print("\n------ ALIGNING IMAGES HORIZONTALLY ------\n")
    horizontal_align.run()
    print("\n------ PROCESSING MASKS ------\n")
    process_masks_new.run()
    print("\n------ CONVERTING MASKS TO POLYGONS ------\n")
    masks_to_polygons.run()