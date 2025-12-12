import horizontal_align
import process_masks_new
import masks_to_polygons
import polygons_to_dxf


if __name__ == '__main__':
    print("\n------ ALIGNING IMAGES HORIZONTALLY ------\n")
    horizontal_align.run()
    print("\n------ PROCESSING MASKS ------\n")
    process_masks_new.run()
    print("\n------ CONVERTING MASKS TO POLYGONS ------\n")
    masks_to_polygons.run()
    print("\n------ CONVERTING POLYGONS TO DXF ------\n")
    polygons_to_dxf.run()