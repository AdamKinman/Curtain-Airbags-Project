import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from random import shuffle

from image_files import find_image_path, list_image_files

from config import (
    HORIZONTAL_ALIGN_IMAGES as IMAGE_FOLDER,
    WINDOW_MASK_FOLDER,
    SIDE_MIRROR_MASK_FOLDER as MIRROR_MASK_FOLDER,
    ROTATED_LABELS_FOLDER,
    PROCESSED_MASK_FOLDER
)
DAYLIGHT_OPENINGS_LABEL_KEY = 4

def rotatedLabelsToDict(rotatedLabelText):
    lines = rotatedLabelText.strip().split('\n')
    output = {}
    for line in lines[1:]:  # Skip the first line (angle)
        parts = line.split(' ')
        key = int(parts[0])
        coords = list(map(float, parts[1:]))
        points = []
        for i in range(0, len(coords), 2):
            points.append((coords[i], coords[i+1]))
        output.setdefault(key, []).append(points)
    return output

def squeezeMask(mask):
    """Squeeze a mask tensor to 2D, handling (1, H, W) or (H, W) formats."""
    if mask.ndim == 3 and mask.shape[0] == 1:
        return mask.squeeze(0)
    return mask

# Returns minimum distance between two masks, or 0 if they overlap
def distanceBetweenMasks(mask1, mask2):
    # Ensure masks are 2D
    mask1 = squeezeMask(mask1)
    mask2 = squeezeMask(mask2)
    # Check if masks overlap
    overlap = torch.logical_and(mask1, mask2)
    if overlap.any():
        return 0.0
    
    mask2Np = mask2.cpu().numpy().astype(np.uint8) * 255
    distTransform = cv2.distanceTransform(255 - mask2Np, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    mask1Indices = mask1.cpu().numpy()
    minDistance = np.min(distTransform[mask1Indices])
    return float(minDistance)

def getFrontWindowIndex(windowMasks, mirrorMask):
    distances = [distanceBetweenMasks(windowMask, mirrorMask) for windowMask in windowMasks]
    minDistance = min(distances)
    return distances.index(minDistance)

def identifyCorners(mask, angleThreshold=150, minDistance=6, blockSize=5):
    """
    Identify corners (sudden turns) in the outline of a mask shape.
    
    Args:
        mask: 2D torch tensor of boolean values representing a shape
        angleThreshold: Maximum angle (in degrees) for a point to be considered a corner.
                       Lower values = sharper corners only. Default 60.
        minDistance: Minimum distance between detected corners. Default 10.
        blockSize: Size of neighborhood for corner refinement. Default 5.
        
    Returns:
        List of (x, y) tuples representing corner positions
    """
    # Ensure mask is 2D
    mask = squeezeMask(mask)
    
    # Convert mask to numpy uint8 for OpenCV
    maskNp = mask.cpu().numpy().astype(np.uint8) * 255
    
    # Find contours of the shape
    contours, _ = cv2.findContours(maskNp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return []
    
    # Get the largest contour (the main shape)
    contour = max(contours, key=cv2.contourArea)
    
    # Flatten contour to 2D array of points
    contourPoints = contour.reshape(-1, 2)
    numPoints = len(contourPoints)
    
    if numPoints < 3:
        return []
    
    corners = []
    
    # Calculate angles at each point along the contour
    for i in range(numPoints):
        # Get neighboring points with wrap-around
        prevIdx = (i - minDistance) % numPoints
        nextIdx = (i + minDistance) % numPoints
        
        prevPoint = contourPoints[prevIdx].astype(float)
        currPoint = contourPoints[i].astype(float)
        nextPoint = contourPoints[nextIdx].astype(float)
        
        # Calculate vectors from current point to neighbors
        vec1 = prevPoint - currPoint
        vec2 = nextPoint - currPoint
        
        # Calculate angle between vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            continue
            
        cosAngle = np.dot(vec1, vec2) / (norm1 * norm2)
        cosAngle = np.clip(cosAngle, -1, 1)  # Handle numerical errors
        angle = np.degrees(np.arccos(cosAngle))
        
        # If angle is below threshold, it's a corner
        if angle < angleThreshold:
            corners.append((int(currPoint[0]), int(currPoint[1]), angle))
    
    # Non-maximum suppression: keep only the sharpest corner within minDistance
    if not corners:
        return []
    
    # Sort by angle (sharpest first)
    corners.sort(key=lambda x: x[2])
    
    filteredCorners = []
    for corner in corners:
        x, y, angle = corner
        # Check if this corner is far enough from all accepted corners
        tooClose = False
        for fx, fy in filteredCorners:
            dist = np.sqrt((x - fx) ** 2 + (y - fy) ** 2)
            if dist < minDistance:
                tooClose = True
                break
        if not tooClose:
            filteredCorners.append((x, y))
    
    return filteredCorners


def fitQuadratic(points):
    """Fit a quadratic polynomial y = a + bx + cx^2 using least squares."""
    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)
    
    # Build design matrix for quadratic polynomial
    X = np.column_stack([np.ones_like(x), x, x**2])
    
    # Solve least squares: X @ coeffs = y
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    return tuple(coeffs)  # (a, b, c)


def fitCubic(points):
    """Fit a cubic polynomial y = a + bx + cx^2 + dx^3 using least squares."""
    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)
    
    # Build design matrix for cubic polynomial
    X = np.column_stack([np.ones_like(x), x, x**2, x**3])
    
    # Solve least squares: X @ coeffs = y
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    return tuple(coeffs)  # (a, b, c, d)


def extrapolateOutlineBehindMirror(windowMask, mirrorMask, corners, 
                                   cornerDistanceThreshold=15, 
                                   excludeNearCorner=15,
                                   minPointsForFit=20,
                                   useCubic=True):
    """
    Predict the outline of the window behind the side mirror using polynomial extrapolation.
    
    Args:
        windowMask: 2D torch tensor of the window mask
        mirrorMask: 2D torch tensor of the mirror mask
        corners: List of (x, y) tuples representing detected corners
        cornerDistanceThreshold: Max distance from mirror for a corner to be considered "close"
        excludeNearCorner: Number of points to exclude near each corner
        useCubic: If True, use cubic polynomial (degree 3). If False, use quadratic (degree 2).
        minPointsForFit: Minimum number of points required for curve fitting
        
    Returns:
        Tuple of two tuples ((a1, b1, c1, d1), (a2, b2, c2, d2)) for the two cubic functions,
        or None if extrapolation is not possible
    """
    if mirrorMask is None or len(corners) < 2:
        return None
    
    # Ensure masks are 2D
    windowMask = squeezeMask(windowMask)
    mirrorMask = squeezeMask(mirrorMask)
    
    # Get the outline (contour) of the window
    maskNp = windowMask.cpu().numpy().astype(np.uint8) * 255
    contours, _ = cv2.findContours(maskNp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None
    
    contour = max(contours, key=cv2.contourArea)
    outlinePoints = contour.reshape(-1, 2)
    numPoints = len(outlinePoints)
    
    # Compute distance transform from mirror mask to find distances
    mirrorNp = mirrorMask.cpu().numpy().astype(np.uint8) * 255
    distFromMirror = cv2.distanceTransform(255 - mirrorNp, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    # Find outline indices for all corners
    allCornerIndices = []
    for cx, cy in corners:
        distances = np.sqrt(np.sum((outlinePoints - np.array([cx, cy])) ** 2, axis=1))
        outlineIdx = np.argmin(distances)
        allCornerIndices.append(outlineIdx)
    
    # Sort corners by outline index to establish order
    cornersSortedByOutline = sorted(enumerate(corners), key=lambda x: allCornerIndices[x[0]])
    sortedIndices = [allCornerIndices[i] for i, _ in cornersSortedByOutline]
    
    # Identify corners close to the mirror
    cornersNearMirror = []
    for sortedPos, (origIdx, (cx, cy)) in enumerate(cornersSortedByOutline):
        if 0 <= cy < distFromMirror.shape[0] and 0 <= cx < distFromMirror.shape[1]:
            dist = distFromMirror[cy, cx]
            if dist < cornerDistanceThreshold:
                cornersNearMirror.append((sortedPos, cx, cy, allCornerIndices[origIdx]))
    
    if len(cornersNearMirror) < 2:
        return None
    
    # Use first and last corners near mirror (already sorted by outline position)
    firstCornerNearMirror = cornersNearMirror[0]
    lastCornerNearMirror = cornersNearMirror[-1]
    
    firstSortedPos = firstCornerNearMirror[0]
    lastSortedPos = lastCornerNearMirror[0]
    firstIdx = firstCornerNearMirror[3]
    lastIdx = lastCornerNearMirror[3]
    
    # Find previous corner (before firstCornerNearMirror in outline order)
    prevSortedPos = (firstSortedPos - 1) % len(cornersSortedByOutline)
    prevCornerIdx = sortedIndices[prevSortedPos]
    
    # Find next corner (after lastCornerNearMirror in outline order)
    nextSortedPos = (lastSortedPos + 1) % len(cornersSortedByOutline)
    nextCornerIdx = sortedIndices[nextSortedPos]
    
    # Get points between previous corner and first corner (excluding near both corners)
    # Going from prevCornerIdx towards firstIdx
    pointsBeforeFirst = []
    if prevCornerIdx < firstIdx:
        # Normal case: previous corner comes before in the array
        startIdx = prevCornerIdx + excludeNearCorner
        endIdx = firstIdx - excludeNearCorner
        if startIdx < endIdx:
            for i in range(startIdx, endIdx):
                pointsBeforeFirst.append(outlinePoints[i])
    else:
        # Wrap around case: previous corner is after first corner in array
        # Go from prevCornerIdx to end, then from start to firstIdx
        startIdx = prevCornerIdx + excludeNearCorner
        endIdx = firstIdx - excludeNearCorner
        for i in range(startIdx, numPoints):
            pointsBeforeFirst.append(outlinePoints[i])
        for i in range(0, endIdx + 1):
            pointsBeforeFirst.append(outlinePoints[i])
    
    # Get points between last corner and next corner (excluding near both corners)
    # Going from lastIdx towards nextCornerIdx
    pointsAfterLast = []
    if lastIdx < nextCornerIdx:
        # Normal case: next corner comes after in the array
        startIdx = lastIdx + excludeNearCorner
        endIdx = nextCornerIdx - excludeNearCorner
        if startIdx < endIdx:
            for i in range(startIdx, endIdx):
                pointsAfterLast.append(outlinePoints[i])
    else:
        # Wrap around case: next corner is before last corner in array
        # Go from lastIdx to end, then from start to nextCornerIdx
        startIdx = lastIdx + excludeNearCorner
        endIdx = nextCornerIdx - excludeNearCorner
        for i in range(startIdx, numPoints):
            pointsAfterLast.append(outlinePoints[i])
        for i in range(0, endIdx + 1):
            pointsAfterLast.append(outlinePoints[i])
    
    if len(pointsBeforeFirst) < minPointsForFit or len(pointsAfterLast) < minPointsForFit:
        return None
    
    pointsBeforeFirst = np.array(pointsBeforeFirst)
    pointsAfterLast = np.array(pointsAfterLast)
    
    # Step 4: Fit polynomial functions using least squares
    fitFunc = fitCubic if useCubic else fitQuadratic
    
    try:
        coeffsFirst = fitFunc(pointsBeforeFirst)
        coeffsLast = fitFunc(pointsAfterLast)
    except np.linalg.LinAlgError:
        return None
    
    return (coeffsFirst, coeffsLast)


def evalPolynomial(coeffs, x):
    """Evaluate a polynomial at x. Works for both quadratic and cubic."""
    if len(coeffs) == 3:  # Quadratic: a + bx + cx^2
        a, b, c = coeffs
        return a + b * x + c * x**2
    else:  # Cubic: a + bx + cx^2 + dx^3
        a, b, c, d = coeffs
        return a + b * x + c * x**2 + d * x**3


def isPointInRotatedRect(point, rectCorners, margin=50):
    """
    Check if a point is inside a rotated rectangle defined by its 4 corners,
    with an optional margin around the rectangle.
    
    Args:
        point: (x, y) tuple
        rectCorners: List of 4 (x, y) tuples representing the corners in order
        margin: Distance to expand the rectangle outward (positive) or inward (negative).
                Default 0 means exact rectangle bounds.
        
    Returns:
        True if point is inside the (expanded/contracted) rectangle
    """
    # If margin is specified, expand the rectangle
    if margin != 0:
        # Calculate the center of the rectangle
        centerX = sum(c[0] for c in rectCorners) / len(rectCorners)
        centerY = sum(c[1] for c in rectCorners) / len(rectCorners)
        
        # Expand each corner outward from center by margin
        expandedCorners = []
        for cx, cy in rectCorners:
            # Vector from center to corner
            dx = cx - centerX
            dy = cy - centerY
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                # Normalize and scale by margin
                scale = (dist + margin) / dist
                expandedCorners.append((centerX + dx * scale, centerY + dy * scale))
            else:
                expandedCorners.append((cx, cy))
        rectCorners = expandedCorners
    
    px, py = point
    n = len(rectCorners)
    
    for i in range(n):
        x1, y1 = rectCorners[i]
        x2, y2 = rectCorners[(i + 1) % n]
        
        # Cross product of edge vector and point vector
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        
        # For a convex polygon, all cross products should have the same sign
        if i == 0:
            sign = cross >= 0
        elif (cross >= 0) != sign:
            return False
    
    return True


def getMirrorCentroid(mirrorMask):
    """
    Calculate the centroid (center of mass) of a mirror mask.
    
    Args:
        mirrorMask: 2D torch tensor or numpy array of the mirror mask
        
    Returns:
        (x, y) tuple of the centroid, or None if mask is empty
    """
    if isinstance(mirrorMask, torch.Tensor):
        mirrorMask = squeezeMask(mirrorMask)
        mirrorNp = mirrorMask.cpu().numpy().astype(np.uint8) * 255
    else:
        mirrorNp = mirrorMask
    
    # Find all non-zero pixels
    ys, xs = np.where(mirrorNp > 0)
    
    if len(xs) == 0:
        return None
    
    # Calculate centroid
    centroidX = np.mean(xs)
    centroidY = np.mean(ys)
    
    return (centroidX, centroidY)


def getIntersectionPoint(curve1, curve2, daylightOpeningsRect, mirrorMask, xRange=None, numSamples=1000, margin=30):
    """
    Find the intersection point of two polynomial curves within a daylight openings rectangle.
    Chooses the intersection point closest to the mirror centroid.
    
    Args:
        curve1: Tuple of coefficients for first curve (quadratic or cubic)
        curve2: Tuple of coefficients for second curve (quadratic or cubic)
        daylightOpeningsRect: List of 4 (x, y) tuples representing the rotated rectangle corners
        mirrorMask: 2D torch tensor of the mirror mask (to find closest point to mirror)
        xRange: Optional tuple (xMin, xMax) to limit search range. If None, uses expanded range based on rect and mirror.
        numSamples: Number of samples for numerical intersection finding
        margin: Margin to expand the daylight openings rectangle for point checking
        
    Returns:
        (x, y) tuple of intersection point, or None if no valid intersection exists
    """
    if curve1 is None or curve2 is None or daylightOpeningsRect is None:
        return None
    
    # Get x range - expand to include both the daylight openings rect AND the mirror area
    if xRange is None:
        rectXs = [p[0] for p in daylightOpeningsRect]
        xMin, xMax = min(rectXs), max(rectXs)
        
        # Also include the mirror centroid area in the search range
        if mirrorMask is not None:
            mirrorCentroid = getMirrorCentroid(mirrorMask)
            if mirrorCentroid is not None:
                centroidX, _ = mirrorCentroid
                # Expand search range to include mirror area with some padding
                xMin = min(xMin, centroidX - 100)
                xMax = max(xMax, centroidX + 100)
    else:
        xMin, xMax = xRange
    
    # Find intersections by looking for sign changes in (curve1 - curve2)
    xValues = np.linspace(xMin, xMax, numSamples)
    y1 = evalPolynomial(curve1, xValues)
    y2 = evalPolynomial(curve2, xValues)
    diff = y1 - y2
    
    # Find sign changes (potential intersections)
    signChanges = np.where(np.diff(np.sign(diff)) != 0)[0]
    
    # Collect all intersection points (without filtering by rect yet)
    allIntersectionPoints = []
    for idx in signChanges:
        # Refine intersection using linear interpolation between the two points
        x1, x2 = xValues[idx], xValues[idx + 1]
        d1, d2 = diff[idx], diff[idx + 1]
        
        # Linear interpolation to find x where diff = 0
        if d2 - d1 != 0:
            xIntersect = x1 - d1 * (x2 - x1) / (d2 - d1)
        else:
            xIntersect = (x1 + x2) / 2
        
        yIntersect = evalPolynomial(curve1, xIntersect)
        allIntersectionPoints.append((float(xIntersect), float(yIntersect)))
    
    if not allIntersectionPoints:
        return None
    
    # Find the intersection closest to the mirror centroid
    closestPoint = None
    if mirrorMask is not None:
        mirrorCentroid = getMirrorCentroid(mirrorMask)
        
        if mirrorCentroid is not None:
            centroidX, centroidY = mirrorCentroid
            
            minDist = float('inf')
            for px, py in allIntersectionPoints:
                # Calculate Euclidean distance to mirror centroid
                dist = np.sqrt((px - centroidX)**2 + (py - centroidY)**2)
                if dist < minDist:
                    minDist = dist
                    closestPoint = (px, py)
        else:
            # If can't find centroid, use the first intersection
            closestPoint = allIntersectionPoints[0]
    else:
        # If no mirror mask, use the first intersection
        closestPoint = allIntersectionPoints[0]
    
    if closestPoint is None:
        return None
    
    # Now check if the closest intersection is inside the daylight openings rectangle
    if not isPointInRotatedRect(closestPoint, daylightOpeningsRect, margin):
        return None
    
    return closestPoint


def updateMask(frontMask, cornerNearMirror1, cornerNearMirror2, curve1, curve2, intersectionPoint, smoothingKernel=5):
    """
    Update the front window mask by replacing the section between the two mirror-adjacent 
    corners with extrapolated curves meeting at an intersection point.
    
    Args:
        frontMask: 2D torch tensor of the front window mask
        cornerNearMirror1: (x, y) tuple of the first corner near the mirror
        cornerNearMirror2: (x, y) tuple of the second corner near the mirror
        curve1: Coefficients for the first curve (leading to cornerNearMirror1)
        curve2: Coefficients for the second curve (following cornerNearMirror2)
        intersectionPoint: (x, y) tuple where the two curves intersect, or None
        smoothingKernel: Size of the kernel for morphological smoothing (must be odd)
        
    Returns:
        Updated 2D torch tensor mask
    """
    frontMask = squeezeMask(frontMask)
    maskNp = frontMask.cpu().numpy().astype(np.uint8) * 255
    
    # Find contours of the original mask
    contours, _ = cv2.findContours(maskNp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return frontMask
    
    contour = max(contours, key=cv2.contourArea)
    outlinePoints = contour.reshape(-1, 2)
    numPoints = len(outlinePoints)
    
    # If no intersection point, just return smoothed original mask
    if intersectionPoint is None:
        smoothed = smoothMask(maskNp,smoothingKernel)
        return torch.from_numpy(smoothed > 0)
    
    # Find the indices of the two corners in the outline
    corner1 = np.array(cornerNearMirror1)
    corner2 = np.array(cornerNearMirror2)
    
    dist1 = np.sqrt(np.sum((outlinePoints - corner1) ** 2, axis=1))
    dist2 = np.sqrt(np.sum((outlinePoints - corner2) ** 2, axis=1))
    
    idx1 = np.argmin(dist1)
    idx2 = np.argmin(dist2)
    
    # Ensure idx1 < idx2 for consistent processing
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
        corner1, corner2 = corner2, corner1
        curve1, curve2 = curve2, curve1
    
    # Determine the orientation of the contour (clockwise or counter-clockwise)
    # by checking the signed area
    signedArea = 0
    for i in range(numPoints):
        j = (i + 1) % numPoints
        signedArea += outlinePoints[i][0] * outlinePoints[j][1]
        signedArea -= outlinePoints[j][0] * outlinePoints[i][1]
    isClockwise = signedArea < 0
    
    # Decide which section to replace (the shorter path between corners that goes through the mirror area)
    # We'll replace the section from idx1 to idx2 (direct path)
    # Keep points from idx2 to end and from start to idx1
    
    # Points to keep: from idx2 to idx1 (going around)
    if idx2 < numPoints:
        keepPoints = list(outlinePoints[idx2:])
    else:
        keepPoints = []
    keepPoints.extend(list(outlinePoints[:idx1 + 1]))
    
    # Generate new points along the curves
    # From corner1 to intersection along curve1
    intersectX, intersectY = intersectionPoint
    corner1X, corner1Y = corner1
    corner2X, corner2Y = corner2
    
    # Generate points from corner1 to intersection along curve1
    if corner1X != intersectX:
        numNewPoints1 = max(int(abs(intersectX - corner1X)), 10)
        xValues1 = np.linspace(corner1X, intersectX, numNewPoints1)
        yValues1 = evalPolynomial(curve1, xValues1)
        curvePoints1 = np.column_stack([xValues1, yValues1]).astype(int)
    else:
        curvePoints1 = np.array([[int(corner1X), int(corner1Y)], [int(intersectX), int(intersectY)]])
    
    # Generate points from intersection to corner2 along curve2
    if intersectX != corner2X:
        numNewPoints2 = max(int(abs(corner2X - intersectX)), 10)
        xValues2 = np.linspace(intersectX, corner2X, numNewPoints2)
        yValues2 = evalPolynomial(curve2, xValues2)
        curvePoints2 = np.column_stack([xValues2, yValues2]).astype(int)
    else:
        curvePoints2 = np.array([[int(intersectX), int(intersectY)], [int(corner2X), int(corner2Y)]])
    
    # Combine curve points (avoiding duplicate intersection point)
    newCurvePoints = list(curvePoints1) + list(curvePoints2[1:])
    
    # The new outline should maintain the same orientation
    # keepPoints goes from corner2 around to corner1
    # newCurvePoints goes from corner1 to corner2 through the intersection
    
    # Build the new contour: keepPoints (corner2 -> ... -> corner1) + newCurvePoints (corner1 -> intersection -> corner2)
    # But we need to connect them properly
    
    # Actually, let's reconsider:
    # - keepPoints: starts near corner2, goes around, ends near corner1
    # - newCurvePoints: starts at corner1, goes to intersection, ends at corner2
    
    # So the full contour should be: keepPoints + newCurvePoints (reversed to go from corner1 end back to corner2)
    # Wait, that's already the right direction since newCurvePoints ends at corner2 and keepPoints starts at corner2
    
    # Let's connect: newCurvePoints (corner1 -> corner2) + keepPoints (corner2 -> corner1)
    # But keepPoints ends at corner1, so it should be: keepPoints + newCurvePoints
    
    # Actually keepPoints is from idx2 to end + start to idx1, so it goes from corner2 around to corner1
    # newCurvePoints goes from corner1 through intersection to corner2
    # So: at end of keepPoints we're at corner1, newCurvePoints starts at corner1 -> good connection
    # newCurvePoints ends at corner2, keepPoints starts at corner2 -> good connection
    
    newContour = np.array(keepPoints + newCurvePoints, dtype=np.int32)
    
    # Create new mask from the new contour
    newMaskNp = np.zeros_like(maskNp)
    cv2.fillPoly(newMaskNp, [newContour], 255)
    
    smoothed = smoothMask(newMaskNp,smoothingKernel)
    
    return torch.from_numpy(smoothed > 0)


def smoothMask(mask,smoothingKernel=5,iterations=7):
    if iterations < 1: return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smoothingKernel, smoothingKernel))
    smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    for i in range(iterations-1):
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
    return smoothed


def getCornersNearMirror(windowMask, mirrorMask, corners, cornerDistanceThreshold=15):
    """
    Identify and return the first and last corners near the mirror.
    
    Args:
        windowMask: 2D torch tensor of the window mask
        mirrorMask: 2D torch tensor of the mirror mask
        corners: List of (x, y) tuples representing detected corners
        cornerDistanceThreshold: Max distance from mirror for a corner to be considered "close"
        
    Returns:
        Tuple of ((x1, y1), (x2, y2)) for the two corners near the mirror, or None if < 2 corners found
    """
    if mirrorMask is None or len(corners) < 2:
        return None
    
    windowMask = squeezeMask(windowMask)
    mirrorMask = squeezeMask(mirrorMask)
    
    # Get the outline (contour) of the window
    maskNp = windowMask.cpu().numpy().astype(np.uint8) * 255
    contours, _ = cv2.findContours(maskNp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None
    
    contour = max(contours, key=cv2.contourArea)
    outlinePoints = contour.reshape(-1, 2)
    
    # Compute distance transform from mirror mask
    mirrorNp = mirrorMask.cpu().numpy().astype(np.uint8) * 255
    distFromMirror = cv2.distanceTransform(255 - mirrorNp, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    # Find outline indices for all corners
    allCornerIndices = []
    for cx, cy in corners:
        distances = np.sqrt(np.sum((outlinePoints - np.array([cx, cy])) ** 2, axis=1))
        outlineIdx = np.argmin(distances)
        allCornerIndices.append(outlineIdx)
    
    # Sort corners by outline index
    cornersSortedByOutline = sorted(enumerate(corners), key=lambda x: allCornerIndices[x[0]])
    
    # Identify corners close to the mirror
    cornersNearMirror = []
    for sortedPos, (origIdx, (cx, cy)) in enumerate(cornersSortedByOutline):
        if 0 <= cy < distFromMirror.shape[0] and 0 <= cx < distFromMirror.shape[1]:
            dist = distFromMirror[cy, cx]
            if dist < cornerDistanceThreshold:
                cornersNearMirror.append((cx, cy))
    
    if len(cornersNearMirror) < 2:
        return None
    
    # Return first and last corners near mirror
    return (cornersNearMirror[0], cornersNearMirror[-1])


def processAndSaveMasks():
    """
    Process all window masks by extrapolating behind the mirror and save them.
    Saves all window masks (with the front window updated) to PROCESSED_MASK_FOLDER.
    """
    # Create output folder if it doesn't exist
    os.makedirs(PROCESSED_MASK_FOLDER, exist_ok=True)
    
    files = list_image_files(IMAGE_FOLDER)
    for file in files:
        
        baseName = file.split('.')[0]
        windowMaskPath = os.path.join(WINDOW_MASK_FOLDER, f"{baseName}.pt")
        mirrorMaskPath = os.path.join(MIRROR_MASK_FOLDER, f"{baseName}.pt")
        outputPath = os.path.join(PROCESSED_MASK_FOLDER, f"{baseName}.pt")
        
        if not os.path.exists(windowMaskPath):
            continue
        
        # Load window masks - tensor shape is (num_masks, 1, H, W)
        windowMasksTensor = torch.load(windowMaskPath)
        numMasks = windowMasksTensor.shape[0]
        
        # Convert to list of 2D masks for processing
        windowMasks = [squeezeMask(windowMasksTensor[i]) for i in range(numMasks)]
        
        # Load mirror mask to identify front window
        mirrorMask = None
        if os.path.exists(mirrorMaskPath):
            mirrorMasksTensor = torch.load(mirrorMaskPath)
            if mirrorMasksTensor.ndim >= 3 and mirrorMasksTensor.shape[0] > 0:
                mirrorMask = squeezeMask(mirrorMasksTensor[0])
                for i in range(1, mirrorMasksTensor.shape[0]):
                    mirrorMask = torch.logical_or(mirrorMask, squeezeMask(mirrorMasksTensor[i]))
            elif mirrorMasksTensor.ndim == 2:
                mirrorMask = mirrorMasksTensor
        
        # Identify front window (closest to mirror)
        frontWindowIdx = None
        if mirrorMask is not None and len(windowMasks) > 0:
            frontWindowIdx = getFrontWindowIndex(windowMasks, mirrorMask)
        
        # Process front window mask
        updatedFrontMask = None
        if frontWindowIdx is not None:
            frontMask = windowMasks[frontWindowIdx]
            corners = identifyCorners(frontMask)
            
            # Load rotated labels for daylight openings
            daylightOpeningsRect = None
            rotatedLabelPath = os.path.join(ROTATED_LABELS_FOLDER, f"{baseName}.txt")
            if os.path.exists(rotatedLabelPath):
                with open(rotatedLabelPath, 'r') as f:
                    rotatedLabelText = f.read()
                rotatedLabels = rotatedLabelsToDict(rotatedLabelText)
                if DAYLIGHT_OPENINGS_LABEL_KEY in rotatedLabels and rotatedLabels[DAYLIGHT_OPENINGS_LABEL_KEY]:
                    daylightOpeningsNorm = rotatedLabels[DAYLIGHT_OPENINGS_LABEL_KEY][0]
                    # Get image dimensions from mask
                    imgHeight, imgWidth = frontMask.shape[-2:]
                    daylightOpeningsRect = [(x * imgWidth, y * imgHeight) for x, y in daylightOpeningsNorm]
            
            # Get extrapolated curves behind mirror
            if mirrorMask is not None and corners:
                extrapolatedCurves = extrapolateOutlineBehindMirror(frontMask, mirrorMask, corners, useCubic=False)
                cornersNearMirror = getCornersNearMirror(frontMask, mirrorMask, corners)
                
                # Find intersection point
                intersectionPt = None
                if extrapolatedCurves is not None and daylightOpeningsRect is not None:
                    intersectionPt = getIntersectionPoint(
                        extrapolatedCurves[0], 
                        extrapolatedCurves[1], 
                        daylightOpeningsRect, 
                        mirrorMask,
                    )
                
                # Update the mask
                if cornersNearMirror is not None and extrapolatedCurves is not None:
                    updatedFrontMask = updateMask(
                        frontMask,
                        cornersNearMirror[0],
                        cornersNearMirror[1],
                        extrapolatedCurves[0],
                        extrapolatedCurves[1],
                        intersectionPt
                    )
                else:
                    updatedFrontMask = updateMask(frontMask, None, None, None, None, None)
            else:
                updatedFrontMask = updateMask(frontMask, None, None, None, None, None)
        
        # Build output tensor with all masks
        # Keep original shape format (num_masks, 1, H, W)
        # Get device from original tensor
        device = windowMasksTensor.device
        
        if numMasks == 0:
            # No masks to process, save empty tensor
            torch.save(windowMasksTensor, outputPath)
            print(f"Saved (empty): {outputPath}")
            continue
        
        outputMasks = []
        for i in range(numMasks):
            if i == frontWindowIdx and updatedFrontMask is not None:
                # Use updated front mask, add back the channel dimension and move to same device
                outputMasks.append(updatedFrontMask.unsqueeze(0).to(device))
            else:
                # Keep original mask
                outputMasks.append(windowMasksTensor[i])
        
        # Stack all masks into single tensor
        outputTensor = torch.stack(outputMasks, dim=0)
        
        # Save to output folder
        torch.save(outputTensor, outputPath)
    
    print(f"\nProcessing complete. Masks saved to {PROCESSED_MASK_FOLDER}")


def displayMask(baseName):
    """
    Display a single image with window masks overlaid.
    Front window is shown in a different color, with corner markers.
    
    Args:
        baseName: Base file name (without extension) of the image to display
    """
    windowMaskPath = os.path.join(WINDOW_MASK_FOLDER, f"{baseName}.pt")
    mirrorMaskPath = os.path.join(MIRROR_MASK_FOLDER, f"{baseName}.pt")
    imagePath = find_image_path(IMAGE_FOLDER, baseName)
    
    if not os.path.exists(windowMaskPath):
        print(f"Window mask not found: {windowMaskPath}")
        return
    
    if not imagePath or not os.path.exists(imagePath):
        print(f"Image not found for base name: {baseName}")
        return
    
    # Load image
    image = Image.open(imagePath)
    image = image.convert("RGBA")
    
    # Load window masks - tensor shape is (num_masks, 1, H, W)
    windowMasksTensor = torch.load(windowMaskPath)
    # Convert to list of 2D masks
    windowMasks = [squeezeMask(windowMasksTensor[i]) for i in range(windowMasksTensor.shape[0])]
    
    # Load mirror mask to identify front window
    mirrorMask = None
    if os.path.exists(mirrorMaskPath):
        mirrorMasksTensor = torch.load(mirrorMaskPath)
        if mirrorMasksTensor.ndim >= 3 and mirrorMasksTensor.shape[0] > 0:
            # Combine all mirror masks into one (along the first dimension)
            mirrorMask = squeezeMask(mirrorMasksTensor[0])
            for i in range(1, mirrorMasksTensor.shape[0]):
                mirrorMask = torch.logical_or(mirrorMask, squeezeMask(mirrorMasksTensor[i]))
        elif mirrorMasksTensor.ndim == 2:
            mirrorMask = mirrorMasksTensor
    
    # Identify front window (closest to mirror)
    frontWindowIdx = None
    if mirrorMask is not None and len(windowMasks) > 0:
        frontWindowIdx = getFrontWindowIndex(windowMasks, mirrorMask)
    
    # Process front window mask - get updated mask with extrapolation
    updatedFrontMask = None
    corners = None
    extrapolatedCurves = None
    intersectionPt = None
    cornersNearMirror = None
    daylightOpeningsRect = None
    
    if frontWindowIdx is not None:
        frontMask = windowMasks[frontWindowIdx]
        corners = identifyCorners(frontMask)
        
        # Load rotated labels for daylight openings
        rotatedLabelPath = os.path.join(ROTATED_LABELS_FOLDER, f"{baseName}.txt")
        if os.path.exists(rotatedLabelPath):
            with open(rotatedLabelPath, 'r') as f:
                rotatedLabelText = f.read()
            rotatedLabels = rotatedLabelsToDict(rotatedLabelText)
            if DAYLIGHT_OPENINGS_LABEL_KEY in rotatedLabels and rotatedLabels[DAYLIGHT_OPENINGS_LABEL_KEY]:
                daylightOpeningsNorm = rotatedLabels[DAYLIGHT_OPENINGS_LABEL_KEY][0]
                imgWidth, imgHeight = image.size
                daylightOpeningsRect = [(x * imgWidth, y * imgHeight) for x, y in daylightOpeningsNorm]
        
        # Get extrapolated curves behind mirror
        if mirrorMask is not None and corners:
            extrapolatedCurves = extrapolateOutlineBehindMirror(frontMask, mirrorMask, corners, useCubic=False)
            cornersNearMirror = getCornersNearMirror(frontMask, mirrorMask, corners)
            
            # Find intersection point
            if extrapolatedCurves is not None and daylightOpeningsRect is not None:
                intersectionPt = getIntersectionPoint(
                    extrapolatedCurves[0], 
                    extrapolatedCurves[1], 
                    daylightOpeningsRect, 
                    mirrorMask,
                )
            
            # Update the mask
            if cornersNearMirror is not None and extrapolatedCurves is not None:
                updatedFrontMask = updateMask(
                    frontMask,
                    cornersNearMirror[0],
                    cornersNearMirror[1],
                    extrapolatedCurves[0],
                    extrapolatedCurves[1],
                    intersectionPt
                )
            else:
                # Just smooth the original mask
                updatedFrontMask = updateMask(frontMask, None, None, None, None, None)
        else:
            # Just smooth the original mask
            updatedFrontMask = updateMask(frontMask, None, None, None, None, None)
    
    # Create overlay for masks
    maskOverlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    
    # Colors
    #frontWindowColor = (0, 255, 0, 128)  # Green for front window
    frontWindowColor = (255, 0, 0, 128)
    otherWindowColor = (255, 0, 0, 128)  # Red for other windows
    
    # Draw each window mask
    for i, mask in enumerate(windowMasks):
        # Use updated mask for front window
        #if i == frontWindowIdx and updatedFrontMask is not None:
        #    maskNp = updatedFrontMask.cpu().numpy() > 0
        #else:
        #    maskNp = mask.cpu().numpy() > 0
        maskNp = mask.cpu().numpy() > 0
        
        color = frontWindowColor if i == frontWindowIdx else otherWindowColor
        
        maskUint8 = (maskNp * 255).astype(np.uint8)
        maskPil = Image.fromarray(maskUint8, mode='L')
        singleMaskImg = Image.new("RGBA", image.size, (0, 0, 0, 0))
        singleMaskImg.paste(color, (0, 0), mask=maskPil)
        maskOverlay = Image.alpha_composite(maskOverlay, singleMaskImg)
    
    # Combine image with mask overlay
    resultImage = Image.alpha_composite(image, maskOverlay)
    
    # Convert to numpy for matplotlib
    resultNp = np.array(resultImage)
    
    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(resultNp)
    
    # Add corner markers for front window
    if frontWindowIdx is not None and corners:
        # Plot corner markers
        cornerX = [c[0] for c in corners]
        cornerY = [c[1] for c in corners]
        #plt.scatter(cornerX, cornerY, c='yellow', s=30, marker='o', edgecolors='black', linewidths=1, zorder=5)
        
        # Plot extrapolated curves if available
        if extrapolatedCurves is not None:
            imgWidth, imgHeight = image.size
            xValues = np.arange(0, imgWidth)
            
            # Plot first curve (before first corner)
            y1 = evalPolynomial(extrapolatedCurves[0], xValues)
            # Clip to image bounds
            mask1 = (y1 >= 0) & (y1 < imgHeight)
            plt.plot(xValues[mask1], y1[mask1], 'c-', linewidth=5, label='Curve 1 (before)', zorder=6)
            
            # Plot second curve (after last corner)
            y2 = evalPolynomial(extrapolatedCurves[1], xValues)
            # Clip to image bounds
            mask2 = (y2 >= 0) & (y2 < imgHeight)
            plt.plot(xValues[mask2], y2[mask2], 'm-', linewidth=5, label='Curve 2 (after)', zorder=6)
            
            # Plot intersection point if found
            if intersectionPt is not None:
                plt.scatter([intersectionPt[0]], [intersectionPt[1]], 
                           c='white', s=100, marker='*', edgecolors='black', 
                           linewidths=2, zorder=7, label='Intersection')
            
            plt.legend(loc='upper right')
    
    plt.title(f"{baseName} - Front window: green, Others: red, Corners: yellow")
    plt.axis('off')
    plt.xlim(0, image.size[0])
    plt.ylim(image.size[1], 0)  # Inverted y-axis for image coordinates
    plt.show()


def displayProcessedMasks():
    """
    Display images with window masks overlaid.
    Iterates through all images and calls displayMask for each.
    """
    files = list_image_files(IMAGE_FOLDER)
    shuffle(files)
    for file in files:
        
        baseName = file.split('.')[0]
        displayMask(baseName)


if __name__ == "__main__":
    processAndSaveMasks()
    displayProcessedMasks()
