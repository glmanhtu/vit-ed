"""
Created by Zayd Hammoudeh (zayd.hammoudeh@sjsu.edu)
"""
import random

from enum import Enum

import numpy
import cv2  # Open CV


class Location(object):
    """
    Location Object

    Used to represent any two dimensional location in matrix row/column notation.
    """

    def __init__(self, coord):
        (row, column) = coord
        self.row = row
        self.column = column


class PuzzlePieceRotation(Enum):
    """Puzzle Piece PieceRotation

    Enumerated type for representing the amount of rotation for a puzzle piece.

    Note:
        Pieces can only be rotated in 90 degree increments.

    """

    degree_0 = 0      # No rotation
    degree_90 = 90    # 90 degree rotation
    degree_180 = 180  # 180 degree rotation
    degree_270 = 270  # 270 degree rotation
    degree_360 = 360

    @staticmethod
    def all_rotations():
        """
        All Rotation Accessor

        Gets a list of all supported rotations for a puzzle piece.  The list is ascending from 0 degrees to 270
        degrees increasing.

        Returns ([PuzzlePieceRotation]):
        List of all puzzle rotations.
        """
        return [PuzzlePieceRotation.degree_0, PuzzlePieceRotation.degree_90,
                PuzzlePieceRotation.degree_180, PuzzlePieceRotation.degree_270]

    @staticmethod
    def random_rotation():
        """
        Random Rotation

        Generates and returns a random rotation.

        Returns (PuzzlePieceRotation):
        A random puzzle piece rotation
        """
        return random.choice(PuzzlePieceRotation.all_rotations())


class PuzzlePieceSide(Enum):
    """Puzzle Piece Side

    Enumerated type for representing the four sides of the a puzzle piece.

    Note:
        Pieces can only be rotated in 90 degree increments.

    """

    top = 0
    right = 1
    bottom = 2
    left = 3

    @staticmethod
    def get_numb_sides():
        """
        Accessor for the number of sizes for a puzzle piece.

        Returns (int):
        Since these are rectangular pieces, it returns size 4.  This is currently fixed.
        """
        return 4

    @staticmethod
    def get_all_sides():
        """
        Static method to extract all the sides of a piece.

        Returns ([PuzzlePieceSide]):
        List of all sides of a puzzle piece starting at the top and moving clockwise.
        """
        return [PuzzlePieceSide.top, PuzzlePieceSide.right, PuzzlePieceSide.bottom, PuzzlePieceSide.left]

    @property
    def complementary_side(self):
        """
        Determines and returns the complementary side of this implicit side parameter.  For example, if this side
        is "left" then the function returns "right" and vice versa.

        Returns (PuzzlePieceSide):
        Complementary side of the piece.
        """
        if self == PuzzlePieceSide.top:
            return PuzzlePieceSide.bottom

        if self == PuzzlePieceSide.right:
            return PuzzlePieceSide.left

        if self == PuzzlePieceSide.bottom:
            return PuzzlePieceSide.top

        if self == PuzzlePieceSide.left:
            return PuzzlePieceSide.right

    @property
    def side_name(self):
        """
        Gets the name of a puzzle piece side without the class name

        Returns (str):
            The name of the side as a string
        """
        return str(self).split(".")[1]


class PuzzlePiece(object):
    """
    Puzzle Piece Object.  It is a very simple object that stores the puzzle piece's pixel information in a
    NumPY array.  It also stores the piece's original information (e.g. X/Y location and puzzle ID) along with
    what was determined by the solver.
    """

    NUMB_LAB_COLORSPACE_DIMENSIONS = 3

    _PERFORM_ASSERTION_CHECKS = True

    def __init__(self, puzzle_id, location, lab_img, piece_id=None, puzzle_grid_size=None):
        """
        Puzzle Piece Constructor.

        Args:
            puzzle_id (int): Puzzle identification number
            location ([int]): (row, column) location of this piece.
            lab_img: Image data in the form of a numpy array.
            piece_id (int): Piece identification number.
            puzzle_grid_size ([int]): Grid size of the puzzle
        """

        # Verify the piece id information
        if piece_id is None and puzzle_grid_size is not None:
            raise ValueError("Using the puzzle grid size is not supported if piece id is \"None\".")

        # Piece ID is left to the solver to set
        self._piece_id = piece_id
        self.origin_piece_id = piece_id
        self._orig_piece_id = piece_id

        self._orig_puzzle_id = puzzle_id
        self._assigned_puzzle_id = None

        # Store the original location of the puzzle piece and initialize a placeholder x/y location.
        self._orig_loc = location
        self._assigned_loc = None

        # Store the image data
        self._img = lab_img
        (length, width, dim) = self._img.shape
        if width != length:
            raise ValueError("Only square puzzle pieces are supported at this time.")
        if dim != PuzzlePiece.NUMB_LAB_COLORSPACE_DIMENSIONS:
            raise ValueError("This image does not appear to be in the LAB colorspace as it does not have 3 dimensions")
        self._width = width

        # Rotation gets set later.
        self._rotation = None
        self._actual_neighbor_ids = None
        if puzzle_grid_size is not None:
            self.calculate_actual_neighbor_id_numbers(puzzle_grid_size)

    def calculate_actual_neighbor_id_numbers(self, puzzle_grid_size):
        """
        Neighbor ID Calculator

        Given a grid size, this function calculates the identification number of this piece's neighbors.  If a piece
        has no neighbor, then location associated with that puzzle piece is filled with "None".

        Args:
            puzzle_grid_size (List[int]): Grid size (number of rows, number of columns) for this piece's puzzle.
        """

        # Only need to calculate the actual neighbor id information once
        if self._actual_neighbor_ids is not None:
            return
        # Initialize actual neighbor id information
        self._actual_neighbor_ids = []

        # Extract the information on the puzzle grid size
        (numb_rows, numb_cols) = puzzle_grid_size

        # Check the top location first
        # If the row is 0, then it has no top neighbor
        if self._orig_loc[0] == 0:
            neighbor_id = None
        else:
            neighbor_id = self._orig_piece_id - numb_cols
        self._actual_neighbor_ids.append((neighbor_id, PuzzlePieceSide.top))

        # Check the right side
        # If in the last column, it has no right neighbor
        if self._orig_loc[1] + 1 == numb_cols:
            neighbor_id = None
        else:
            neighbor_id = self._orig_piece_id + 1
        self._actual_neighbor_ids.append((neighbor_id, PuzzlePieceSide.right))

        # Check the bottom side
        # If in the last column, it has no right neighbor
        if self._orig_loc[0] + 1 == numb_rows:
            neighbor_id = None
        else:
            neighbor_id = self._orig_piece_id + numb_cols
        self._actual_neighbor_ids.append((neighbor_id, PuzzlePieceSide.bottom))

        # Check the right side
        # If in the last column, it has no left neighbor
        if self._orig_loc[1] == 0:
            neighbor_id = None
        else:
            neighbor_id = self._orig_piece_id - 1
        self._actual_neighbor_ids.append((neighbor_id, PuzzlePieceSide.left))

        # Convert the list to a tuple since it is immutable
        self._actual_neighbor_ids = tuple(self._actual_neighbor_ids)

    def is_correctly_placed(self, puzzle_offset_upper_left_location):
        """
        Piece Placement Checker

        Checks whether the puzzle piece is correctly placed.

        Args:
            puzzle_offset_upper_left_location (Tuple[int]): Modified location for the origin of the puzzle

        Returns (bool):
            True if the puzzle piece is in the correct location and False otherwise.
        """

        # Verify all dimensions
        for i in range(0, len(self._orig_loc)):
            # If for the current dimension
            if self._assigned_loc[i] - self._orig_loc[i] - puzzle_offset_upper_left_location[i] != 0:
                return False
        # Mark as correctly placed
        return True

    def is_neighbor(self, piece, side: PuzzlePieceSide):
        current_loc = self._orig_loc
        other_loc = piece._orig_loc
        if side == PuzzlePieceSide.top:
            return (current_loc[1] == other_loc[1]) and (current_loc[0] - other_loc[0] == 1)
        if side == PuzzlePieceSide.bottom:
            return (current_loc[1] == other_loc[1]) and (other_loc[0] - current_loc[0] == 1)
        if side == PuzzlePieceSide.left:
            return (current_loc[0] == other_loc[0]) and (current_loc[1] - other_loc[1] == 1)
        if side == PuzzlePieceSide.right:
            return (current_loc[0] == other_loc[0]) and (other_loc[1] - current_loc[1] == 1)
        raise Exception(f'Side {side} does not exists!')

    @property
    def original_neighbor_id_numbers_and_sides(self):
        """
        Neighbor Identification Number Property

        In a puzzle, each piece has up to four neighbors.  This function access that identification number information.

        Returns (List[int, PuzzlePieceSide]):
            Identification number for the puzzle piece on the specified side of the original object.

        """
        # Verify that the array containing the neighbor id numbers is not none
        assert self._actual_neighbor_ids is not None

        # Return the piece's neighbor identification numbers
        return self._actual_neighbor_ids

    @property
    def original_puzzle_id(self):
        """
        Direct Accuracy Original Puzzle ID Number Accessor

        Returns (int):
            The puzzle ID associated with the ORIGINAL set of puzzle results
        """
        return self._orig_puzzle_id

    @property
    def original_piece_id(self):
        """
        Original Piece ID Number

        Gets the original (i.e., correct) piece identification number

        Returns (int):
            Original identification number assigned to the piece at its creation.  Should be globally unique.
        """
        return self._orig_piece_id

    @property
    def width(self):
        """
        Gets the size of the square puzzle piece.  Since it is square, width its width equals its length.

        Returns (int): Width of the puzzle piece in pixels.

        """
        return self._width

    @property
    def location(self):
        """
        Gets the location of the puzzle piece on the board.

        Returns ([int]): Tuple of the (row, column)

        """
        return self._assigned_loc

    @location.setter
    def location(self, new_loc):
        """
        Updates the puzzle piece location.

        Args:
            new_loc ([int]): New puzzle piece location.

        """
        if len(new_loc) != 2:
            raise ValueError("Location of a puzzle piece must be a two dimensional tuple")
        self._assigned_loc = new_loc

    @property
    def puzzle_id(self):
        """
        Gets the location of the puzzle piece on the board.

        Returns (int): Assigned Puzzle ID number.

        """
        return self._assigned_puzzle_id

    @puzzle_id.setter
    def puzzle_id(self, new_puzzle_id):
        """
        Updates the puzzle ID number for the puzzle piece.

        Returns (int): Board identification number

        """
        self._assigned_puzzle_id = new_puzzle_id

    @property
    def id_number(self):
        """
        Puzzle Piece ID Getter

        Gets the identification number for a puzzle piece.

        Returns (int): Puzzle piece indentification number
        """
        return self._piece_id

    @id_number.setter
    def id_number(self, new_piece_id):
        """
        Piece ID Setter

        Sets the puzzle piece's identification number.

        Args:
            new_piece_id (int): Puzzle piece identification number
        """
        self._piece_id = new_piece_id

    @property
    def lab_image(self):
        """
        Get's a puzzle piece's image in the LAB colorspace.

        Returns:
        Numpy array of the piece's lab image.
        """
        return self._img

    @property
    def rotation(self):
        """
        Rotation Accessor

        Gets the puzzle piece's rotation.

        Returns (PuzzlePieceRotation):

        The puzzle piece's rotation
        """
        return self._rotation

    @rotation.setter
    def rotation(self, new_rotation):
        """
        Puzzle Piece Rotation Setter

        Updates a puzzle piece's rotation.

        Args:
            new_rotation (PuzzlePieceRotation): New rotation for the puzzle piece.
        """
        self._rotation = new_rotation

    def get_neighbor_locations_and_sides(self):
        """
        Neighbor Locations and Sides

        Given a puzzle piece, this function returns the four surrounding coordinates/location and the sides of this
        puzzle piece that corresponds to those locations so that it can be added to the open slot list.

        Returns ([([int], PuzzlePieceSide)]): Valid puzzle piece locations and the respective puzzle
        piece side.
        """

        if PuzzlePiece._PERFORM_ASSERTION_CHECKS:
            assert self.location is not None
            assert self.rotation is not None

        return PuzzlePiece._get_neighbor_locations_and_sides(self.location, self.rotation)

    @staticmethod
    def _get_neighbor_locations_and_sides(piece_loc, piece_rotation):
        """
        Neighbor Locations and Sides

        Static method that given a piece location and rotation, it returns the four surrounding coordinates/location
        and the puzzle piece side that aligns with it so that it can be added to the open slot list.

        Args:
            piece_loc ([int]):
            piece_rotation (PuzzlePieceRotation):

        Returns ([([int], PuzzlePieceSide)]): Valid puzzle piece locations and the respective puzzle
        piece side.
        """
        # Get the top location and respective side
        top_loc = (piece_loc[0] - 1, piece_loc[1])
        # noinspection PyTypeChecker
        location_piece_side_tuples = [(top_loc, PuzzlePiece._determine_unrotated_side(piece_rotation,
                                                                                      PuzzlePieceSide.top))]
        # Get the right location and respective side
        right_loc = (piece_loc[0], piece_loc[1] + 1)
        # noinspection PyTypeChecker
        location_piece_side_tuples.append((right_loc, PuzzlePiece._determine_unrotated_side(piece_rotation,
                                                                                            PuzzlePieceSide.right)))
        # Get the bottom location and its respective side
        bottom_loc = (piece_loc[0] + 1, piece_loc[1])
        # noinspection PyTypeChecker
        location_piece_side_tuples.append((bottom_loc, PuzzlePiece._determine_unrotated_side(piece_rotation,
                                                                                             PuzzlePieceSide.bottom)))
        # Get the right location and respective side
        left_loc = (piece_loc[0], piece_loc[1] - 1)
        # noinspection PyTypeChecker
        location_piece_side_tuples.append((left_loc, PuzzlePiece._determine_unrotated_side(piece_rotation,
                                                                                           PuzzlePieceSide.left)))
        # Return the location/piece side tuples
        return location_piece_side_tuples

    def bgr_image(self):
        """
        Get's a puzzle piece's image in the BGR colorspace.

        Returns:
        Numpy array of the piece's BGR image.
        """
        return cv2.cvtColor(self._img, cv2.COLOR_LAB2BGR)

    def get_row_pixels(self, row_numb, reverse=False):
        """
        Extracts a row of pixels from a puzzle piece.

        Args:
            row_numb (int): Pixel row in the image.  Must be between 0 and the width of the piece - 1 (inclusive).
            reverse (Optional bool): Select whether to reverse the pixel information.

        Returns:

        """
        if row_numb < 0:
            raise ValueError("Row number for a piece must be greater than or equal to zero.")
        if row_numb >= self._width:
            raise ValueError("Row number for a piece must be less than the puzzle's pieces width")

        if reverse:
            return self._img[row_numb, ::-1, :]
        else:
            return self._img[row_numb, :, :]

    def get_column_pixels(self, col_numb, reverse=False):
        """
        Extracts a row of pixels from a puzzle piece.

        Args:
            col_numb (int): Pixel column in the image.  Must be between 0 and the width of the piece - 1 (inclusive).
            reverse (Optional bool): Select whether to reverse the pixel information.

        Returns:

        """
        if col_numb < 0:
            raise ValueError("Column number for a piece must be greater than or equal to zero.")
        if col_numb >= self._width:
            raise ValueError("Column number for a piece must be less than the puzzle's pieces width")
        # If you reverse, change the order of the pixels.
        if reverse:
            return self._img[::-1, col_numb, :]
        else:
            return self._img[:, col_numb, :]

    @staticmethod
    def calculate_asymmetric_distance(piece_i, piece_i_side, piece_j, piece_j_side):
        """
        Uses the Asymmetric Distance function to calculate the distance between two puzzle pieces.

        Args:
            piece_i (PuzzlePiece):
            piece_i_side (PuzzlePieceSide):
            piece_j (PuzzlePiece):
            piece_j_side (PuzzlePieceSide): Side of piece j that is adjacent to piece i.

        Returns (double):
            Distance between
        """

        # Get the border and second to last ROW on the TOP side of piece i
        if piece_i_side == PuzzlePieceSide.top:
            i_border = piece_i.get_row_pixels(0)
            i_second_to_last = piece_i.get_row_pixels(1)

        # Get the border and second to last COLUMN on the RIGHT side of piece i
        elif piece_i_side == PuzzlePieceSide.right:
            i_border = piece_i.get_column_pixels(piece_i.width - 1)
            i_second_to_last = piece_i.get_column_pixels(piece_i.width - 2)

        # Get the border and second to last ROW on the BOTTOM side of piece i
        elif piece_i_side == PuzzlePieceSide.bottom:
            i_border = piece_i.get_row_pixels(piece_i.width - 1)
            i_second_to_last = piece_i.get_row_pixels(piece_i.width - 2)

        # Get the border and second to last COLUMN on the LEFT side of piece i
        elif piece_i_side == PuzzlePieceSide.left:
            i_border = piece_i.get_column_pixels(0)
            i_second_to_last = piece_i.get_column_pixels(1)
        else:
            raise ValueError("Invalid edge for piece i")

        # If rotation is allowed need to reverse pixel order in some cases.
        reverse = False  # By default do not reverse
        # Always need to reverse when they are the same side
        if piece_i_side == piece_j_side:
            reverse = True
        # Get the pixels along the TOP of piece_j
        if piece_j_side == PuzzlePieceSide.top:
            if piece_i_side == PuzzlePieceSide.right:
                reverse = True
            j_border = piece_j.get_row_pixels(0, reverse)

        # Get the pixels along the RIGHT of piece_j
        elif piece_j_side == PuzzlePieceSide.right:
            if piece_i_side == PuzzlePieceSide.top:
                reverse = True
            j_border = piece_j.get_column_pixels(piece_i.width - 1, reverse)

        # Get the pixels along the BOTTOM of piece_j
        elif piece_j_side == PuzzlePieceSide.bottom:
            if piece_i_side == PuzzlePieceSide.left:
                reverse = True
            j_border = piece_j.get_row_pixels(piece_i.width - 1, reverse)

        # Get the pixels along the RIGHT of piece_j
        elif piece_j_side == PuzzlePieceSide.left:
            if piece_i_side == PuzzlePieceSide.bottom:
                reverse = True
            j_border = piece_j.get_column_pixels(0, reverse)
        else:
            raise ValueError("Invalid edge for piece i")

        # Calculate the value of pixels on piece j's edge.
        predicted_j = 2 * (i_border.astype(numpy.int16)) - i_second_to_last.astype(numpy.int16)
        # noinspection PyUnresolvedReferences
        pixel_diff = predicted_j.astype(numpy.int16) - j_border.astype(numpy.int16)

        # Return the sum of the absolute values.
        pixel_diff = numpy.absolute(pixel_diff)
        return numpy.sum(pixel_diff, dtype=numpy.int32)

    def set_placed_piece_rotation(self, placed_side, neighbor_piece_side, neighbor_piece_rotation):
        """
        Placed Piece Rotation Setter

        Given an already placed neighbor piece's adjacent side and rotation, this function sets the rotation
        of some newly placed piece that is put adjacent to that neighbor piece.

        Args:
            placed_side (PuzzlePieceSide): Side of the placed puzzle piece that is adjacent to the neighbor piece

            neighbor_piece_side (PuzzlePieceSide): Side of the neighbor piece that is adjacent to the newly
            placed piece.

            neighbor_piece_rotation (PuzzlePieceRotation): Rotation of the already placed neighbor piece
        """
        # Calculate the placed piece's new rotation
        self.rotation = PuzzlePiece._calculate_placed_piece_rotation(placed_side, neighbor_piece_side,
                                                                     neighbor_piece_rotation)

    @staticmethod
    def _calculate_placed_piece_rotation(placed_piece_side, neighbor_piece_side, neighbor_piece_rotation):
        """
        Placed Piece Rotation Calculator

        Given an already placed neighbor piece, this function determines the correct rotation for a newly placed
        piece.

        Args:
            placed_piece_side (PuzzlePieceSide): Side of the placed puzzle piece adjacent to the existing piece
            neighbor_piece_side (PuzzlePieceSide): Side of the neighbor of the placed piece that is touching
            neighbor_piece_rotation (PuzzlePieceRotation): Rotation of the neighbor piece

        Returns (PuzzlePieceRotation): Rotation of the placed puzzle piece given the rotation and side
        of the neighbor piece.
        """
        # Get the neighbor piece rotation
        unrotated_complement = neighbor_piece_side.complementary_side

        placed_rotation_val = int(neighbor_piece_rotation.value)
        # noinspection PyUnresolvedReferences
        placed_rotation_val += 90 * (PuzzlePieceRotation.degree_360.value + (unrotated_complement.value
                                                                             - placed_piece_side.value))
        # Calculate the normalized rotation
        # noinspection PyUnresolvedReferences
        placed_rotation_val %= PuzzlePieceRotation.degree_360.value
        # Check if a valid rotation value.
        if PuzzlePiece._PERFORM_ASSERTION_CHECKS:
            assert placed_rotation_val % 90 == 0
        # noinspection PyUnresolvedReferences
        return PuzzlePieceRotation(placed_rotation_val % PuzzlePieceRotation.degree_360.value)

    @staticmethod
    def _determine_unrotated_side(piece_rotation, rotated_side):
        """
        Unrotated Side Determiner

        Given a piece's rotation and the side of the piece (from the reference of the puzzle), find its actual
        (i.e. unrotated) side.

        Args:
            piece_rotation (PuzzlePieceRotation): Specified rotation for a puzzle piece.
            rotated_side (PuzzlePieceSide): From a Puzzle perspective, this is the exposed side

        Returns(PuzzlePieceSide): Actual side of the puzzle piece
        """
        rotated_side_val = rotated_side.value
        # Get the number of 90 degree rotations
        numb_90_degree_rotations = int(piece_rotation.value / 90)

        # Get the unrotated side
        unrotated_side = (rotated_side_val + (PuzzlePieceSide.get_numb_sides() - numb_90_degree_rotations))
        unrotated_side %= PuzzlePieceSide.get_numb_sides()

        # Return the actual side
        return PuzzlePieceSide(unrotated_side)

    @staticmethod
    def _get_neighbor_piece_rotated_side(placed_piece_loc, neighbor_piece_loc):
        """

        Args:
            placed_piece_loc ([int]): Location of the newly placed piece
            neighbor_piece_loc ([int): Location of the neighbor of the newly placed piece

        Returns (PuzzlePieceSide): Side of the newly placed piece where the placed piece is now location.

        Notes: This does not take into account any rotation of the neighbor piece.  That is why this function is
        referred has "rotated side" in its name.
        """
        # Calculate the row and column distances
        row_dist = placed_piece_loc[0] - neighbor_piece_loc[0]
        col_dist = placed_piece_loc[1] - neighbor_piece_loc[1]

        # Perform some checking on the pieces
        if PuzzlePiece._PERFORM_ASSERTION_CHECKS:
            # Verify the pieces are in the same puzzle
            assert abs(row_dist) + abs(col_dist) == 1

        # Determine the relative side of the placed piece
        if row_dist == -1:
            return PuzzlePieceSide.top
        elif row_dist == 1:
            return PuzzlePieceSide.bottom
        elif col_dist == -1:
            return PuzzlePieceSide.left
        else:
            return PuzzlePieceSide.right
