import logging
import sys
from unittest import TestCase

from mlops_examples.domain.data_generation.mandelbrot_generator import MandelbrotGenerator
from tests.fixtures.timer import Timer


class TestMandelbrotGenerator(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    def test_should_generate_mandelbrot(self):
        # act
        with Timer("mandelbrot rows=3 max=25"):
            result = MandelbrotGenerator().get(3)
            print(result)
        with Timer("mandelbrot rows=5000 max=25"):
            result = MandelbrotGenerator().get(5000)
        with Timer("mandelbrot rows=500000 max=255"):
            result = MandelbrotGenerator().with_max(255).get(500000)
