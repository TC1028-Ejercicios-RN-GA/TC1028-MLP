#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 18:38:00 2020

@author: avmejia
"""

import unittest

import mlp


class Testprueba (unittest.TestCase):
    
    def test_codificacion(self):
        x=[1,0.5,1.3,0.6]
        w=[[0.2,0.4],
           [-1.2,2.3],
           [0.5,-0.5],
           [1.1,2]]
        self.assertEqual(mlp.salida_capa(x,w),[1.402524224033636, 1.1224564282529819])
        
        

if __name__ == '__main__':
    unittest.main()
