# Licensed under a 3-clause BSD style license - see LICENSE.rst

# This file was automatically generated from ply. To re-generate this file,
# remove it from this folder, then build astropy and run the tests in-place:
#
#   python setup.py build_ext --inplace
#   pytest astropy/units
#
# You can then commit the changes to this file.


# generic_parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'CLOSE_PAREN COMMA DIVISION FUNCNAME OPEN_PAREN POWER PRODUCT SIGN UFLOAT UINT UNIT\n            main : unit\n                 | structured_unit\n                 | structured_subunit\n            \n            structured_subunit : OPEN_PAREN structured_unit CLOSE_PAREN\n            \n            structured_unit : subunit COMMA\n                            | subunit COMMA subunit\n            \n            subunit : unit\n                    | structured_unit\n                    | structured_subunit\n            \n            unit : product_of_units\n                 | factor product_of_units\n                 | factor PRODUCT product_of_units\n                 | division_product_of_units\n                 | factor division_product_of_units\n                 | factor PRODUCT division_product_of_units\n                 | inverse_unit\n                 | factor inverse_unit\n                 | factor PRODUCT inverse_unit\n                 | factor\n            \n            division_product_of_units : division_product_of_units DIVISION product_of_units\n                                      | product_of_units\n            \n            inverse_unit : DIVISION unit_expression\n            \n            factor : factor_fits\n                   | factor_float\n                   | factor_int\n            \n            factor_float : signed_float\n                         | signed_float UINT signed_int\n                         | signed_float UINT POWER numeric_power\n            \n            factor_int : UINT\n                       | UINT signed_int\n                       | UINT POWER numeric_power\n                       | UINT UINT signed_int\n                       | UINT UINT POWER numeric_power\n            \n            factor_fits : UINT POWER OPEN_PAREN signed_int CLOSE_PAREN\n                        | UINT POWER OPEN_PAREN UINT CLOSE_PAREN\n                        | UINT POWER signed_int\n                        | UINT POWER UINT\n                        | UINT SIGN UINT\n                        | UINT OPEN_PAREN signed_int CLOSE_PAREN\n            \n            product_of_units : unit_expression PRODUCT product_of_units\n                             | unit_expression product_of_units\n                             | unit_expression\n            \n            unit_expression : function\n                            | unit_with_power\n                            | OPEN_PAREN product_of_units CLOSE_PAREN\n            \n            unit_with_power : UNIT POWER numeric_power\n                            | UNIT numeric_power\n                            | UNIT\n            \n            numeric_power : sign UINT\n                          | OPEN_PAREN paren_expr CLOSE_PAREN\n            \n            paren_expr : sign UINT\n                       | signed_float\n                       | frac\n            \n            frac : sign UINT DIVISION sign UINT\n            \n            sign : SIGN\n                 |\n            \n            signed_int : SIGN UINT\n            \n            signed_float : sign UINT\n                         | sign UFLOAT\n            \n            function : FUNCNAME OPEN_PAREN main CLOSE_PAREN\n            '
    
_lr_action_items = {'OPEN_PAREN':([0,6,10,11,12,13,14,15,16,17,18,20,21,22,25,28,29,30,31,36,40,42,45,46,47,50,51,60,62,63,65,67,68,71,72,73,75,76,81,82,85,86,87,88,90,91,],[10,28,31,28,-23,-24,-25,28,-43,-44,41,-26,45,49,28,28,28,10,31,28,66,-30,10,49,-47,-58,-59,-45,-32,49,-37,-36,-31,-38,-27,49,-46,-49,-33,-57,-39,-28,-60,-50,-35,-34,]),'DIVISION':([0,5,6,7,10,11,12,13,14,16,17,18,20,22,24,25,26,30,31,33,37,42,45,47,50,51,52,53,56,60,61,62,65,67,68,71,72,75,76,81,82,85,86,87,88,89,90,91,],[15,-21,15,29,15,-42,-23,-24,-25,-43,-44,-29,-26,-48,-21,15,29,15,15,-21,-41,-30,15,-47,-58,-59,-21,29,-20,-45,-40,-32,-37,-36,-31,-38,-27,-46,-49,-33,-57,-39,-28,-60,-50,92,-35,-34,]),'UINT':([0,10,18,19,20,22,23,30,31,40,43,45,46,48,49,50,51,63,64,66,69,73,78,92,93,],[18,18,39,-55,44,-56,50,18,18,65,71,18,-56,76,-56,-58,-59,-56,82,83,82,-56,89,-56,94,]),'FUNCNAME':([0,6,10,11,12,13,14,15,16,17,18,20,22,25,28,29,30,31,36,42,45,47,50,51,60,62,65,67,68,71,72,75,76,81,82,85,86,87,88,90,91,],[21,21,21,21,-23,-24,-25,21,-43,-44,-29,-26,-48,21,21,21,21,21,21,-30,21,-47,-58,-59,-45,-32,-37,-36,-31,-38,-27,-46,-49,-33,-57,-39,-28,-60,-50,-35,-34,]),'UNIT':([0,6,10,11,12,13,14,15,16,17,18,20,22,25,28,29,30,31,36,42,45,47,50,51,60,62,65,67,68,71,72,75,76,81,82,85,86,87,88,90,91,],[22,22,22,22,-23,-24,-25,22,-43,-44,-29,-26,-48,22,22,22,22,22,22,-30,22,-47,-58,-59,-45,-32,-37,-36,-31,-38,-27,-46,-49,-33,-57,-39,-28,-60,-50,-35,-34,]),'SIGN':([0,10,18,22,30,31,39,40,41,44,45,46,49,63,66,73,92,],[19,19,43,19,19,19,64,69,64,64,19,19,19,19,69,19,19,]),'UFLOAT':([0,10,19,23,30,31,45,49,66,69,78,],[-56,-56,-55,51,-56,-56,-56,-56,-56,-55,51,]),'$end':([1,2,3,4,5,6,7,8,11,12,13,14,16,17,18,20,22,24,26,27,30,34,35,37,38,42,47,50,51,52,53,54,56,57,58,59,60,61,62,65,67,68,71,72,75,76,81,82,85,86,87,88,90,91,],[0,-1,-2,-3,-10,-19,-13,-16,-42,-23,-24,-25,-43,-44,-29,-26,-48,-11,-14,-17,-5,-7,-9,-41,-22,-30,-47,-58,-59,-12,-15,-18,-20,-6,-8,-4,-45,-40,-32,-37,-36,-31,-38,-27,-46,-49,-33,-57,-39,-28,-60,-50,-35,-34,]),'CLOSE_PAREN':([2,3,4,5,6,7,8,11,12,13,14,16,17,18,20,22,24,26,27,30,32,33,34,35,37,38,42,47,50,51,52,53,54,55,56,57,58,59,60,61,62,65,67,68,70,71,72,74,75,76,77,79,80,81,82,83,84,85,86,87,88,89,90,91,94,],[-1,-2,-3,-10,-19,-13,-16,-42,-23,-24,-25,-43,-44,-29,-26,-48,-11,-14,-17,-5,59,60,-7,-9,-41,-22,-30,-47,-58,-59,-12,-15,-18,60,-20,-6,-8,-4,-45,-40,-32,-37,-36,-31,85,-38,-27,87,-46,-49,88,-52,-53,-33,-57,90,91,-39,-28,-60,-50,-51,-35,-34,-54,]),'COMMA':([2,3,4,5,6,7,8,9,11,12,13,14,16,17,18,20,22,24,26,27,30,32,33,34,35,37,38,42,47,50,51,52,53,54,56,57,58,59,60,61,62,65,67,68,71,72,75,76,81,82,85,86,87,88,90,91,],[-7,-8,-9,-10,-19,-13,-16,30,-42,-23,-24,-25,-43,-44,-29,-26,-48,-11,-14,-17,-5,-8,-10,-7,-9,-41,-22,-30,-47,-58,-59,-12,-15,-18,-20,30,-8,-4,-45,-40,-32,-37,-36,-31,-38,-27,-46,-49,-33,-57,-39,-28,-60,-50,-35,-34,]),'PRODUCT':([6,11,12,13,14,16,17,18,20,22,42,47,50,51,60,62,65,67,68,71,72,75,76,81,82,85,86,87,88,90,91,],[25,36,-23,-24,-25,-43,-44,-29,-26,-48,-30,-47,-58,-59,-45,-32,-37,-36,-31,-38,-27,-46,-49,-33,-57,-39,-28,-60,-50,-35,-34,]),'POWER':([18,22,39,44,],[40,46,63,73,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'main':([0,45,],[1,74,]),'unit':([0,10,30,31,45,],[2,34,34,34,2,]),'structured_unit':([0,10,30,31,45,],[3,32,58,32,3,]),'structured_subunit':([0,10,30,31,45,],[4,35,35,35,4,]),'product_of_units':([0,6,10,11,25,28,29,30,31,36,45,],[5,24,33,37,52,55,56,5,33,61,5,]),'factor':([0,10,30,31,45,],[6,6,6,6,6,]),'division_product_of_units':([0,6,10,25,30,31,45,],[7,26,7,53,7,7,7,]),'inverse_unit':([0,6,10,25,30,31,45,],[8,27,8,54,8,8,8,]),'subunit':([0,10,30,31,45,],[9,9,57,9,9,]),'unit_expression':([0,6,10,11,15,25,28,29,30,31,36,45,],[11,11,11,11,38,11,11,11,11,11,11,11,]),'factor_fits':([0,10,30,31,45,],[12,12,12,12,12,]),'factor_float':([0,10,30,31,45,],[13,13,13,13,13,]),'factor_int':([0,10,30,31,45,],[14,14,14,14,14,]),'function':([0,6,10,11,15,25,28,29,30,31,36,45,],[16,16,16,16,16,16,16,16,16,16,16,16,]),'unit_with_power':([0,6,10,11,15,25,28,29,30,31,36,45,],[17,17,17,17,17,17,17,17,17,17,17,17,]),'signed_float':([0,10,30,31,45,49,66,],[20,20,20,20,20,79,79,]),'sign':([0,10,22,30,31,40,45,46,49,63,66,73,92,],[23,23,48,23,23,48,23,48,78,48,78,48,93,]),'signed_int':([18,39,40,41,44,66,],[42,62,67,70,72,84,]),'numeric_power':([22,40,46,63,73,],[47,68,75,81,86,]),'paren_expr':([49,66,],[77,77,]),'frac':([49,66,],[80,80,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> main","S'",1,None,None,None),
  ('main -> unit','main',1,'p_main','generic.py',138),
  ('main -> structured_unit','main',1,'p_main','generic.py',139),
  ('main -> structured_subunit','main',1,'p_main','generic.py',140),
  ('structured_subunit -> OPEN_PAREN structured_unit CLOSE_PAREN','structured_subunit',3,'p_structured_subunit','generic.py',151),
  ('structured_unit -> subunit COMMA','structured_unit',2,'p_structured_unit','generic.py',160),
  ('structured_unit -> subunit COMMA subunit','structured_unit',3,'p_structured_unit','generic.py',161),
  ('subunit -> unit','subunit',1,'p_subunit','generic.py',184),
  ('subunit -> structured_unit','subunit',1,'p_subunit','generic.py',185),
  ('subunit -> structured_subunit','subunit',1,'p_subunit','generic.py',186),
  ('unit -> product_of_units','unit',1,'p_unit','generic.py',192),
  ('unit -> factor product_of_units','unit',2,'p_unit','generic.py',193),
  ('unit -> factor PRODUCT product_of_units','unit',3,'p_unit','generic.py',194),
  ('unit -> division_product_of_units','unit',1,'p_unit','generic.py',195),
  ('unit -> factor division_product_of_units','unit',2,'p_unit','generic.py',196),
  ('unit -> factor PRODUCT division_product_of_units','unit',3,'p_unit','generic.py',197),
  ('unit -> inverse_unit','unit',1,'p_unit','generic.py',198),
  ('unit -> factor inverse_unit','unit',2,'p_unit','generic.py',199),
  ('unit -> factor PRODUCT inverse_unit','unit',3,'p_unit','generic.py',200),
  ('unit -> factor','unit',1,'p_unit','generic.py',201),
  ('division_product_of_units -> division_product_of_units DIVISION product_of_units','division_product_of_units',3,'p_division_product_of_units','generic.py',212),
  ('division_product_of_units -> product_of_units','division_product_of_units',1,'p_division_product_of_units','generic.py',213),
  ('inverse_unit -> DIVISION unit_expression','inverse_unit',2,'p_inverse_unit','generic.py',222),
  ('factor -> factor_fits','factor',1,'p_factor','generic.py',228),
  ('factor -> factor_float','factor',1,'p_factor','generic.py',229),
  ('factor -> factor_int','factor',1,'p_factor','generic.py',230),
  ('factor_float -> signed_float','factor_float',1,'p_factor_float','generic.py',236),
  ('factor_float -> signed_float UINT signed_int','factor_float',3,'p_factor_float','generic.py',237),
  ('factor_float -> signed_float UINT POWER numeric_power','factor_float',4,'p_factor_float','generic.py',238),
  ('factor_int -> UINT','factor_int',1,'p_factor_int','generic.py',251),
  ('factor_int -> UINT signed_int','factor_int',2,'p_factor_int','generic.py',252),
  ('factor_int -> UINT POWER numeric_power','factor_int',3,'p_factor_int','generic.py',253),
  ('factor_int -> UINT UINT signed_int','factor_int',3,'p_factor_int','generic.py',254),
  ('factor_int -> UINT UINT POWER numeric_power','factor_int',4,'p_factor_int','generic.py',255),
  ('factor_fits -> UINT POWER OPEN_PAREN signed_int CLOSE_PAREN','factor_fits',5,'p_factor_fits','generic.py',273),
  ('factor_fits -> UINT POWER OPEN_PAREN UINT CLOSE_PAREN','factor_fits',5,'p_factor_fits','generic.py',274),
  ('factor_fits -> UINT POWER signed_int','factor_fits',3,'p_factor_fits','generic.py',275),
  ('factor_fits -> UINT POWER UINT','factor_fits',3,'p_factor_fits','generic.py',276),
  ('factor_fits -> UINT SIGN UINT','factor_fits',3,'p_factor_fits','generic.py',277),
  ('factor_fits -> UINT OPEN_PAREN signed_int CLOSE_PAREN','factor_fits',4,'p_factor_fits','generic.py',278),
  ('product_of_units -> unit_expression PRODUCT product_of_units','product_of_units',3,'p_product_of_units','generic.py',297),
  ('product_of_units -> unit_expression product_of_units','product_of_units',2,'p_product_of_units','generic.py',298),
  ('product_of_units -> unit_expression','product_of_units',1,'p_product_of_units','generic.py',299),
  ('unit_expression -> function','unit_expression',1,'p_unit_expression','generic.py',310),
  ('unit_expression -> unit_with_power','unit_expression',1,'p_unit_expression','generic.py',311),
  ('unit_expression -> OPEN_PAREN product_of_units CLOSE_PAREN','unit_expression',3,'p_unit_expression','generic.py',312),
  ('unit_with_power -> UNIT POWER numeric_power','unit_with_power',3,'p_unit_with_power','generic.py',321),
  ('unit_with_power -> UNIT numeric_power','unit_with_power',2,'p_unit_with_power','generic.py',322),
  ('unit_with_power -> UNIT','unit_with_power',1,'p_unit_with_power','generic.py',323),
  ('numeric_power -> sign UINT','numeric_power',2,'p_numeric_power','generic.py',334),
  ('numeric_power -> OPEN_PAREN paren_expr CLOSE_PAREN','numeric_power',3,'p_numeric_power','generic.py',335),
  ('paren_expr -> sign UINT','paren_expr',2,'p_paren_expr','generic.py',344),
  ('paren_expr -> signed_float','paren_expr',1,'p_paren_expr','generic.py',345),
  ('paren_expr -> frac','paren_expr',1,'p_paren_expr','generic.py',346),
  ('frac -> sign UINT DIVISION sign UINT','frac',5,'p_frac','generic.py',355),
  ('sign -> SIGN','sign',1,'p_sign','generic.py',361),
  ('sign -> <empty>','sign',0,'p_sign','generic.py',362),
  ('signed_int -> SIGN UINT','signed_int',2,'p_signed_int','generic.py',371),
  ('signed_float -> sign UINT','signed_float',2,'p_signed_float','generic.py',377),
  ('signed_float -> sign UFLOAT','signed_float',2,'p_signed_float','generic.py',378),
  ('function -> FUNCNAME OPEN_PAREN main CLOSE_PAREN','function',4,'p_function','generic.py',384),
]
