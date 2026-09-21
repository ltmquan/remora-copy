"""Helpers for typed structural tensor lowering (embedded in generated code)."""

def _tensor_dot(a,b,shapes):
    ash,bsh=shapes
    if len(ash)==1 and len(bsh)>1:return torch.matmul(b,a)
    if len(bsh)==1:return torch.matmul(a,b)
    af,bf=tuple(ash[:-1]),tuple(bsh[:-1]);frame=max((af,bf),key=len)
    return (_prefix_expand(a,(ash[-1],),frame)*_prefix_expand(b,(bsh[-1],),frame)).sum(dim=-1)


def _tensor_add_initial(value,initial,result_shape,cell_shape):
    # The zero argument may carry its own lifted prefix frame.
    frame=tuple(result_shape[:len(result_shape)-len(cell_shape)]) if cell_shape else tuple(result_shape)
    return _prefix_expand(value,tuple(cell_shape),frame)+_prefix_expand(initial,tuple(cell_shape),frame)
